import ast
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/conv2d/fomo_voc_person/fomo_synthetic.py'
spec = importlib.util.spec_from_file_location('fomo_synthetic_no_overlap', SCRIPT)
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)


def assert_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].dtype == b[key].dtype


@pytest.mark.parametrize('seed', [0, 7, 42])
def test_centers_and_determinism(seed):
    data = synth.make(512, 3, seed)
    assert_equal(data, synth.make(512, 3, seed))
    expected = np.zeros_like(data['y'])
    palette = synth._palette(3)
    assert len(data['centers']) > 0
    for (cx, cy), owner, cls in zip(data['centers'], data['image_of_center'], data['center_class']):
        gx, gy = int(cx // synth.CELL), int(cy // synth.CELL)
        assert expected[owner, gy, gx] == 0
        expected[owner, gy, gx] = cls
        np.testing.assert_array_equal(data['x'][owner, int(cy), int(cx)], palette[cls - 1])
    np.testing.assert_array_equal(data['y'], expected)


class FakeRng:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def integers(self, low, high, size=None, dtype=None):
        if size is not None:
            return np.zeros(size, dtype=dtype)
        self.calls += 1
        value = next(self.values)
        assert low <= value < high
        return value


def fake_make(monkeypatch, values, images=1):
    rng = FakeRng(values)
    monkeypatch.setattr(synth.np.random, 'default_rng', lambda seed: rng)
    data = synth.make(images, 2, 0)
    assert next(rng.values, None) is None
    return data, rng.calls


def test_retry_protects_entire_cell(monkeypatch):
    data, calls = fake_make(monkeypatch, [2, 10, 1, 1, 1, 10, 2, 7, 1, 24, 24])
    assert calls == 11
    np.testing.assert_array_equal(data['centers'], [[6, 6], [29, 29]])
    np.testing.assert_array_equal(data['center_class'], [1, 2])
    palette = synth._palette(2)
    np.testing.assert_array_equal(data['x'][0, 6, 6], palette[0])
    np.testing.assert_array_equal(data['x'][0, 1, 12], [0, 0, 0])
    np.testing.assert_array_equal(data['x'][0, 29, 29], palette[1])


def test_drop_then_continue_and_reset(monkeypatch):
    values = [3, 10, 1, 1, 1, 10, 2] + [0, 0] * 32 + [10, 2, 24, 24]
    values += [1, 10, 2, 1, 1]
    data, calls = fake_make(monkeypatch, values, images=2)
    assert calls == len(values)
    np.testing.assert_array_equal(data['centers'], [[6, 6], [29, 29], [6, 6]])
    np.testing.assert_array_equal(data['image_of_center'], [0, 0, 1])
    np.testing.assert_array_equal(data['center_class'], [1, 2, 2])
    assert np.count_nonzero(data['y'][0]) == 2
    np.testing.assert_array_equal(data['x'][0, 0, 0], [0, 0, 0])


@pytest.mark.parametrize('rect,intersects', [
    ((0, 8, 8), False), ((1, 8, 8), True),
    ((16, 8, 8), False), ((15, 8, 8), True),
    ((8, 0, 8), False), ((8, 1, 8), True),
    ((8, 16, 8), False), ((8, 15, 8), True),
])
def test_cell_edges(rect, intersects):
    assert synth._covers_cell(*rect, 1, 1) == intersects


def test_overlap_outside_cell_and_image_edges(monkeypatch):
    data, _ = fake_make(monkeypatch, [3, 10, 1, 0, 0, 10, 2, 8, 8, 11, 1, 85, 85])
    np.testing.assert_array_equal(data['centers'], [[5, 5], [13, 13], [90.5, 90.5]])
    np.testing.assert_array_equal(data['center_class'], [1, 2, 1])
    palette = synth._palette(2)
    np.testing.assert_array_equal(data['x'][0, 8, 8], palette[1])
    np.testing.assert_array_equal(data['x'][0, 5, 5], palette[0])
    np.testing.assert_array_equal(data['x'][0, 95, 95], palette[0])


def test_model_dimensions_are_literal_and_equal():
    tree = ast.parse((SCRIPT.parent / 'model.py').read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names = target.elts if isinstance(target, ast.Tuple) else [target]
                keys = [name.id for name in names if isinstance(name, ast.Name)]
                if any(key in ('SIZE', 'CELL', 'GRID') for key in keys):
                    try:
                        values = ast.literal_eval(node.value)
                    except (ValueError, TypeError) as error:
                        pytest.fail(f'model dimensions must be literals: {error}')
                    if not isinstance(target, ast.Tuple):
                        values = (values,)
                    found.update(zip(keys, values))
    for name in ('SIZE', 'CELL', 'GRID'):
        assert name in found, f'model dimension {name} is missing'
        assert found[name] == getattr(synth, name)


def test_main_output_paths(monkeypatch, tmp_path):
    assert Path(synth.HERE) == SCRIPT.parent
    benchmark, cwd = tmp_path / 'benchmark', tmp_path / 'cwd'
    benchmark.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(synth, 'HERE', str(benchmark))
    args = ['--images', '3', '--classes', '2', '--seed', '4', '--out']
    synth.main(args + ['data/fomo_train.npz'])
    assert not (cwd / 'data/fomo_train.npz').exists()
    absolute = tmp_path / 'absolute.npz'
    synth.main(args + [str(absolute)])
    for path in (benchmark / 'data/fomo_train.npz', absolute):
        with np.load(path, allow_pickle=False) as loaded:
            assert_equal(synth.make(3, 2, 4), loaded)
    assert Path(os.getcwd()) == cwd
