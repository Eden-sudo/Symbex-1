import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/conv2d/fomo_voc_person/fomo_synthetic.py'
spec = importlib.util.spec_from_file_location('fomo_synthetic', SCRIPT)
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)


def assert_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].dtype == b[key].dtype


def test_determinism_and_global_rng():
    np.random.seed(123)
    state = np.random.get_state()
    a = synth.make(8, 3, 42)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    np.random.seed(456)
    assert_equal(a, synth.make(8, 3, 42))
    assert not np.array_equal(a['x'], synth.make(8, 3, 43)['x'])


@pytest.mark.parametrize('n,k,maximum', [(0, 1, 3), (3, 255, 0), (30, 3, 3)])
def test_contract(n, k, maximum):
    data = synth.make(n, k, 12, maximum)
    assert set(data) == {'x', 'y', 'centers', 'image_of_center', 'center_class', 'class_names'}
    assert data['x'].shape == (n, 96, 96, 3) and data['x'].dtype == np.uint8
    assert data['y'].shape == (n, 12, 12) and data['y'].dtype == np.uint8
    centers, owners, classes = (data[key] for key in ('centers', 'image_of_center', 'center_class'))
    assert centers.shape == (len(owners), 2) and centers.dtype == np.float32
    assert owners.dtype == classes.dtype == np.int32 and len(owners) == len(classes)
    assert data['class_names'].dtype.kind == 'U'
    assert data['class_names'].tolist() == [f'class_{i}' for i in range(1, k + 1)]
    expected = np.zeros_like(data['y'])
    for (cx, cy), owner, cls in zip(centers, owners, classes):
        assert 0 <= owner < n and 1 <= cls <= k and 0 <= cx < 96 and 0 <= cy < 96
        gx, gy = min(int(cx // 8), 11), min(int(cy // 8), 11)
        assert expected[owner, gy, gx] == 0
        expected[owner, gy, gx] = cls
    np.testing.assert_array_equal(expected, data['y'])
    assert np.all(np.bincount(owners, minlength=n) <= maximum)


def test_single_shapes():
    data = synth.make(80, 3, 7, 1)
    palette = synth._palette(3)
    assert 0 < len(data['centers']) < 80
    for image, pixels in enumerate(data['x']):
        indices = np.flatnonzero(data['image_of_center'] == image)
        mask = np.any(pixels >= 64, axis=2)
        if not len(indices):
            assert not mask.any()
            continue
        index = indices[0]
        ys, xs = np.where(mask)
        side = xs.max() - xs.min() + 1
        assert 10 <= side <= 20 and ys.max() - ys.min() + 1 == side
        assert mask.sum() == side * side
        assert np.all(pixels[mask] == palette[data['center_class'][index] - 1])
        assert np.all(pixels[~mask] <= 31)
        np.testing.assert_array_equal(data['centers'][index], [xs.min() + side / 2, ys.min() + side / 2])


def test_selection():
    objects = [(0, 0, 10, 1), (1, 1, 10, 2), (0, 0, 10, 3),
               (3, 3, 10, 1), (2, 2, 10, 2), (81, 81, 20, 1)]
    assert synth._select_objects(objects) == [objects[0], objects[3], objects[5]]
    assert synth._select_objects([(1, 1, 10, 1), (0, 0, 10, 2)]) == [(0, 0, 10, 2)]


def test_collision_integration(monkeypatch):
    class FakeRng:
        def __init__(self):
            self.values = iter([2, 10, 1, 1, 1, 10, 2, 0, 0])

        def integers(self, low, high, size=None, dtype=None):
            if size is not None:
                return np.zeros(size, dtype=dtype)
            return next(self.values)

    monkeypatch.setattr(synth.np.random, 'default_rng', lambda seed: FakeRng())
    data = synth.make(1, 2, 0)
    np.testing.assert_array_equal(data['centers'], [[5, 5]])
    np.testing.assert_array_equal(data['center_class'], [2])
    assert data['y'][0, 0, 0] == 2
    assert not np.any(np.all(data['x'][0] == synth._palette(2)[0], axis=2))


def test_palette():
    palette = synth._palette(255)
    np.testing.assert_array_equal(palette[:3], synth._palette(3))
    assert np.all(palette > 31)
    delta = palette.astype(np.int32)[:, None] - palette.astype(np.int32)[None, :]
    dist = np.sum(delta ** 2, axis=2)
    np.fill_diagonal(dist, 999999)
    assert dist.min() >= 30 ** 2
    assert dist[:3, :3].min() >= 180 ** 2


@pytest.mark.parametrize('index,name', [(0, 'n_images'), (1, 'n_classes'), (2, 'seed'), (3, 'max_objects')])
@pytest.mark.parametrize('value', [True, 1.0, '1', None])
def test_invalid_types(index, name, value):
    args = [1, 1, 0, 3]
    args[index] = value
    with pytest.raises(TypeError, match=name):
        synth.make(*args)


@pytest.mark.parametrize('index,name,value', [(0, 'n_images', -1), (0, 'n_images', 2**31),
    (1, 'n_classes', 0), (1, 'n_classes', 256), (2, 'seed', -1),
    (3, 'max_objects', -1), (3, 'max_objects', 2**63 - 1)])
def test_invalid_ranges(index, name, value):
    args = [1, 1, 0, 3]
    args[index] = value
    with pytest.raises(ValueError, match=name):
        synth.make(*args)


def test_save(tmp_path):
    path = tmp_path / 'nested' / 'dataset'
    for data in (synth.make(np.int64(2), np.int64(1), np.int64(1)), synth.make(0, 1, 0)):
        synth.save(path, data)
        with np.load(path, allow_pickle=False) as loaded:
            assert_equal(data, loaded)
    with pytest.raises(OSError):
        synth.save(tmp_path, data)


def test_cli_and_import(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp_path, capture_output=True, text=True)

    assert run('--help').returncode == 0
    args = ['--images', '3', '--classes', '2', '--seed', '4', '--out', 'data/out']
    assert run(*args).returncode == 0
    with np.load(tmp_path / 'data/out', allow_pickle=False) as loaded:
        assert_equal(synth.make(3, 2, 4), loaded)
    for images in ('bad', '-1'):
        result = run('--images', images, '--classes', '2', '--seed', '4', '--out', 'invalid')
        assert result.returncode == 2 and not (tmp_path / 'invalid').exists()
        assert 'images' in result.stderr
    code = (f"import runpy, sys; m = runpy.run_path({str(SCRIPT)!r}); m['make'](1, 2, 0); "
            "assert not any(k == p or k.startswith(p + '.') for k in sys.modules for p in ('torch', 'torchvision', 'PIL'))")
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, check=True)
