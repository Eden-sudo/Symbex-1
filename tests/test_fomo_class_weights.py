"""Cell-frequency weights for multiclass FOMO, without data or torch."""
import builtins
import importlib.util
import os
from fractions import Fraction

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOMO_DIR = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person")


@pytest.fixture(scope="module")
def weights():
    spec = importlib.util.spec_from_file_location("fomo_class_weights", os.path.join(FOMO_DIR, "fomo_class_weights.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.class_weights


def grid(counts):
    return np.repeat(np.arange(len(counts)), counts).reshape(-1, 12, 12)


def test_class_order_and_dtype(weights):
    result = weights(grid([96, 32, 16]), 2)
    np.testing.assert_array_equal(result, [1, 3, 6])
    assert result.shape == (3,)
    assert result.dtype == np.float32


def test_lower_limit(weights):
    np.testing.assert_array_equal(weights(grid([24, 48, 72]), 2), [1, 1, 1])


def test_cap_and_missing_class(weights):
    y = grid([140, 3, 1, 0])
    np.testing.assert_array_equal(weights(y, 3, 10), [1, 10, 10, 10])
    np.testing.assert_allclose(weights(y, 3), [1, 140 / 3, 50, 50], rtol=1e-6)


@pytest.mark.parametrize("counts, expected", [([108, 36], 3), ([143, 1], 50), ([36, 108], 1), ([144, 0], 50)])
def test_one_class(weights, counts, expected):
    np.testing.assert_array_equal(weights(grid(counts), 1), [1, expected])


def test_no_background_or_data(weights):
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(weights(grid([0, 144, 0]), 2), [1, 1, 50])
        np.testing.assert_array_equal(weights(np.empty((0, 12, 12), dtype=np.uint8), 2), [1, 50, 50])


def test_maximum(weights):
    y = grid([140, 3, 1, 0])
    np.testing.assert_array_equal(weights(y, 3, 1), [1, 1, 1, 1])
    np.testing.assert_array_equal(weights(y, np.int64(3), np.float64(2.3)), np.array([1, 2.3, 2.3, 2.3], np.float32))
    np.testing.assert_array_equal(weights(y, 3, Fraction(3, 2)), [1, 1.5, 1.5, 1.5])


def test_aggregate_and_batch_order(weights):
    y = np.concatenate([grid([108, 36]), grid([72, 72])])
    expected = [1, 180 / 108]
    np.testing.assert_allclose(weights(y, 1), expected)
    np.testing.assert_allclose(weights(y[::-1], 1), expected)
    np.testing.assert_allclose(weights(np.tile(y, (2, 1, 1)), 1), expected)


def test_readonly_noncontiguous_and_independent(weights):
    y = grid([96, 32, 16]).astype(np.int16)[:, :, ::-1]
    y.flags.writeable = False
    before = y.copy()
    first = weights(y, 2)
    second = weights(y, 2)
    np.testing.assert_array_equal(first, weights(before, 2))
    np.testing.assert_array_equal(y, before)
    first[0] = 9
    assert second[0] == 1


@pytest.mark.parametrize("n", [0, -1, 1.0, True, np.bool_(False)])
def test_invalid_classes(weights, n):
    with pytest.raises(ValueError, match="n_classes"):
        weights(grid([144]), n)


@pytest.mark.parametrize("maximum", [0, -1, float('nan'), float('inf'), -float('inf'), 1e100, 10**100, '2', True])
def test_invalid_maximum(weights, maximum):
    with pytest.raises(ValueError, match="max_weight"):
        weights(grid([144]), 1, maximum)


@pytest.mark.parametrize("y", [
    np.zeros((12, 12), dtype=int), np.zeros((1, 1, 12, 12), dtype=int),
    np.zeros((1, 12, 11), dtype=int), grid([143, 1]) - 1, grid([143, 0, 1]),
    np.zeros((1, 12, 12), dtype=bool), np.zeros((1, 12, 12), dtype=float),
    np.zeros((1, 12, 12), dtype=complex), np.full((1, 12, 12), '0'),
    np.zeros((1, 12, 12), dtype=object),
])
def test_invalid_y(weights, y):
    with pytest.raises(ValueError, match="y"):
        weights(y, 1)


def test_no_torch_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("fomo_class_weights_guarded", os.path.join(FOMO_DIR, "fomo_class_weights.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    np.testing.assert_array_equal(mod.class_weights(grid([108, 36]), 1), [1, 3])
