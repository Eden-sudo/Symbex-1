"""Per-class thresholds folded into a FOMO cell head, and the unchanged scalar rule."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from compiler_core.export import boolean


@pytest.fixture(scope="module")
def models():
    path = Path(__file__).resolve().parents[1] / "benchmarks/conv2d/fomo_voc_person/model.py"
    spec = importlib.util.spec_from_file_location("threshold_fomo_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fomo(models, classes=3):
    model = models.BoolFOMO(channels=(4, 4, 8), blocks=(1, 1, 1), levels=2, classes=classes).eval()
    with torch.no_grad():
        model.head_bn.running_mean.copy_(torch.linspace(-0.7, 0.8, classes))
        model.head_bn.running_var.copy_(torch.linspace(0.6, 1.4, classes))
        model.head_bn.weight.copy_(torch.linspace(0.8, 1.2, classes))
        model.head_bn.bias.copy_(torch.linspace(-0.3, 0.4, classes))
    return model


def bias(model):
    bn = model.head_bn
    g = bn.weight.detach().double()
    return (bn.bias.detach().double() - bn.running_mean.detach().double() * g /
            torch.sqrt(bn.running_var.detach().double() + bn.eps)).numpy()


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("threshold", [0.0, 0.312345, -0.713579])
def test_scalar_rule(models, classes, threshold):
    model = fomo(models, classes)
    baseline = boolean.fold(model)
    equal(baseline, boolean.fold(model, 0.0))
    expected = copy.deepcopy(baseline)
    c = bias(model)
    if threshold:
        c = c.copy()
        c[1] -= threshold
    expected["fc_bias"] = np.round(c * 2 ** boolean.FC_SHIFT).astype(np.int64)
    actual = boolean.fold(model, threshold)
    equal(actual, expected)
    if classes == 3:
        assert actual["fc_bias"][2] == baseline["fc_bias"][2]


@pytest.mark.parametrize("container", [list, tuple, np.array])
@pytest.mark.parametrize("offsets", [[0.75, -1.25], [0.0, 0.0]])
def test_sequence_rule(models, container, offsets):
    model = fomo(models)
    thresholds = container(offsets)
    original = copy.deepcopy(thresholds)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    expected = copy.deepcopy(boolean.fold(model))
    c = bias(model)
    c[1:] -= np.asarray(offsets)
    expected["fc_bias"] = np.round(c * 2 ** boolean.FC_SHIFT).astype(np.int64)
    equal(boolean.fold(model, thresholds), expected)
    equal(boolean.fold(model, thresholds), expected)
    np.testing.assert_array_equal(thresholds, original)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_binary_sequence_matches_scalar(models):
    model = fomo(models, 2)
    equal(boolean.fold(model, [0.312345]), boolean.fold(model, 0.312345))


def grids(model, images, offsets):
    with torch.no_grad():
        logits = model(torch.from_numpy(images).permute(0, 3, 1, 2))
        adjusted = logits - torch.tensor([0.0, *offsets]).view(1, -1, 1, 1)
        return adjusted.argmax(1).numpy()


def test_simulation_matches_logits(models, rng):
    model = fomo(models)
    images = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)
    baseline = grids(model, images, [0.0, 0.0])
    changed = False
    for offsets in ([12.75, -12.25], [-12.25, 12.75]):
        expected = grids(model, images, offsets)
        actual = boolean.simulate(boolean.fold(model, offsets), images, device="cpu", return_bits=True)
        np.testing.assert_array_equal(actual, expected)
        changed |= np.any(expected != baseline)
    assert changed


@pytest.mark.parametrize("biases, offsets, winner", [([0.0, 0.5, -0.5], [0.5, -0.5], 0),
                                                     ([-1.0, 0.5, -0.5], [0.5, -0.5], 1)])
def test_ties(models, rng, biases, offsets, winner):
    model = fomo(models)
    with torch.no_grad():
        model.head_bn.weight.zero_()
        model.head_bn.bias.copy_(torch.tensor(biases))
    images = rng.integers(0, 256, size=(2, 16, 16, 3), dtype=np.uint8)
    expected = grids(model, images, offsets)
    assert np.all(expected == winner)
    np.testing.assert_array_equal(boolean.simulate(boolean.fold(model, offsets), images, device="cpu",
                                                  return_bits=True), expected)


@pytest.mark.parametrize("offsets", [[], [1.0], [1.0, 2.0, 3.0]])
def test_bad_length(models, offsets):
    with pytest.raises(ValueError, match=f"threshold length {len(offsets)}, expected 2"):
        boolean.fold(fomo(models), offsets)


def test_nested_and_overflow(models):
    model = fomo(models)
    with pytest.raises(ValueError, match="one-dimensional"):
        boolean.fold(model, [[1.0], [2.0]])
    with pytest.raises(ValueError, match="classifier bias does not fit"):
        boolean.fold(model, [100000.0, 0.0])


@pytest.mark.parametrize("head", ["binary", "pool2"])
def test_no_cell_head_ignores_thresholds(models, head):
    model = models._r8.BoolResNet8(channels=(4, 4, 8), blocks=(1, 1, 1), levels=2, classes=3, head=head).eval()
    expected = boolean.fold(model)
    for threshold in (0.5, [1.0], np.array([1.0, 2.0, 3.0, 4.0])):
        equal(boolean.fold(model, threshold), expected)
