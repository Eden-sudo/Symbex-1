"""Scalar threshold boundaries and competing per-cell classes."""
import copy
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from compiler_core.export import boolean


@pytest.fixture(scope="module")
def models():
    path = Path(__file__).resolve().parents[1] / "benchmarks/conv2d/fomo_voc_person/model.py"
    spec = importlib.util.spec_from_file_location("competing_fomo_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


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


@pytest.mark.parametrize("classes", [3, 4])
@pytest.mark.parametrize("value", [0.312345, -0.713579])
@pytest.mark.parametrize("scalar", [float, np.float64, np.array])
def test_multiclass_nonzero_scalar(models, classes, value, scalar):
    model = models.BoolFOMO((4, 4, 8), (1, 1, 1), levels=2, classes=classes)
    with pytest.raises(ValueError, match="boolean export, per-cell head:.*one threshold per object class"):
        boolean.fold(model, scalar(value))


@pytest.mark.parametrize("classes", [2, 3, 4])
@pytest.mark.parametrize("zero", [0.0, -0.0, np.float64(0.0), np.array(-0.0)])
def test_scalar_zero(models, classes, zero):
    model = models.BoolFOMO((4, 4, 8), (1, 1, 1), levels=2, classes=classes)
    equal(boolean.fold(model), boolean.fold(model, zero))


@pytest.mark.parametrize("threshold", [0.0, 0.312345, -0.713579])
def test_binary_scalar_unchanged(models, threshold):
    model = models.BoolFOMO((4, 4, 8), (1, 1, 1), levels=2, classes=2)
    expected = copy.deepcopy(boolean.fold(model))
    bn = model.head_bn
    g = bn.weight.detach().double()
    c = (bn.bias.detach().double() - bn.running_mean.detach().double() * g /
         torch.sqrt(bn.running_var.detach().double() + bn.eps)).numpy()
    c[1] -= threshold
    expected["fc_bias"] = np.round(c * 2 ** boolean.FC_SHIFT).astype(np.int64)
    equal(boolean.fold(model, threshold), expected)
    equal(boolean.fold(model, [threshold]), expected)


def test_competing_classes(models, rng, tmp_path):
    assert shutil.which("gcc"), "needs gcc on PATH"
    model = models.BoolFOMO((8, 8, 16), (1, 1, 1), classes=4).eval()
    with torch.no_grad():
        for bn in model.modules():
            if isinstance(bn, torch.nn.BatchNorm2d):
                bn.running_mean.uniform_(-0.5, 0.5)
                bn.running_var.uniform_(0.7, 1.3)
                bn.weight.uniform_(0.8, 1.2)
                bn.bias.uniform_(-0.3, 0.3)
        model.head_bn.weight.mul_(0.3)
        model.head_bn.bias.copy_(model.head_bn.running_mean + 0.5 * torch.arange(4))
    images = rng.integers(0, 256, size=(4, 96, 96, 3), dtype=np.uint8)
    thresholds = [0.25, -0.5, 0.5]
    with torch.no_grad():
        logits = model(torch.from_numpy(images).permute(0, 3, 1, 2))
        adjusted = logits - logits.new_tensor([0.0, *thresholds]).view(1, 4, 1, 1)
        top = adjusted.topk(2, dim=1).values
        assert torch.all(top[:, 0] > top[:, 1])
        expected = adjusted.argmax(1).numpy().astype(np.uint8)
        assert np.any(expected != logits.argmax(1).numpy())
    assert expected.shape == (len(images), 12, 12)
    assert len(np.unique(expected)) >= 3
    net = boolean.fold(model, thresholds)
    actual = boolean.simulate(net, images, device="cpu", return_bits=True)
    np.testing.assert_array_equal(actual, expected)
    header = tmp_path / "model_symbex.h"
    boolean.write_header(net, header, in_size=96)
    lib = Path(__file__).resolve().parents[1] / "Symbex1_PIO/lib/lib_symbex"
    checksums = [boolean._grid_checksum(grid) for grid in expected]
    equal_count, total, message = boolean.validate_c(header, lib, images, checksums)
    assert equal_count == total == len(images), message
