"""Multiclass FOMO grids agree between PyTorch, the integer simulator and C."""
import importlib.util
import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from compiler_core.export import boolean

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_SYMBEX = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")


@pytest.fixture(scope="module")
def fomo():
    path = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person", "model.py")
    spec = importlib.util.spec_from_file_location("fomo_mc_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cpu_threads():
    assert shutil.which("gcc"), "needs gcc on PATH"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def _make_model(fomo):
    torch.manual_seed(0)
    model = fomo.BoolFOMO(channels=(8, 8, 16), blocks=(1, 1, 1), classes=4)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.copy_(torch.randn(mod.running_mean.shape, generator=gen) * 3)
                mod.running_var.copy_(torch.rand(mod.running_var.shape, generator=gen) * 4 + 0.1)
    return model.eval()


def _assert_c_exact(net, x, expected_grids, tmp_path):
    header = str(tmp_path / "model_symbex.h")
    boolean.write_header(net, header, in_size=96)
    checksums = np.array([boolean._grid_checksum(g) for g in expected_grids], dtype=np.int64)
    equal, total, message = boolean.validate_c(header, LIB_SYMBEX, x, checksums)
    assert message == "" and equal == total == 32, (equal, total, message)


def test_multiclass_fomo_export_is_bit_exact(fomo, cpu_threads, tmp_path):
    model = _make_model(fomo)
    x = np.random.default_rng(0).integers(0, 256, size=(32, 96, 96, 3), dtype=np.uint8)
    with torch.no_grad():
        bn = model.head_bn
        bn.weight.copy_(torch.sqrt(bn.running_var + bn.eps))
        # Each head dot product sums 16 +/-1 products and is even. Distinct half-integer offsets
        # prevent ties, which float32 BatchNorm rounding could resolve differently from integer first-maximum.
        bn.bias.copy_(bn.running_mean + 0.5 * torch.arange(4))
        expected = model(torch.from_numpy(x).permute(0, 3, 1, 2)).argmax(1).numpy()
    net = boolean.fold(model)
    got = boolean.simulate(net, x, return_bits=True)
    assert expected.shape == got.shape == (32, 12, 12)
    assert np.unique(expected).size >= 3
    np.testing.assert_array_equal(got, expected)
    _assert_c_exact(net, x, expected, tmp_path)


def test_multiclass_fomo_ties_choose_first_maximum(fomo, cpu_threads, tmp_path):
    model = _make_model(fomo)
    x = np.random.default_rng(0).integers(0, 256, size=(32, 96, 96, 3), dtype=np.uint8)
    with torch.no_grad():
        model.head_bn.weight.zero_()
        model.head_bn.bias.copy_(torch.tensor([0., 1., 1., 1.]))
        expected = model(torch.from_numpy(x).permute(0, 3, 1, 2)).argmax(1).numpy()
    net = boolean.fold(model)
    got = boolean.simulate(net, x, return_bits=True)
    assert expected.shape == got.shape == (32, 12, 12)
    ones = np.ones((32, 12, 12), dtype=np.uint8)
    np.testing.assert_array_equal(expected, ones)
    np.testing.assert_array_equal(got, ones)
    _assert_c_exact(net, x, ones, tmp_path)
