"""Ternary method: gradients reach W and the zeroing threshold, and the folded integer network is bit-exact
against the numpy reference, PyTorch and the compiled C kernel (native and SWAR popcount)."""
import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from compiler_core.export.dense import analyze
from compiler_core.export.ternary import fold, validate_c, write_header
from compiler_core.methods import apply_plan
from compiler_core.methods.dense.ternary import DenseTernary
from compiler_core.validate.ternary import verify_bit_exact

LIB_SYMBEX = os.path.join(os.path.dirname(__file__), "..", "Symbex1_PIO", "lib", "lib_symbex")


def test_gradient_reaches_weights_and_zeroing_threshold():
    torch.manual_seed(0)
    layer = DenseTernary(70, 33)
    layer(torch.randn(8, 70)).sum().backward()
    assert layer.weight.grad.abs().sum() > 0
    assert layer.delta_ratio.grad is not None and layer.delta_ratio.grad.abs() > 0


def _random_ternary_mlp(seed):
    """70-45-33-10: widths that need word padding; random BN statistics with negative gammas (row negation)."""
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Flatten(), nn.Linear(70, 45, bias=False), nn.BatchNorm1d(45), nn.ReLU(),
                      nn.Linear(45, 33, bias=False), nn.BatchNorm1d(33), nn.ReLU(), nn.Linear(33, 10))
    m = apply_plan(m, {"1": "bin", "4": "bin", "7": "bin"}, bin_method="ternary")
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.running_mean.normal_()
            mod.running_var.uniform_(0.5, 2)
            mod.weight.data.normal_()
        if hasattr(mod, "beta"):
            mod.beta.data.normal_(0, 0.5)
    return m.eval()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_folded_ternary_network_is_bit_exact_against_numpy_pytorch_and_c(seed, tmp_path):
    m = _random_ternary_mlp(seed)
    xu8 = np.random.default_rng(seed).integers(0, 256, (300, 70)).astype(np.uint8)
    xs = torch.from_numpy(xu8).float() / 255.0
    layers = fold(analyze(m, xs[:2], cls=DenseTernary))

    r = verify_bit_exact(layers, xu8)
    assert r["equal"], r["raw_equal"]
    with torch.no_grad():
        assert np.array_equal(m(xs).argmax(1).numpy(), r["_pred"])

    if shutil.which("gcc") is None:
        pytest.skip("gcc not found: C exactness skipped")
    header = str(tmp_path / "tern.h")
    write_header(layers, header)
    for flags in ([], ["-DSYMBEX_POPCOUNT_FORCE_SWAR"]):
        equal, n, err = validate_c(header, LIB_SYMBEX, xu8, r["_pred"], flags=flags)
        assert err == "" and equal == n == len(xu8), err
