"""Ternary method: gradients reach W and the zeroing threshold, and the folded integer network is bit-exact
against the numpy reference, PyTorch and the compiled C kernel (native and SWAR popcount)."""
import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from compiler_core.export.dense import analizar
from compiler_core.export.ternary import escribir_header, plegar, validar_c
from compiler_core.methods import apply_plan
from compiler_core.methods.linear.ternary import LinearTernary
from compiler_core.validate.ternary import verificar_bit_exacto

LIB_SYMBEX = os.path.join(os.path.dirname(__file__), "..", "Symbex1_PIO", "lib", "lib_symbex")


def test_gradient_reaches_weights_and_zeroing_threshold():
    torch.manual_seed(0)
    capa = LinearTernary(70, 33)
    capa(torch.randn(8, 70)).sum().backward()
    assert capa.weight.grad.abs().sum() > 0
    assert capa.delta_ratio.grad is not None and capa.delta_ratio.grad.abs() > 0


def _random_ternary_mlp(seed):
    """70-45-33-10: widths that need word padding; random BN statistics with negative gammas (row negation)."""
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Flatten(), nn.Linear(70, 45, bias=False), nn.BatchNorm1d(45), nn.ReLU(),
                      nn.Linear(45, 33, bias=False), nn.BatchNorm1d(33), nn.ReLU(), nn.Linear(33, 10))
    m = apply_plan(m, {"1": "bin", "4": "bin", "7": "bin"}, metodo_bin="ternary")
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
    capas = plegar(analizar(m, xs[:2], clase=LinearTernary))

    r = verificar_bit_exacto(capas, xu8)
    assert r["iguales"], r["raw_iguales"]
    with torch.no_grad():
        assert np.array_equal(m(xs).argmax(1).numpy(), r["_pred"])

    if shutil.which("gcc") is None:
        pytest.skip("gcc not found: C exactness skipped")
    header = str(tmp_path / "tern.h")
    escribir_header(capas, header)
    for flags in ([], ["-DSYMBEX_POPCOUNT_FORZAR_SWAR"]):
        iguales, n, err = validar_c(header, LIB_SYMBEX, xu8, r["_pred"], flags=flags)
        assert err == "" and iguales == n == len(xu8), err
