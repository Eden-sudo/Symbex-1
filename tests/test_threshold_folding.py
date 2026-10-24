"""Threshold folding: BatchNorm (incl. negative and zero gamma), bias, input normalization, ReLU and the next
layer's sign activation folded into ONE integer threshold per neuron (export/dense.plegar), checked
bit-for-bit against the float PyTorch model; plus the fixed-point BN folding of the conv path
(export/plain._fold_bn, validate/plain._aplicar_rprelu)."""
import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from compiler_core.export import dense
from compiler_core.export.plain import _fold_bn
from compiler_core.methods.linear.plain import LinearPlain
from compiler_core.validate.plain import _aplicar_rprelu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_SYMBEX = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")

MEAN, STD = (0.4, 0.6), (0.3, 0.3)
C, H, W = 2, 5, 5                     # 50 inputs: not a multiple of 32


def _randomize_bn(bn, gen, zero_gamma_index=None, stats=True):
    n = bn.num_features
    with torch.no_grad():
        gamma = torch.randn(n, generator=gen).abs() + 0.3
        gamma[: n // 3] = -gamma[: n // 3]                  # a third of the neurons: negative gamma
        if zero_gamma_index is not None:
            gamma[zero_gamma_index] = 0.0
        bn.weight.copy_(gamma)
        bn.bias.copy_(torch.randn(n, generator=gen) * (0.5 if not stats else 1.0))
        if stats:
            bn.running_mean.copy_(torch.randn(n, generator=gen) * 3)
            bn.running_var.copy_(torch.rand(n, generator=gen) * 4 + 0.1)


def _normalized(x_u8, dtype=torch.float32):
    x = torch.from_numpy(x_u8.astype(np.float64)).to(dtype).view(-1, C, H, W) / 255.0
    return (x - torch.tensor(MEAN, dtype=dtype).view(1, C, 1, 1)) / torch.tensor(STD, dtype=dtype).view(1, C, 1, 1)


def _build_mlp(relu, bias, gen, calib_u8, hidden=(37, 20), classes=6):
    """Random MLP whose BatchNorm statistics come from `calib_u8`, so the folded thresholds fall INSIDE the
    range of the raw scores (random statistics would saturate most neurons to a constant bit)."""
    sizes = [C * H * W, *hidden, classes]
    layers = [nn.Flatten()]
    for i in range(len(sizes) - 1):
        fl = nn.Linear(sizes[i], sizes[i + 1], bias=bias)
        layers.append(LinearPlain.from_float(fl, binarize_input=(i > 0)))
        bn = nn.BatchNorm1d(sizes[i + 1], momentum=None)    # cumulative average: one batch = its statistics
        _randomize_bn(bn, gen, zero_gamma_index=0 if i == 0 else None, stats=False)
        layers.append(bn)
        if relu and i < len(sizes) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LinearPlain) and not isinstance(m.rsign, nn.Identity):
                # beta > 0 except a few: after a ReLU any beta <= 0 is a constant 1 bit
                m.rsign.beta.copy_(torch.randn(m.rsign.beta.shape, generator=gen).abs() * 0.5 + 0.05)
                m.rsign.beta[0, :3] = -m.rsign.beta[0, :3].abs() - 0.05   # beta <= 0 (always-1 path with ReLU)
                m.weight[0, :5] = 0.0                                      # exact zeros: 0 -> +1
        model.train()
        model(_normalized(calib_u8))
    return model.eval()


def _float_reference(model, x_u8):
    """Hidden +-1 activations (as each next layer receives them) and logits, in float64."""
    model = model.double()
    x = _normalized(x_u8, torch.float64)
    bits, hooks = [], []
    for m in model.modules():
        if isinstance(m, LinearPlain) and not isinstance(m.rsign, nn.Identity):
            hooks.append(m.rsign.register_forward_hook(lambda mod, e, s: bits.append(s.numpy().astype(np.int64))))
    with torch.no_grad():
        logits = model(x).numpy()
    for h in hooks:
        h.remove()
    return bits, logits


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_folded_integer_thresholds_reproduce_float_hidden_bits_exactly(rng, relu, bias):
    gen = torch.Generator().manual_seed(7 + 2 * relu + bias)
    model = _build_mlp(relu, bias, gen, rng.integers(0, 256, size=(256, C * H * W), dtype=np.uint8))
    x_u8 = rng.integers(0, 256, size=(1000, C * H * W), dtype=np.uint8)
    example = torch.zeros(2, C, H, W)
    capas = dense.plegar(dense.analizar(model, example), mean=MEAN, std=STD)

    ref_bits, ref_logits = _float_reference(model, x_u8)
    pred, sim_bits = dense.simular(capas, x_u8, devolver_bits=True)

    assert len(sim_bits) == len(ref_bits) == 2
    for layer, (got, want) in enumerate(zip(sim_bits, ref_bits)):
        varying = (want.min(0) != want.max(0)).mean()
        assert varying > 0.5, f"hidden layer {layer}: only {varying:.0%} of the neurons change: weak test data"
        mismatches = int((got != want).sum())
        assert mismatches == 0, f"hidden layer {layer}: {mismatches} bits differ from the float model"

    # logits are fixed point: compare the argmax where the float top-2 margin is above the rounding error
    top2 = np.sort(ref_logits, axis=1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 1e-3
    assert clear.mean() > 0.9
    np.testing.assert_array_equal(pred[clear], ref_logits.argmax(1)[clear])


def test_negative_gamma_rows_are_negated_and_threshold_flipped(rng):
    gen = torch.Generator().manual_seed(3)
    model = _build_mlp(relu=False, bias=False, gen=gen,
                       calib_u8=rng.integers(0, 256, size=(64, C * H * W), dtype=np.uint8))
    grupos = dense.analizar(model, torch.zeros(2, C, H, W))
    capas = dense.plegar(grupos, mean=MEAN, std=STD)
    capa1, bn1 = grupos[1][0], grupos[1][1]
    original = np.where(capa1.weight.detach().numpy() >= 0, 1, -1)
    negative = (bn1.weight.detach().numpy() < 0)
    assert negative.any() and (~negative).any()
    np.testing.assert_array_equal(capas[1]["signos"][negative], -original[negative])
    np.testing.assert_array_equal(capas[1]["signos"][~negative], original[~negative])


def test_plegar_rejects_a_hidden_layer_that_does_not_binarize_its_input(rng):
    gen = torch.Generator().manual_seed(4)
    model = _build_mlp(relu=False, bias=False, gen=gen,
                       calib_u8=rng.integers(0, 256, size=(64, C * H * W), dtype=np.uint8))
    grupos = dense.analizar(model, torch.zeros(2, C, H, W))
    grupos[1][0].rsign = nn.Identity()
    with pytest.raises(ValueError, match="does not binarize its input"):
        dense.plegar(grupos, mean=MEAN, std=STD)


def test_umbral_edge_cases():
    imin, imax = dense.INT32_MIN, dense.INT32_MAX
    assert dense._umbral(2.0, 1.0, -0.5, relu=True) == (False, imin)     # ReLU(z) >= 0 >= beta: always 1
    assert dense._umbral(0.0, 1.0, 0.5, relu=False) == (False, imin)      # constant z = 1 >= 0.5
    assert dense._umbral(0.0, 0.0, 0.5, relu=False) == (False, imax)      # constant z = 0 < 0.5
    assert dense._umbral(2.0, 1.0, 4.0, relu=False) == (False, 2)         # 2 raw + 1 >= 4  <=> raw >= 1.5
    assert dense._umbral(-2.0, 1.0, 4.0, relu=False) == (True, 2)         # raw <= -1.5  <=>  -raw >= 2
    assert dense._umbral(1e-30, 0.0, 1.0, relu=False) == (False, imax)    # clamped to int32


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not found")
def test_exported_header_compiled_with_lib_symbex_matches_integer_simulation(rng, tmp_path):
    gen = torch.Generator().manual_seed(11)
    model = _build_mlp(relu=True, bias=True, gen=gen,
                       calib_u8=rng.integers(0, 256, size=(64, C * H * W), dtype=np.uint8))
    capas = dense.plegar(dense.analizar(model, torch.zeros(2, C, H, W)), mean=MEAN, std=STD)
    x_u8 = rng.integers(0, 256, size=(300, C * H * W), dtype=np.uint8)
    ruta = str(tmp_path / "modelo_symbex.h")
    dense.escribir_header(capas, ruta)
    iguales, total, msg = dense.validar_c(ruta, LIB_SYMBEX, x_u8, dense.simular(capas, x_u8))
    assert msg == ""
    assert iguales == total == 300


# ---------------------------------------------------------------------------------------------------
# conv path: fixed-point BatchNorm folding
# ---------------------------------------------------------------------------------------------------
def test_fold_bn_fixed_point_matches_float_with_negative_gamma():
    torch.manual_seed(1)
    bn = nn.BatchNorm2d(8)
    _randomize_bn(bn, torch.Generator().manual_seed(5))
    alpha = np.abs(np.random.default_rng(0).normal(size=8)) + 0.01
    q_act, q_mult = 16, 20
    escala, shift = _fold_bn(bn, alpha, q_act, q_mult)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    np.testing.assert_allclose(escala / 2 ** q_mult, alpha * g, atol=2 ** -q_mult)
    np.testing.assert_allclose(shift / 2 ** q_act, bn.bias.detach().numpy() - bn.running_mean.numpy() * g,
                               atol=2 ** -q_act)
    assert (np.sign(escala) == np.sign(g)).all() and (escala < 0).any()


def test_fixed_point_bn_rprelu_matches_float_reference(rng):
    """y = BN(alpha * raw) + identity, then RPReLU: (y - gamma) * (slope if <= 0 else 1) + zeta."""
    q_act, q_mult = 16, 20
    ch = 6
    bn = nn.BatchNorm2d(ch)
    _randomize_bn(bn, torch.Generator().manual_seed(9))
    alpha = np.abs(rng.normal(size=ch)) + 0.05
    escala_q, shift_q = _fold_bn(bn, alpha, q_act, q_mult)
    gamma, zeta, slope = rng.normal(size=ch), rng.normal(size=ch), rng.uniform(0.0, 0.5, size=ch)
    q = lambda v: np.round(v * 2 ** q_act).astype(np.int64)
    raw = rng.integers(-72, 73, size=(ch, 4, 4))
    identity = rng.normal(size=(ch, 4, 4))

    got = _aplicar_rprelu(raw, q(identity), q(gamma), q(zeta), q(slope), escala_q, shift_q, q_act, q_mult)

    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    y = (alpha * g)[:, None, None] * raw + (bn.bias.detach().numpy() - bn.running_mean.numpy() * g)[:, None, None]
    d = y + identity - gamma[:, None, None]
    want = np.where(d <= 0, d * slope[:, None, None], d) + zeta[:, None, None]
    # rounding of each fixed-point term: escala (|raw| * 2^-q_mult), slope (|d| * 2^-q_act), a few q_act LSBs
    tol = np.abs(raw) * 2.0 ** -q_mult + np.abs(d) * 2.0 ** -q_act + 6 * 2.0 ** -q_act
    assert (np.abs(got / 2 ** q_act - want) <= tol).all()
