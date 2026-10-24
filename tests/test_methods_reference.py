"""Every method in compiler_core/methods vs a straightforward numpy reference, on tiny random models:
    linear/plain, conv/plain, kbit (linear + conv), linear/bitslice, linear/gated.
For bitslice and gated the hardware simulators (simulator.py) and verify_fidelity are checked too, and
linear/plain is also run through the C engine's XNOR-popcount (lib_symbex compiled for this host)."""
import ctypes

import numpy as np
import pytest
import torch
import torch.nn as nn

from compiler_core.export.dense import _empaquetar
from compiler_core.methods import apply_method, apply_plan
from compiler_core.methods.base import RSign
from compiler_core.methods.conv.plain import ConvPlain
from compiler_core.methods.kbit import ConvKBit, CuantizadorActivacion, LinearKBit
from compiler_core.methods.linear.bitslice import SymbexVotingPool
from compiler_core.methods.linear.core import BipolarStepSTE
from compiler_core.methods.linear.gated import SymbexBlockGatedPool
from compiler_core.methods.linear.plain import LinearPlain
from compiler_core.simulator import simulate_bitslice_cpp, simulate_block_gated_cpp
from compiler_core.validator import verify_fidelity
from helpers import SymbexLayerData, as_ptr, pack_lsb_words


def _np(t):
    return t.detach().cpu().numpy().astype(np.float64)


def _sign(x):
    """0 -> +1 (plain convention)."""
    return np.where(x >= 0, 1.0, -1.0)


def _set_betas(module, gen):
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, RSign):
                m.beta.copy_(torch.randn(m.beta.shape, generator=gen) * 0.5)


# ---------------------------------------------------------------------------------------------------
# plain
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_in,n_out,bias", [(33, 7, False), (100, 5, True), (8, 3, True)])
def test_linear_plain_matches_numpy_sign_alpha_reference(n_in, n_out, bias):
    gen = torch.Generator().manual_seed(n_in)
    layer = LinearPlain.from_float(nn.Linear(n_in, n_out, bias=bias), binarize_input=True)
    _set_betas(layer, gen)
    with torch.no_grad():
        layer.weight[0, :3] = 0.0
        if bias:
            layer.bias.copy_(torch.randn(n_out, generator=gen))
    x = torch.randn(16, n_in, generator=gen)
    with torch.no_grad():
        got = layer(x).double().numpy()
    w = _np(layer.weight)
    xb = _sign(_np(x) - _np(layer.rsign.beta))
    want = xb @ (_sign(w) * np.abs(w).mean(1, keepdims=True)).T + (_np(layer.bias) if bias else 0)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_in", [33, 100])
def test_linear_plain_equals_c_engine_xnor_popcount_times_alpha(c_libs, n_in):
    n_out = 5
    gen = torch.Generator().manual_seed(1)
    layer = LinearPlain(n_in, n_out, binarize_input=True)
    _set_betas(layer, gen)
    x = torch.randn(8, n_in, generator=gen)
    with torch.no_grad():
        want = layer(x).double().numpy()
    w = _np(layer.weight)
    packed = np.ascontiguousarray(_empaquetar(_sign(w).astype(np.int64))[0].astype(np.uint32))
    alpha = np.abs(w).mean(1)
    lib = c_libs["native"]
    c_layer = SymbexLayerData(n_in, n_out, 1, as_ptr(packed, ctypes.c_uint32))
    for i in range(len(x)):
        bits = pack_lsb_words((_np(x[i:i + 1]) - _np(layer.rsign.beta) >= 0).astype(np.int64))[0]
        raw = np.zeros(n_out, dtype=np.int32)
        lib.symbex_dense_forward(as_ptr(bits, ctypes.c_uint32), as_ptr(raw, ctypes.c_int32), ctypes.byref(c_layer))
        np.testing.assert_allclose(raw * alpha, want[i], rtol=1e-5, atol=1e-5)


def _conv_reference(x, w, bias, stride, padding):
    """Direct loops over output positions; zero padding (the padded input contributes 0)."""
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    oh, ow = (h + 2 * padding - k) // stride + 1, (wd + 2 * padding - k) // stride + 1
    out = np.zeros((n, o, oh, ow))
    for i in range(oh):
        for j in range(ow):
            win = xp[:, :, i * stride:i * stride + k, j * stride:j * stride + k]
            out[:, :, i, j] = np.tensordot(win, w, axes=([1, 2, 3], [1, 2, 3]))
    return out + (0 if bias is None else bias[None, :, None, None])


@pytest.mark.parametrize("k,stride,padding,bias", [(3, 1, 1, False), (3, 2, 1, True), (1, 1, 0, False),
                                                   (5, 1, 2, True)])
def test_conv_plain_matches_numpy_reference(k, stride, padding, bias):
    gen = torch.Generator().manual_seed(k * 10 + stride)
    layer = ConvPlain.from_float(nn.Conv2d(3, 4, k, stride=stride, padding=padding, bias=bias),
                                 binarize_input=True)
    _set_betas(layer, gen)
    if bias:
        with torch.no_grad():
            layer.bias.copy_(torch.randn(4, generator=gen))
    x = torch.randn(2, 3, 7, 7, generator=gen)
    with torch.no_grad():
        got = layer(x).double().numpy()
    w = _np(layer.weight)
    wb = _sign(w) * np.abs(w).mean(axis=(1, 2, 3), keepdims=True)
    xb = _sign(_np(x) - _np(layer.rsign.beta))
    want = _conv_reference(xb, wb, _np(layer.bias) if bias else None, stride, padding)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_conv_plain_first_layer_keeps_real_input():
    layer = ConvPlain.from_float(nn.Conv2d(2, 3, 3, padding=1, bias=False), binarize_input=False)
    x = torch.randn(1, 2, 5, 5)
    with torch.no_grad():
        got = layer(x).double().numpy()
    w = _np(layer.weight)
    want = _conv_reference(_np(x), _sign(w) * np.abs(w).mean(axis=(1, 2, 3), keepdims=True), None, 1, 1)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_apply_method_and_apply_plan_wire_first_layer_and_options():
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.Flatten(), nn.Linear(4 * 4 * 4, 8), nn.Linear(8, 3))
    apply_method(model, "plain")
    assert isinstance(model[0], ConvPlain) and isinstance(model[0].rsign, nn.Identity)
    assert isinstance(model[2], LinearPlain) and isinstance(model[2].rsign, RSign)
    assert isinstance(model[3].rsign, RSign)

    model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 6), nn.Linear(6, 2))
    apply_plan(model, {"0": 4, "2": "bin"})
    assert isinstance(model[0], LinearKBit) and isinstance(model[0].cuant_entrada, nn.Identity)
    assert isinstance(model[2], LinearPlain) and isinstance(model[2].rsign, RSign)
    assert isinstance(model[3], nn.Linear)                     # not in the plan: stays FP32
    with pytest.raises(ValueError):
        apply_plan(nn.Sequential(nn.Linear(2, 2)), {"0": 1})
    with pytest.raises(ValueError):
        apply_method(nn.Sequential(nn.Linear(2, 2)), "gated")


# ---------------------------------------------------------------------------------------------------
# kbit
# ---------------------------------------------------------------------------------------------------
def _np_quant_weight(w, k):
    L = 2 ** (k - 1) - 1
    s = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-8).reshape((-1,) + (1,) * (w.ndim - 1)) / L
    return np.clip(np.round(w / s), -L, L) * s


def _np_quant_act(x, k, lo, hi):
    levels = 2 ** k - 1
    s = max(hi - lo, 1e-8) / levels
    return np.clip(np.round((x - lo) / s), 0, levels) * s + lo


def _calibrated(layer, x):
    layer.train()
    with torch.no_grad():
        layer(x)                                    # first batch: lo / hi = min / max of the input
    return layer.eval()


@pytest.mark.parametrize("k", [2, 4, 8])
def test_linear_kbit_matches_numpy_fake_quantization(k):
    gen = torch.Generator().manual_seed(k)
    layer = LinearKBit.from_float(nn.Linear(20, 6), k, quantize_input=True)
    x = torch.randn(32, 20, generator=gen)
    _calibrated(layer, x)
    q = layer.cuant_entrada
    assert float(q.lo) == float(x.min()) and float(q.hi) == float(x.max())
    with torch.no_grad():
        got = layer(x).double().numpy()
    xq = _np_quant_act(_np(x), k, float(q.lo), float(q.hi))
    want = xq @ _np_quant_weight(_np(layer.weight), k).T + _np(layer.bias)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    # weights land on 2L+1 symmetric levels per output channel
    wq = _np_quant_weight(_np(layer.weight), k)
    L = 2 ** (k - 1) - 1
    steps = wq / (np.abs(_np(layer.weight)).max(1, keepdims=True) / L)
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-6)


@pytest.mark.parametrize("k,stride,padding", [(2, 1, 1), (4, 2, 1), (8, 1, 0)])
def test_conv_kbit_matches_numpy_fake_quantization(k, stride, padding):
    gen = torch.Generator().manual_seed(k + stride)
    layer = ConvKBit.from_float(nn.Conv2d(3, 4, 3, stride=stride, padding=padding), k, quantize_input=True)
    x = torch.randn(2, 3, 6, 6, generator=gen)
    _calibrated(layer, x)
    q = layer.cuant_entrada
    with torch.no_grad():
        got = layer(x).double().numpy()
    xq = _np_quant_act(_np(x), k, float(q.lo), float(q.hi))
    want = _conv_reference(xq, _np_quant_weight(_np(layer.weight), k), _np(layer.bias), stride, padding)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_activation_quantizer_running_minmax_is_an_ema_after_the_first_batch():
    q = CuantizadorActivacion(4, momentum=0.1).train()
    a, b = torch.tensor([-1.0, 3.0]), torch.tensor([-3.0, 1.0])
    q(a)
    q(b)
    assert float(q.lo) == pytest.approx(0.9 * -1.0 + 0.1 * -3.0)
    assert float(q.hi) == pytest.approx(0.9 * 3.0 + 0.1 * 1.0)


# ---------------------------------------------------------------------------------------------------
# bitslice (SymbexVotingPool): PyTorch vs simulate_bitslice_cpp vs numpy reference
# ---------------------------------------------------------------------------------------------------
def _np_bitslice_weights(w, mean, std, w_max, k_bits):
    levels = 2 ** k_bits - 1
    thr = max(2.0 * std, 1e-4)
    mask = np.abs(w - mean) > thr
    wq = np.clip(np.round((np.clip(w, mean - thr, mean + thr) / w_max) * (levels / 2) + levels / 2), 0, levels)
    rec = 2 * wq - levels
    for n in range(w.shape[0]):
        if mask[n].any():
            mag = np.clip(np.round(np.abs(w[n][mask[n]]).mean() / w_max * levels), 0, levels * 3)
            rec[n, mask[n]] += mag * np.where(wq[n, mask[n]] >= (levels + 1) / 2, 1, -1)
    return rec


def _bitslice_student(in_f, hid, classes, k_bits, M):
    student = nn.Sequential(SymbexVotingPool(in_f, hid, expansion_factor=M, k_bits=k_bits), BipolarStepSTE(),
                            SymbexVotingPool(hid, classes, expansion_factor=M, k_bits=k_bits))
    with torch.no_grad():
        student[0].weight[:, :, 0] *= 8.0                    # make sure there are outliers
    student.train()
    with torch.no_grad():
        student(torch.ones(1, in_f))                         # initializes running mean / std / W_max
    return student.eval()


@pytest.mark.parametrize("k_bits,M", [(1, 1), (2, 2), (3, 2), (4, 3)])
def test_bitslice_simulator_and_numpy_reference_match_pytorch(rng, k_bits, M):
    in_f, hid, classes = 24, 16, 5
    student = _bitslice_student(in_f, hid, classes, k_bits, M)
    x = np.where(rng.random((64, in_f)) < 0.5, -1.0, 1.0).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        torch_out = (student(xt) / torch.clamp(student[2].output_scale, min=1e-4)).double().numpy()
    sim_out = simulate_bitslice_cpp(student, xt, k_bits=k_bits)

    cur = x.astype(np.float64)
    for layer in (student[0], student[2]):
        votes = 0
        for m in range(layer.M):
            votes = votes + cur @ _np_bitslice_weights(_np(layer.weight[m]), layer.running_mean[m].item(),
                                                       layer.running_std[m].item(),
                                                       layer.running_W_max[m].item(), k_bits).T
        cur = np.where(votes > 0, 1.0, -1.0) if layer is student[0] else votes

    np.testing.assert_array_equal(sim_out, cur)
    np.testing.assert_allclose(torch_out, cur, rtol=1e-5, atol=1e-3)
    assert verify_fidelity(student, xt, x, "bitslice", k_bits=k_bits) == 100.0


# ---------------------------------------------------------------------------------------------------
# gated (SymbexBlockGatedPool): PyTorch vs simulate_block_gated_cpp vs numpy reference
# ---------------------------------------------------------------------------------------------------
def _np_gated_predict(student, x):
    l0, l1 = student[0], student[2]
    sgn = lambda t: np.where(_np(t) > 0, 1.0, -1.0)           # gated convention: 0 -> -1
    gate = x @ sgn(l0.gate_weight).T - np.arange(l0.num_blocks) * 1e-4
    active = np.zeros_like(gate, dtype=bool)
    np.put_along_axis(active, np.argsort(-gate, axis=1, kind="stable")[:, :l0.k_active], True, axis=1)
    neuron_active = np.repeat(active, l0.block_size, axis=1)[:, :l0.out_features]
    core = x @ sgn(l0.core_weight).T
    hidden = np.where(neuron_active & (core > 0), 1.0, -1.0)
    return (hidden @ sgn(l1.core_weight).T).argmax(1)


@pytest.mark.parametrize("in_f,hid,block,ratio", [(40, 24, 8, 0.5), (33, 20, 8, 1.0), (17, 64, 16, 0.25)])
def test_gated_simulator_and_numpy_reference_match_pytorch(rng, in_f, hid, block, ratio):
    torch.manual_seed(in_f)
    student = nn.Sequential(SymbexBlockGatedPool(in_f, hid, block_size=block, active_ratio=ratio), BipolarStepSTE(),
                            SymbexBlockGatedPool(hid, 10, block_size=32)).eval()
    with torch.no_grad():
        student[0].core_weight[0, :3] = 0.0                    # exact zeros: 0 -> -1 everywhere
    x = np.where(rng.random((128, in_f)) < 0.5, -1.0, 1.0).astype(np.float32)
    with torch.no_grad():
        torch_pred = student(torch.from_numpy(x)).argmax(1).numpy()
    sim_pred = simulate_block_gated_cpp(student, x)
    ref_pred = _np_gated_predict(student, x.astype(np.float64))
    np.testing.assert_array_equal(sim_pred, ref_pred)
    np.testing.assert_array_equal(torch_pred, ref_pred)
    assert verify_fidelity(student, torch.from_numpy(x), x, "gated") == 100.0
