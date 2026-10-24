"""Bit-packing / unpacking roundtrip of every export path:
    export/plain._pack_rows (+ _filas layouts)   LSB-first 32-bit words, rows padded to `multiplo` words
    export/dense._empaquetar                     LSB-first 32-bit words, ceil(n / 32) per row
    export/gated (header)                        MSB-first bytes, weight > 0 -> 1
    export/bitslice._export_layer_to_arrays_v1   k MSB-first bit planes + outlier mask + magnitudes
"""
import re

import numpy as np
import pytest
import torch

from compiler_core.export.bitslice import _export_layer_to_arrays_v1, _write_header_v1
from compiler_core.export.dense import _empaquetar
from compiler_core.export.gated import export_model_v2_to_h
from compiler_core.export.plain import _filas, _pack_rows, _padding
from compiler_core.methods.linear.bitslice import SymbexVotingPool
from compiler_core.methods.linear.core import BipolarStepSTE
from compiler_core.methods.linear.gated import SymbexBlockGatedPool
from compiler_core.validator import validate_hardware_limits
from helpers import unpack_lsb_words

LENGTHS = [1, 7, 31, 32, 33, 64, 100, 129, 300]


def _weights_with_zeros(rng, shape):
    w = rng.normal(size=shape)
    w.flat[:: 7] = 0.0                                  # exact zeros: must pack as +1 (0 -> +1)
    return w


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("multiplo", [1, 4])
def test_pack_rows_roundtrip_padding_and_word_count(rng, n, multiplo):
    rows = _weights_with_zeros(rng, (5, n))
    packed, words = _pack_rows(rows, multiplo=multiplo)
    assert words % multiplo == 0
    assert words * 32 >= n and (words - multiplo) * 32 < n          # minimal padding
    assert all(len(r) == words for r in packed)
    bits, pad = unpack_lsb_words(np.array(packed, dtype=np.uint64), n)
    np.testing.assert_array_equal(bits, (rows >= 0).astype(np.int64))
    assert pad.sum() == 0


@pytest.mark.parametrize("n", LENGTHS)
def test_empaquetar_roundtrip_and_agrees_with_pack_rows(rng, n):
    signos = np.where(_weights_with_zeros(rng, (4, n)) >= 0, 1, -1)
    packed, words = _empaquetar(signos)
    assert words == (n + 31) // 32
    assert packed.max() < 2 ** 32
    bits, pad = unpack_lsb_words(packed, n)
    np.testing.assert_array_equal(2 * bits - 1, signos)
    assert pad.sum() == 0
    other, other_words = _pack_rows(signos, multiplo=1)
    assert other_words == words
    np.testing.assert_array_equal(np.array(other, dtype=np.uint64), packed)


@pytest.mark.parametrize("layout", ["hwc", "chw"])
def test_filas_layout_order_matches_the_engine_window(rng, layout):
    out, ic, k = 3, 5, 3
    w = rng.normal(size=(out, ic, k, k))
    filas = _filas(w, layout)
    assert filas.shape == (out, ic * k * k)
    for o in range(out):
        for c in range(ic):
            for kh in range(k):
                for kw in range(k):
                    idx = (kh * k + kw) * ic + c if layout == "hwc" else (c * k + kh) * k + kw
                    assert filas[o, idx] == w[o, c, kh, kw]


# ---------------------------------------------------------------------------------------------------
# gated header (MSB-first bytes)
# ---------------------------------------------------------------------------------------------------
def _read_c_array(text, name):
    body = re.search(rf"{name}\[[^\]]*\](?:\[[^\]]*\])?\s*PROGMEM\s*=\s*\{{(.*?)\}};", text, re.S).group(1)
    rows = re.findall(r"\{([^{}]*)\}", body)
    return np.array([[int(v, 16) for v in r.split(",") if v.strip()] for r in rows], dtype=np.uint8)


def _unpack_msb_bytes(byte_rows, n):
    bits = np.unpackbits(byte_rows, axis=1, bitorder="big")
    return bits[:, :n].astype(np.int64), bits[:, n:].astype(np.int64)


@pytest.mark.parametrize("in_f,hid", [(13, 20), (40, 24)])
def test_gated_header_weights_unpack_to_the_model_signs(tmp_path, in_f, hid, capsys):
    torch.manual_seed(2)
    student = torch.nn.Sequential(SymbexBlockGatedPool(in_f, hid, block_size=8, active_ratio=0.5),
                                  BipolarStepSTE(), SymbexBlockGatedPool(hid, 10, block_size=32))
    with torch.no_grad():
        student[0].core_weight[0, :4] = 0.0             # 0 -> -1 in the gated convention (bit 0)
    path = tmp_path / "gated.h"
    export_model_v2_to_h(student, str(path), block_size=8, k_active=student[0].k_active)
    text = path.read_text()
    for name, tensor, n in [("gate_weights_bin", student[0].gate_weight, in_f),
                            ("core_weights_bin", student[0].core_weight, in_f),
                            ("out_weights_bin", student[2].core_weight, hid)]:
        bits, pad = _unpack_msb_bytes(_read_c_array(text, name), n)
        np.testing.assert_array_equal(bits, (tensor.detach().numpy() > 0).astype(np.int64))
        assert pad.sum() == 0
    assert f"#define GATE_NUM_BLOCKS {student[0].num_blocks}" in text


# ---------------------------------------------------------------------------------------------------
# bitslice bit planes
# ---------------------------------------------------------------------------------------------------
def _trained_pool(in_f, out_f, k_bits, outlier_scale=None, seed=0):
    torch.manual_seed(seed)
    pool = SymbexVotingPool(in_f, out_f, expansion_factor=1, k_bits=k_bits)
    if outlier_scale is not None:
        with torch.no_grad():
            pool.weight[0, :, 0] = outlier_scale * pool.weight[0].abs().max()   # one huge outlier per neuron
    pool.train()
    pool(torch.ones(1, in_f))                          # initializes the running statistics
    return pool.eval()


def _unpack_planes(exp, k_bits):
    out_f, in_f = exp["out_features"], exp["in_features"]
    bpn = (in_f + 7) // 8
    wq = np.zeros((out_f, in_f), dtype=np.int64)
    for k, plane in enumerate(exp["planes"]):
        bits, pad = _unpack_msb_bytes(np.array(plane, dtype=np.uint8).reshape(out_f, bpn), in_f)
        assert pad.sum() == 0
        wq |= bits << (k_bits - 1 - k)                   # plane 0 = most significant bit
    mask, pad = _unpack_msb_bytes(np.array(exp["outliers"], dtype=np.uint8).reshape(out_f, bpn), in_f)
    assert pad.sum() == 0
    return wq, mask.astype(bool), np.array(exp["outlier_magnitudes"])


@pytest.mark.parametrize("k_bits", [1, 2, 3, 4])
@pytest.mark.parametrize("in_f", [9, 16, 21])
def test_bitslice_planes_reconstruct_the_pytorch_quantized_weights(k_bits, in_f):
    pool = _trained_pool(in_f, 6, k_bits, outlier_scale=3.0)
    exp = _export_layer_to_arrays_v1(pool.weight[0].detach().numpy(), pool.running_mean[0].item(),
                                     pool.running_std[0].item(), pool.running_W_max[0].item(), k_bits)
    wq, mask, mag = _unpack_planes(exp, k_bits)
    levels = 2 ** k_bits - 1
    sign_msb = np.where(wq >= (levels + 1) / 2, 1, -1)
    rebuilt = 2 * wq - levels + mask * mag[:, None] * sign_msb
    with torch.no_grad():
        want = pool._quantize_weights(pool.weight[0], 0).numpy()
    assert mask.any()
    np.testing.assert_array_equal(rebuilt, want)


@pytest.mark.parametrize("k_bits", [
    *range(1, 5),
    *[pytest.param(k, marks=pytest.mark.xfail(strict=True, reason=(
        "validate_hardware_limits accepts k_bits 1..8, but export/bitslice._write_header_v1 has MAX_K_BITS = 4 "
        "plane slots per SymbexSubLayer: for k > 4 it writes k plane names into the 4-slot initializer")))
      for k in range(5, 9)],
])
def test_bitslice_header_has_a_plane_slot_for_every_k_the_validator_accepts(tmp_path, k_bits):
    student = torch.nn.Sequential(_trained_pool(12, 4, k_bits))
    validate_hardware_limits(student, "bitslice")        # accepted
    path = tmp_path / "bitslice.h"
    _write_header_v1(student, str(path), k_bits=k_bits)
    planes = re.search(r"\{ \{([^}]*)\}, layer0_m0_outliers", path.read_text()).group(1).split(",")
    assert len(planes) == 4                              # the struct has exactly MAX_K_BITS = 4 slots
    assert sum(p.strip() != "NULL" for p in planes) == k_bits


@pytest.mark.parametrize("k_bits", [
    *range(1, 6),
    *[pytest.param(k, marks=pytest.mark.xfail(strict=True, reason=(
        "export/bitslice writes outlier magnitudes into an int8_t array, but they are clipped to levels * 3 = "
        "3 * (2^k - 1), which exceeds 127 for k >= 6 (accepted by validate_hardware_limits): the C value wraps")))
      for k in range(6, 9)],
])
def test_bitslice_outlier_magnitudes_fit_the_int8_header_array(k_bits):
    pool = _trained_pool(64, 4, k_bits, outlier_scale=50.0)     # 64 inputs: the outlier barely moves the std
    validate_hardware_limits(torch.nn.Sequential(pool), "bitslice")
    exp = _export_layer_to_arrays_v1(pool.weight[0].detach().numpy(), pool.running_mean[0].item(),
                                     pool.running_std[0].item(), pool.running_W_max[0].item(), k_bits)
    assert max(exp["outlier_magnitudes"]) <= 127


@pytest.mark.parametrize("padding,esperado", [(0, 0), (1, 1), ((2, 2), 2)])
def test_conv_padding_is_taken_from_the_layer_not_the_kernel(padding, esperado):
    conv = torch.nn.Conv2d(4, 8, kernel_size=3, padding=padding)
    assert _padding(conv) == esperado


@pytest.mark.parametrize("padding", ["same", (1, 0)])
def test_conv_padding_rejects_string_and_non_square(padding):
    conv = torch.nn.Conv2d(4, 8, kernel_size=3, padding=padding)
    with pytest.raises(ValueError, match=r"\[EXPORT\]"):
        _padding(conv)
