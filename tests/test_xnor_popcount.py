"""XNOR-popcount dot product identity:  sum_i a_i * b_i = n - 2 * popcount((A ^ B) & mask)
for a, b in {-1, +1}^n packed as bits (+1 -> 1), including lengths that are not a multiple of 32,
where the padding bits of the last word must be masked out.

Checked on (1) the words produced by the exporters' packers and (2) the C engine (hal/generico,
every popcount path) compiled for this host."""
import ctypes

import numpy as np
import pytest

from compiler_core.export.dense import _empaquetar
from compiler_core.export.plain import _pack_rows
from helpers import SymbexLayerData, as_ptr, pack_lsb_words, popcount, unpack_lsb_words

LENGTHS = [1, 5, 31, 32, 33, 63, 64, 65, 100, 127, 129, 784]


def _bipolar(rng, shape):
    return np.where(rng.random(shape) < 0.5, -1, 1).astype(np.int64)


def _tail_mask(n, n_words):
    m = np.zeros(n_words, dtype=np.uint64)
    m[: n // 32] = 0xFFFFFFFF
    if n % 32:
        m[n // 32] = (1 << (n % 32)) - 1
    return m


@pytest.mark.parametrize("n", LENGTHS)
def test_identity_holds_on_words_packed_by_the_dense_exporter(rng, n):
    a, b = _bipolar(rng, (1, n)), _bipolar(rng, (1, n))
    pa, words = _empaquetar(a)
    pb, _ = _empaquetar(b)
    diff = (pa[0] ^ pb[0]) & _tail_mask(n, words)
    assert n - 2 * popcount(diff) == int((a * b).sum())


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("multiplo", [1, 4])
def test_identity_holds_on_words_packed_by_the_conv_exporter(rng, n, multiplo):
    a, b = _bipolar(rng, (1, n)), _bipolar(rng, (1, n))
    pa, words = _pack_rows(a, multiplo=multiplo)
    pb, _ = _pack_rows(b, multiplo=multiplo)
    diff = (np.array(pa[0], dtype=np.uint64) ^ np.array(pb[0], dtype=np.uint64)) & _tail_mask(n, words)
    assert n - 2 * popcount(diff) == int((a * b).sum())


@pytest.mark.parametrize("n", [5, 33, 100])
def test_unmasked_xnor_over_zero_padding_overcounts_by_the_padding_width(rng, n):
    """Why the mask exists: with both paddings 0, XNOR counts every padding bit as a match."""
    a, b = _bipolar(rng, (1, n)), _bipolar(rng, (1, n))
    pa, words = _empaquetar(a)
    pb, _ = _empaquetar(b)
    total = words * 32
    unmasked = 2 * (total - popcount(pa[0] ^ pb[0])) - n
    assert unmasked - int((a * b).sum()) == 2 * (total - n)


# ---------------------------------------------------------------------------------------------------
# C engine: symbex_xnor_popcount (hal/generico) -- native, SWAR and lut16 paths
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", LENGTHS)
def test_c_xnor_popcount_matches_dot_product_with_dirty_padding(c_lib, rng, n):
    """Padding bits are filled with RANDOM, different garbage in a and b: they must never count."""
    a, b = _bipolar(rng, (1, n)), _bipolar(rng, (1, n))
    n_words = (n + 31) // 32 + 1                         # an extra word of garbage too
    pa = pack_lsb_words((a > 0).astype(np.int64), n_words)[0]
    pb = pack_lsb_words((b > 0).astype(np.int64), n_words)[0]
    garbage = rng.integers(0, 2 ** 32, size=(2, n_words), dtype=np.uint64)
    keep = _tail_mask(n, n_words)
    pa = ((pa.astype(np.uint64) & keep) | (garbage[0] & ~keep & 0xFFFFFFFF)).astype(np.uint32)
    pb = ((pb.astype(np.uint64) & keep) | (garbage[1] & ~keep & 0xFFFFFFFF)).astype(np.uint32)
    got = c_lib.symbex_xnor_popcount(as_ptr(pa, ctypes.c_uint32), as_ptr(pb, ctypes.c_uint32), n_words, n)
    assert got == int((a * b).sum())


def test_c_xnor_popcount_clamps_n_bits_to_the_buffer_and_handles_zero(c_lib):
    a = np.array([0xFFFFFFFF, 0x0], dtype=np.uint32)
    b = np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
    pa, pb = as_ptr(a, ctypes.c_uint32), as_ptr(b, ctypes.c_uint32)
    assert c_lib.symbex_xnor_popcount(pa, pb, 2, 64) == 32 - 32       # 32 matches, 32 differences
    assert c_lib.symbex_xnor_popcount(pa, pb, 1, 1000) == 32          # clamped to 32 real bits
    assert c_lib.symbex_xnor_popcount(pa, pb, 2, 0) == 0
    assert c_lib.symbex_xnor_popcount(pa, pb, 2, -3) == 0


def test_c_popcount32_matches_python_on_random_words(c_lib, rng):
    for x in list(rng.integers(0, 2 ** 32, size=200, dtype=np.uint64)) + [0, 0xFFFFFFFF, 0x80000000, 1]:
        assert c_lib.symbex_hal_popcount32(int(x)) == bin(int(x)).count("1")


@pytest.mark.parametrize("n_in,n_out", [(1, 3), (33, 7), (100, 5), (64, 2)])
def test_c_dense_forward_1bit_equals_signed_matmul_on_exported_weights(c_lib, rng, n_in, n_out):
    """symbex_dense_forward (k_bits = 1) on weights packed by export/dense._empaquetar."""
    w = _bipolar(rng, (n_out, n_in))
    x = _bipolar(rng, (1, n_in))
    packed, _ = _empaquetar(w)
    packed = np.ascontiguousarray(packed.astype(np.uint32))
    xin = pack_lsb_words((x > 0).astype(np.int64))[0]
    out = np.zeros(n_out, dtype=np.int32)
    layer = SymbexLayerData(n_in, n_out, 1, as_ptr(packed, ctypes.c_uint32))
    c_lib.symbex_dense_forward(as_ptr(xin, ctypes.c_uint32), as_ptr(out, ctypes.c_int32), ctypes.byref(layer))
    np.testing.assert_array_equal(out, (w @ x[0]).astype(np.int32))


@pytest.mark.parametrize("fn", ["symbex_dense_entrada_u8", "symbex_dense_entrada_u8_mascara",
                                "symbex_dense_entrada_u8_planos"])
@pytest.mark.parametrize("n_in", [1, 31, 33, 100, 784])
def test_c_uint8_first_layer_variants_equal_integer_matmul(c_lib, rng, fn, n_in):
    n_out = 6
    w = _bipolar(rng, (n_out, n_in))
    x = rng.integers(0, 256, size=n_in, dtype=np.uint8)
    x[:3] = 255                                           # the extremes, on purpose
    packed = np.ascontiguousarray(_empaquetar(w)[0].astype(np.uint32))
    out = np.zeros(n_out, dtype=np.int32)
    layer = SymbexLayerData(n_in, n_out, 1, as_ptr(packed, ctypes.c_uint32))
    if fn == "symbex_dense_entrada_u8_planos":
        planos = np.zeros(8 * ((n_in + 31) // 32), dtype=np.uint32)
        c_lib.symbex_dense_entrada_u8_planos(as_ptr(x, ctypes.c_uint8), as_ptr(planos, ctypes.c_uint32),
                                             as_ptr(out, ctypes.c_int32), ctypes.byref(layer))
    else:
        getattr(c_lib, fn)(as_ptr(x, ctypes.c_uint8), as_ptr(out, ctypes.c_int32), ctypes.byref(layer))
    np.testing.assert_array_equal(out, w @ x.astype(np.int64))


@pytest.mark.parametrize("n", [1, 31, 32, 33, 70])
def test_c_umbral_and_bitpack_set_exact_bits_and_zero_padding(c_lib, rng, n):
    raw = rng.integers(-50, 50, size=n, dtype=np.int32)
    umbral = rng.integers(-50, 50, size=n, dtype=np.int32)
    umbral[0] = np.iinfo(np.int32).min                     # always 1
    if n > 1:
        umbral[1] = np.iinfo(np.int32).max                 # always 0
    words = (n + 31) // 32
    bits = np.full(words, 0xFFFFFFFF, dtype=np.uint32)     # dirty buffer: the function must clear it
    c_lib.symbex_dense_umbral(as_ptr(raw, ctypes.c_int32), as_ptr(umbral, ctypes.c_int32),
                              as_ptr(bits, ctypes.c_uint32), n)
    got, pad = unpack_lsb_words(bits[None, :], n)
    np.testing.assert_array_equal(got[0], (raw >= umbral).astype(np.int64))
    assert pad.sum() == 0

    packed = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
    c_lib.symbex_bitpack_from_scores(as_ptr(raw, ctypes.c_int32), as_ptr(packed, ctypes.c_uint32), n)
    got, pad = unpack_lsb_words(packed[None, :], n)
    np.testing.assert_array_equal(got[0], (raw >= 0).astype(np.int64))
    assert pad.sum() == 0
