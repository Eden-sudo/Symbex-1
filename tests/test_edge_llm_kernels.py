"""1-bit matrix-vector kernels of benchmarks/edge_llm_probe/kernels/onebit_gemv.c, compiled for this host and
compared with a numpy reference: every kernel must give EXACTLY the integers of the +-1 dot product."""
import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest

from helpers import as_ptr, pack_lsb_words, unpack_lsb_words

KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "benchmarks", "edge_llm_probe", "kernels")

P = ctypes.POINTER
I8, U8, I16, U32, I32 = ctypes.c_int8, ctypes.c_uint8, ctypes.c_int16, ctypes.c_uint32, ctypes.c_int32
INT = ctypes.c_int
SIGNATURES = {
    "lut4_build": ([P(I8), INT, P(I16)], None),
    "bs4_pack": ([P(U8), INT, P(U32)], I32),
    "onebit_rowsums": ([P(U32), INT, INT, P(I32)], None),
    "xnor_pack": ([P(I8), INT, P(U32)], None),
    "gemv_ref": ([P(U32), P(I8), P(I32), INT, INT], None),
    "gemv_mask": ([P(U32), P(I8), P(I32), INT, INT], None),
    "gemv_lut4_rows": ([P(U32), P(I16), P(I32), INT, INT, INT], None),
    "gemv_lut4": ([P(U32), P(I16), P(I32), INT, INT], None),
    "gemv_bitserial4": ([P(U32), P(U32), I32, P(I32), P(I32), INT, INT], None),
    "gemv_xnor": ([P(U32), P(U32), P(I32), INT, INT], None),
}

# (M, N): the smallest and a large shape, a single row, and shapes where N is not a power of two
SHAPES = [(1, 32), (1, 1024), (3, 96), (7, 64), (16, 1024), (33, 160), (64, 2048)]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found: edge-LLM kernel tests skipped")
    out = str(tmp_path_factory.mktemp("onebit_gemv") / "libonebit_gemv.so")
    cmd = ["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-I", KERNEL_DIR,
           os.path.join(KERNEL_DIR, "onebit_gemv.c"), "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"onebit_gemv.c did not compile:\n{r.stderr[-2000:]}")
    lib = ctypes.CDLL(out)
    for name, (args, res) in SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    return lib


def _weights(rng, m, n, kind="random"):
    """(bits (m, n) 0/1, packed (m, n / 32) uint32 words)."""
    if kind == "plus":
        bits = np.ones((m, n), dtype=np.int64)
    elif kind == "minus":
        bits = np.zeros((m, n), dtype=np.int64)
    else:
        bits = (rng.random((m, n)) < 0.5).astype(np.int64)
    return bits, np.ascontiguousarray(pack_lsb_words(bits))


def _reference(bits, x):
    """The +-1 dot product, in int64 numpy."""
    return (2 * bits - 1) @ x.astype(np.int64)


def _run_int8_kernels(lib, packed, x, m, n):
    """(gemv_ref, gemv_mask, gemv_lut4) outputs and the LUT4 table."""
    y_ref, y_mask, y_lut = (np.zeros(m, dtype=np.int32) for _ in range(3))
    lut = np.zeros(n // 4 * 16, dtype=np.int16)
    lib.gemv_ref(as_ptr(packed, U32), as_ptr(x, I8), as_ptr(y_ref, I32), m, n)
    lib.gemv_mask(as_ptr(packed, U32), as_ptr(x, I8), as_ptr(y_mask, I32), m, n)
    lib.lut4_build(as_ptr(x, I8), n, as_ptr(lut, I16))
    lib.gemv_lut4(as_ptr(packed, U32), as_ptr(lut, I16), as_ptr(y_lut, I32), m, n)
    return y_ref, y_mask, y_lut, lut


def _run_bitserial(lib, packed, a, m, n):
    planes = np.zeros(4 * (n // 32), dtype=np.uint32)
    rowsum = np.zeros(m, dtype=np.int32)
    y = np.zeros(m, dtype=np.int32)
    sum_a = lib.bs4_pack(as_ptr(a, U8), n, as_ptr(planes, U32))
    lib.onebit_rowsums(as_ptr(packed, U32), m, n, as_ptr(rowsum, I32))
    lib.gemv_bitserial4(as_ptr(packed, U32), as_ptr(planes, U32), sum_a, as_ptr(rowsum, I32), as_ptr(y, I32), m, n)
    return y, sum_a, rowsum


def _run_xnor(lib, packed, x_pm1, m, n):
    xbits = np.zeros(n // 32, dtype=np.uint32)
    y = np.zeros(m, dtype=np.int32)
    lib.xnor_pack(as_ptr(x_pm1, I8), n, as_ptr(xbits, U32))
    lib.gemv_xnor(as_ptr(packed, U32), as_ptr(xbits, U32), as_ptr(y, I32), m, n)
    return y, xbits


@pytest.mark.parametrize("m,n", SHAPES)
def test_int8_kernels_match_numpy_on_random_data(lib, rng, m, n):
    bits, packed = _weights(rng, m, n)
    x = rng.integers(-128, 128, size=n).astype(np.int8)
    expected = _reference(bits, x)
    y_ref, y_mask, y_lut, _ = _run_int8_kernels(lib, packed, x, m, n)
    assert np.array_equal(y_ref, expected)
    assert np.array_equal(y_mask, expected)
    assert np.array_equal(y_lut, expected)


@pytest.mark.parametrize("m,n", SHAPES)
def test_bitserial4_matches_numpy_on_random_data(lib, rng, m, n):
    bits, packed = _weights(rng, m, n)
    a = rng.integers(0, 16, size=n).astype(np.uint8)
    y, sum_a, rowsum = _run_bitserial(lib, packed, a, m, n)
    assert sum_a == int(a.sum())
    assert np.array_equal(rowsum, (2 * bits - 1).sum(axis=1))
    assert np.array_equal(y, _reference(bits, a.astype(np.int64) - 8))


@pytest.mark.parametrize("m,n", SHAPES)
def test_xnor_matches_numpy_on_random_data(lib, rng, m, n):
    bits, packed = _weights(rng, m, n)
    x = np.where(rng.random(n) < 0.5, -1, 1).astype(np.int8)
    y, xbits = _run_xnor(lib, packed, x, m, n)
    assert np.array_equal(y, _reference(bits, x))
    x_bits01, _ = unpack_lsb_words(xbits.reshape(1, -1), n)
    assert np.array_equal(2 * x_bits01[0] - 1, x)


@pytest.mark.parametrize("kind", ["plus", "minus"])
@pytest.mark.parametrize("fill", [-128, 127, 0, -1, 1])
def test_uniform_weights_and_uniform_int8_activations(lib, rng, kind, fill):
    m, n = 5, 1024
    bits, packed = _weights(rng, m, n, kind)
    x = np.full(n, fill, dtype=np.int8)
    expected = _reference(bits, x)
    for y in _run_int8_kernels(lib, packed, x, m, n)[:3]:
        assert np.array_equal(y, expected)
    # the corner is exactly +-N * |fill|: no int32 wrap-around, no overflow of the int16 table
    assert abs(int(expected[0])) == n * abs(fill)


def test_alternating_int8_extremes_and_lut_entries_fit_int16(lib, rng):
    m, n = 8, 1024
    bits, packed = _weights(rng, m, n)
    x = np.where(np.arange(n) % 2 == 0, -128, 127).astype(np.int8)
    y_ref, y_mask, y_lut, lut = _run_int8_kernels(lib, packed, x, m, n)
    expected = _reference(bits, x)
    assert np.array_equal(y_ref, expected)
    assert np.array_equal(y_mask, expected)
    assert np.array_equal(y_lut, expected)
    assert int(np.abs(lut.astype(np.int64)).max()) <= 4 * 128


@pytest.mark.parametrize("kind", ["plus", "minus", "random"])
@pytest.mark.parametrize("fill", [0, 15, 8])
def test_bitserial4_uniform_activations_at_both_ends_of_the_range(lib, rng, kind, fill):
    m, n = 6, 1024
    bits, packed = _weights(rng, m, n, kind)
    a = np.full(n, fill, dtype=np.uint8)
    y, _, _ = _run_bitserial(lib, packed, a, m, n)
    assert np.array_equal(y, _reference(bits, a.astype(np.int64) - 8))


def test_bitserial4_alternating_extremes_at_the_minimum_width(lib, rng):
    m, n = 4, 32
    bits, packed = _weights(rng, m, n)
    a = np.where(np.arange(n) % 2 == 0, 0, 15).astype(np.uint8)
    y, _, _ = _run_bitserial(lib, packed, a, m, n)
    assert np.array_equal(y, _reference(bits, a.astype(np.int64) - 8))


@pytest.mark.parametrize("kind", ["plus", "minus"])
def test_xnor_all_equal_and_all_opposite(lib, rng, kind):
    m, n = 3, 64
    _, packed = _weights(rng, m, n, kind)
    w_sign = 1 if kind == "plus" else -1
    for x_sign in (1, -1):
        y, _ = _run_xnor(lib, packed, np.full(n, x_sign, dtype=np.int8), m, n)
        assert np.array_equal(y, np.full(m, n * x_sign * w_sign))


def test_lut4_rows_helper_splits_rows_without_changing_the_result(lib, rng):
    m, n = 20, 256
    bits, packed = _weights(rng, m, n)
    x = rng.integers(-128, 128, size=n).astype(np.int8)
    lut = np.zeros(n // 4 * 16, dtype=np.int16)
    lib.lut4_build(as_ptr(x, I8), n, as_ptr(lut, I16))
    y = np.full(m, 12345, dtype=np.int32)
    lib.gemv_lut4_rows(as_ptr(packed, U32), as_ptr(lut, I16), as_ptr(y, I32), n, 0, 9)
    assert np.all(y[9:] == 12345)  # rows outside [r0, r1) are untouched
    lib.gemv_lut4_rows(as_ptr(packed, U32), as_ptr(lut, I16), as_ptr(y, I32), n, 9, m)
    assert np.array_equal(y, _reference(bits, x))
