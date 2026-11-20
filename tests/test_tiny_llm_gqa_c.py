"""Host C reference versus Python, compiled by the laptop's clang."""
import ctypes
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


@pytest.fixture
def reference(monkeypatch):
    result = []
    for name in ("train", "intsim", "gqa_intsim"):
        spec = importlib.util.spec_from_file_location(name, DIR / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result.append(module)
    return result[1:]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    output = tmp_path_factory.mktemp("gqa") / "gqa.so"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", str(DIR / "c/gqa_attention.c"), "-o", str(output)],
                   check=True, capture_output=True, timeout=10)
    library = ctypes.CDLL(str(output))
    library.gqa_attention.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 8
    library.gqa_attention.restype = ctypes.c_int
    return library


def pointer(a):
    return a.ctypes.data_as(ctypes.c_void_p)


@pytest.mark.parametrize("heads,kv,length", [(1, 1, 1), (4, 1, 7), (6, 2, 9), (4, 4, 3), (2, 1, 512)])
@pytest.mark.parametrize("extreme", [False, True])
def test_bit_exact(lib, reference, heads, kv, length, extreme):
    I, G = reference
    rng = np.random.default_rng(18 + length)
    q = rng.integers(-127, 128, (heads, 64), dtype=np.int8)
    k, v = (rng.integers(-127, 128, (kv, length, 64), dtype=np.int8) for _ in range(2))
    qm = rng.integers(1, 9000, heads, dtype=np.int32)
    km, vm = (rng.integers(1, 9000, (kv, length), dtype=np.int32) for _ in range(2))
    if extreme:
        q[:] = 127
        k[:] = 127
        k[:, ::2] = -127
        qm[:] = km[:] = vm[:] = 1 << 20
        v[:, ::2] = -127
    table = I.EXP_TABLE.astype(np.uint16)
    out = np.full((heads, 64), -123, dtype=np.int64)
    args = [q, qm, k, km, v, vm, table, out]
    assert lib.gqa_attention(heads, kv, length, *map(pointer, args)) == 0
    qq = torch.from_numpy(q.astype(np.int64))[None, :, None, :]
    qs = torch.from_numpy(qm.astype(np.int64))[None, :, None]
    tensors = [torch.from_numpy(a.astype(np.int64))[None] for a in (k, km, v, vm)]
    expected = G.quantized_attention(qq, qs, *tensors, torch.tensor(I.EXP_TABLE))
    np.testing.assert_array_equal(out, expected[0, :, 0].numpy())


def test_reject_without_output_mutation(lib, reference):
    I, _ = reference
    q = np.zeros((2, 64), dtype=np.int8)
    k = np.zeros((1, 2, 64), dtype=np.int8)
    scales = np.ones(2, dtype=np.int32)
    table = I.EXP_TABLE.astype(np.uint16)
    out = np.full((2, 64), 99, dtype=np.int64)
    args = [q, scales, k, scales, k, scales, table, out]
    for heads, kv, length in [(2, 3, 2), (0, 1, 2), (2, 1, 0), (2, 1, 513)]:
        assert lib.gqa_attention(heads, kv, length, *map(pointer, args)) == -1
    scales[0] = 0
    assert lib.gqa_attention(2, 1, 2, *map(pointer, args)) == -1
    scales[0] = 1
    q[0, 0] = -128
    assert lib.gqa_attention(2, 1, 2, *map(pointer, args)) == -1
    assert np.all(out == 99)
