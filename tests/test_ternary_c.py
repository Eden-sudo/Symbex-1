import ctypes
import importlib.util
from pathlib import Path
import subprocess

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


codec, integer = load("ternary_pack"), load("ternary_int")


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    lib = tmp_path_factory.mktemp("ternary") / "ternary.so"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-Wall", "-Wextra", "-Werror",
                    "-O2", "-shared", "-fPIC", str(ROOT / "c/ternary.c"), "-o", str(lib)],
                   check=True, capture_output=True)
    fn = ctypes.CDLL(str(lib)).ternary_matvec_base3
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t,
                   ctypes.c_size_t, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64)]
    fn.restype = ctypes.c_int
    return fn


@pytest.mark.parametrize("cols", [1, 4, 5, 6, 17, 256])
def test_exact_host_matvec(kernel, cols):
    g = torch.Generator().manual_seed(cols)
    w = torch.randint(-1, 2, (7, cols), generator=g, dtype=torch.int8)
    w[0].zero_()
    w[1].fill_(-1)
    data = b"".join(codec.pack(row.tolist()) for row in w)
    packed = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    for x in [torch.zeros(cols, dtype=torch.int8), torch.full((cols,), -128, dtype=torch.int8),
              torch.full((cols,), 127, dtype=torch.int8),
              torch.randint(-128, 128, (cols,), generator=g, dtype=torch.int8)]:
        inputs = (ctypes.c_int8 * cols)(*x.tolist())
        out = (ctypes.c_int64 * 7)(*([999] * 7))
        assert kernel(packed, len(data), 7, cols, inputs, out) == 0
        assert list(out) == integer.accumulate(x, w).tolist()


@pytest.mark.parametrize("data,rows,cols,length", [(b"\x00\xff", 2, 5, 2),
    (b"\x00\x03", 2, 1, 2), (b"\x00", 1, 5, 0), (b"\x00", 0, 5, 1),
    (b"\x00", 1, 0, 1)])
def test_reject_before_output_write(kernel, data, rows, cols, length):
    packed = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    x = (ctypes.c_int8 * 5)(1, 2, 3, 4, 5)
    out = (ctypes.c_int64 * 2)(1234, 5678)
    assert kernel(packed, length, rows, cols, x, out) == -1
    assert list(out) == [1234, 5678]
