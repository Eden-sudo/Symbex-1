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


lb, blob = load("littlebit"), load("littlebit_blob")


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    library = tmp_path_factory.mktemp("littlebit_c") / "littlebit.dylib"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-shared", "-fPIC", str(ROOT / "c/littlebit_blob.c"), "-o", str(library)],
                   check=True, capture_output=True, timeout=15)
    lib = ctypes.CDLL(str(library))
    fn = lib.littlebit_blob_matvec
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return fn


def call(kernel, data, q, outputs):
    # Deliberately unaligned blob address and canaries around output.
    raw = ctypes.create_string_buffer(b"!" + data)
    x = (ctypes.c_int8 * len(q))(*q)
    y = (ctypes.c_int32 * (outputs + 2))(*([12345] * (outputs + 2)))
    status = kernel(ctypes.byref(raw, 1), len(data), x, len(q), ctypes.byref(y, 4), outputs)
    assert y[0] == y[-1] == 12345
    assert bytes(raw)[1:1 + len(data)] == data
    return status, list(y)[1:-1]


@pytest.mark.parametrize("residual", [False, True])
def test_bit_exact(kernel, residual):
    gen = torch.Generator().manual_seed(93)
    for n, r, o in ((1, 1, 1), (9, 3, 7), (13, 9, 11)):
        layer = lb.FactorLinear.from_dense(torch.randn(o, n, generator=gen) * 0.2, r, residual)
        with torch.no_grad():
            for p in layer.paths:
                p.g[::2].neg_()
                p.h[::2].neg_()
                p.l[::2].neg_()
        params = blob.integer.integerize(layer)
        data = blob.dumps(params)
        for _ in range(12):
            q = torch.randint(-128, 128, (n,), generator=gen).tolist()
            expected = blob.integer.forward_int(params, q)
            assert call(kernel, data, q, o) == (0, expected)
            assert call(kernel, data, q, o) == (0, expected)


def test_reject_before_write(kernel):
    params = blob.integer.integerize(lb.FactorLinear.from_dense(torch.ones(3, 5), 2))
    data = blob.dumps(params)
    for bad in (data[:20], data[:-1], data + b"x", b"BAD!" + data[4:]):
        assert call(kernel, bad, [1] * 5, 3) == (-1, [12345] * 3)
    assert call(kernel, data, [1] * 4, 3) == (-1, [12345] * 3)
