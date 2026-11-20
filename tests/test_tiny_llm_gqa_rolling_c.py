"""Bit-exact C ring output AND cache bytes against the Python ring."""
import ctypes
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


class Ring(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ("heads", "kv_heads", "window", "max_positions")] + [
        ("position", ctypes.c_uint64)] + [(name, ctypes.c_void_p) for name in ("k", "v", "km", "vm", "cos", "sin")]


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    output = tmp_path_factory.mktemp("rolling") / "rolling.so"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(DIR / "c/gqa_attention.c"), str(DIR / "c/gqa_rolling.c"), "-o", str(output)],
                   check=True, capture_output=True, timeout=10)
    library = ctypes.CDLL(str(output))
    library.gqa_ring_step.argtypes = [ctypes.POINTER(Ring)] + [ctypes.c_void_p] * 5
    library.gqa_ring_step.restype = ctypes.c_int
    return library


@pytest.fixture
def modules(monkeypatch):
    loaded = []
    for name in ("train", "intsim", "gqa_intsim", "gqa_rolling"):
        spec = importlib.util.spec_from_file_location(name, DIR / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded[1], loaded[3]


@pytest.mark.parametrize("heads,kv,window", [(1, 1, 1), (4, 1, 3), (6, 2, 4), (4, 4, 5)])
def test_multiple_wraps_and_reset(lib, modules, heads, kv, window):
    I, R = modules
    positions = window * 3 + 2
    C, S = R.rope_tables(64, positions)
    cs = [x.numpy().astype(np.int32) for x in (C, S)]
    arrays = [np.zeros((kv, window, 64), np.int8) for _ in range(2)]
    arrays += [np.ones((kv, window), np.int32) for _ in range(2)]
    state = Ring(heads, kv, window, positions, 0, *map(ptr, arrays + cs))
    exp = I.EXP_TABLE.astype(np.uint16)
    out = np.zeros((heads, 64), np.int64)
    rng = np.random.default_rng(41)
    inputs = [(rng.integers(-(1 << 19), (1 << 19) + 1, (heads, 64), dtype=np.int32),
               rng.integers(-12000, 12001, (kv, 64), dtype=np.int32),
               rng.integers(-12000, 12001, (kv, 64), dtype=np.int32)) for _ in range(positions)]
    inputs[0] = tuple(np.zeros_like(a) for a in inputs[0])
    history = []
    for replay in range(2):
        state.position = 0
        python = R.RollingAttention(heads, kv, window, C, S)
        for pos, (q, k, v) in enumerate(inputs):
            assert lib.gqa_ring_step(ctypes.byref(state), *map(ptr, (q, k, v, exp, out))) == 0
            expected = python.step(*(torch.from_numpy(a.astype(np.int64)) for a in (q, k, v)))
            np.testing.assert_array_equal(out, expected.numpy())
            assert state.position == python.position
            # After reset, unused slots intentionally retain old bytes in C.
            used = min(pos + 1, window)
            for actual, oracle in zip(arrays, (python.k, python.v, python.km, python.vm)):
                np.testing.assert_array_equal(actual[:, :used], oracle.numpy()[:, :used])
            if replay:
                np.testing.assert_array_equal(out, history[pos])
            else:
                history.append(out.copy())
        before = [a.copy() for a in arrays] + [out.copy()]
        assert lib.gqa_ring_step(ctypes.byref(state), *map(ptr, (q, k, v, exp, out))) == -1
        assert state.position == positions
        for a, b in zip(arrays + [out], before):
            np.testing.assert_array_equal(a, b)
    state.position = 0
    q[0, 0] = (1 << 19) + 1
    assert lib.gqa_ring_step(ctypes.byref(state), *map(ptr, (q, k, v, exp, out))) == -1
    assert state.position == 0
