"""Shared fixtures: seeded RNGs and the lib_symbex C engine compiled for this host (loaded with ctypes)."""
import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from helpers import SymbexLayerData

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_SYMBEX = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")
C_SOURCES = [
    os.path.join(LIB_SYMBEX, "src", "hal", "generic", "symbex_hal_impl.c"),
    os.path.join(LIB_SYMBEX, "src", "layers", "dense", "symbex_dense.c"),
    os.path.join(LIB_SYMBEX, "src", "ops", "symbex_bitpack.c"),
]

# popcount paths of hal/generic, chosen at compile time
POPCOUNT_VARIANTS = {"native": [], "swar": ["-DSYMBEX_POPCOUNT_FORCE_SWAR"],
                     "lut16": ["-DSYMBEX_POPCOUNT_FORCE_LUT16"]}


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(0)


def _declare(lib):
    u32p, i32p, u8p = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32),
                       ctypes.POINTER(ctypes.c_uint8))
    layer_p = ctypes.POINTER(SymbexLayerData)
    signatures = {
        "symbex_xnor_popcount": ([u32p, u32p, ctypes.c_int, ctypes.c_int], ctypes.c_int32),
        "symbex_hal_popcount32": ([ctypes.c_uint32], ctypes.c_int32),
        "symbex_dense_forward": ([u32p, i32p, layer_p], None),
        "symbex_dense_input_u8": ([u8p, i32p, layer_p], None),
        "symbex_dense_input_u8_mask": ([u8p, i32p, layer_p], None),
        "symbex_dense_input_u8_planes": ([u8p, u32p, i32p, layer_p], None),
        "symbex_dense_threshold":([i32p, i32p, u32p, ctypes.c_uint16], None),
        "symbex_bitpack_from_scores": ([i32p, u32p, ctypes.c_uint16], None),
    }
    for name, (args, res) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    return lib


@pytest.fixture(scope="session")
def c_libs(tmp_path_factory):
    """{variant: ctypes library}, one per popcount path of hal/generic. Skipped without gcc."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found: C engine tests skipped")
    d = tmp_path_factory.mktemp("lib_symbex")
    libs = {}
    for name, flags in POPCOUNT_VARIANTS.items():
        out = str(d / f"libsymbex_{name}.so")
        cmd = ["gcc", "-O2", "-std=c99", "-shared", "-fPIC", "-I", os.path.join(LIB_SYMBEX, "include")] \
            + flags + C_SOURCES + ["-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            pytest.fail(f"lib_symbex did not compile ({name}):\n{r.stderr[-2000:]}")
        libs[name] = _declare(ctypes.CDLL(out))
    return libs


@pytest.fixture(params=sorted(POPCOUNT_VARIANTS))
def c_lib(request, c_libs):
    return c_libs[request.param]
