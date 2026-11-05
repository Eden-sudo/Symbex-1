"""Exported headers keep working when lib_symbex grows (the rules of core/symbex_format.h).

Every current exporter (boolean with the thermometer stem, the int8 stem, a per-cell head and a pooled int8 head; bool1d with a binary and
an int8 first layer; dense AND and XNOR; ternary) writes a header that must:
  - fill every library struct by field name (.field = value), never by position;
  - build in C (gcc -std=gnu11) and in C++ (g++ -std=gnu++17, the Pico and Arduino firmware) with
    -Wall -Wextra -Werror, and give the integer simulation's result, image by image.
The headers frozen at earlier formats (lib_symbex/tests/headers/format<N>/) must still build and give their stored
results against the current library: a break shows up here before it is pushed.

Freeze the current format's headers (only when SYMBEX_FORMAT goes up, never to make a failing test pass):
    SYMBEX_FREEZE_HEADERS=1 pytest -q tests/test_header_format.py
"""
import glob
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
import torch

from compiler_core.export import bool1d, boolean, c_header, dense, ternary
from compiler_core.methods.dense.ternary import DenseTernary
from compiler_core.validate.ternary import verify_bit_exact
from test_ternary import _random_ternary_mlp
from test_threshold_folding import C, H, MEAN, STD, W, _build_mlp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")
FROZEN = os.path.join(LIB, "tests", "headers")
SOURCES = [os.path.join(LIB, "src", *p) for p in (("layers", "conv", "symbex_bool.c"), ("layers", "conv", "symbex_bool1d.c"),
                                                  ("layers", "dense", "symbex_dense.c"), ("layers", "dense", "symbex_ternary.c"),
                                                  ("ops", "symbex_and.c"), ("hal", "generic", "symbex_hal_impl.c"))]
WARNINGS = ["-Wall", "-Wextra", "-Werror"]
N = 200

needs_compilers = pytest.mark.skipif(shutil.which("gcc") is None or shutil.which("g++") is None,
                                     reason="gcc / g++ not found")


# ---------------------------------------------------------------------------------------------------
# tiny networks, one per exporter: (header text written to path, uint8 inputs, the simulation's results)
# ---------------------------------------------------------------------------------------------------

def _signs(rng, *shape):
    return np.where(rng.random(shape) < 0.5, -1.0, 1.0)


def _bool_conv(rng, cout, cin, k, stride, spread):
    return dict(w=_signs(rng, cout, cin, k, k), thr=rng.integers(-spread, spread + 1, cout).astype(np.int64),
                k=k, stride=stride)


def _boolean(path, stem, cells=False, seed=0, pool=0):
    """8x8 images: stem -> 8 channels, a block 8 -> 8, a block 8 -> 16 with stride 2, a 1x1 shortcut and the OR pool"""
    rng = np.random.default_rng(seed)
    levels = 2
    if stem == "int8":
        st = dict(w=rng.integers(-127, 128, (8, 3, 3, 3)).astype(np.float64),
                  thr=rng.integers(-3000, 3001, 8).astype(np.int64), k=3, stride=1, int8=True)
    else:
        st = _bool_conv(rng, 8, 3 * levels, 3, 1, 8)
    blocks = [dict(conv=_bool_conv(rng, 8, 8, 3, 1, 6), shortcut=None, pool=False),
              dict(conv=_bool_conv(rng, 16, 8, 3, 2, 6), shortcut=_bool_conv(rng, 16, 8, 1, 1, 2), pool=True)]
    classes = 3
    net = dict(levels=levels, stem=st, blocks=blocks, cells=cells,
               fc_w=_signs(rng, classes, 16 if cells else 16 * 4 * 4),
               fc_scale=rng.integers(1, 2 ** 16, classes).astype(np.int64),
               fc_bias=rng.integers(-2 ** 20, 2 ** 20, classes).astype(np.int64))
    if pool:                                    # the pooled int8 classifier on the 16 x 4 x 4 final map
        net["pool"] = dict(window=pool, side=4, channels=16, shift=16,
                           q=rng.integers(-128, 128, (classes, 16, 4 // pool, 4 // pool)).astype(np.int64))
    boolean.write_header(net, path, in_size=8)
    x = rng.integers(0, 256, (N, 8, 8, 3)).astype(np.uint8)
    return x.reshape(N, -1), boolean.simulate(net, x)


def _bool1d(path, first, seed=0):
    """windows of 24 steps x 3 channels: conv 3 -> 8 (k 3, OR pool 2), conv 8 -> 12 (k 3, stride 2)"""
    rng = np.random.default_rng(seed)
    w0 = rng.integers(-127, 128, (8, 3, 3)).astype(np.float64) if first == "int8" else _signs(rng, 8, 3, 3)
    spread0 = 4000 if first == "int8" else 300
    layers = [dict(w=w0, thr=rng.integers(-spread0, spread0 + 1, 8).astype(np.int64), k=3, stride=1, pool=2),
              dict(w=_signs(rng, 12, 8, 3), thr=rng.integers(-6, 7, 12).astype(np.int64), k=3, stride=2, pool=1)]
    length = ((24 - 2) // 2 - 3) // 2 + 1
    classes = 4
    net = dict(first=first, layers=layers, length=length, fc_w=_signs(rng, classes, length * 12),
               fc_scale=rng.integers(1, 2 ** 16, classes).astype(np.int64),
               fc_bias=rng.integers(-2 ** 20, 2 ** 20, classes).astype(np.int64))
    bool1d.write_header(net, path, in_len=24, in_ch=3)
    x8 = rng.integers(-128, 128, (N, 24, 3))
    return (x8 + 128).astype(np.uint8).reshape(N, -1), bool1d.simulate(net, x8.astype(np.int8))


def _dense(path, arithmetic, seed=0):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    model = _build_mlp(relu=True, bias=True, gen=gen, calib_u8=rng.integers(0, 256, (64, C * H * W), dtype=np.uint8))
    layers = dense.fold(dense.analyze(model, torch.zeros(2, C, H, W)), mean=MEAN, std=STD)
    dense.write_header(layers, path, arithmetic=arithmetic)
    x = rng.integers(0, 256, (N, C * H * W), dtype=np.uint8)
    return x, dense.simulate(layers, x)


def _ternary(path, seed=0):
    m = _random_ternary_mlp(seed)
    x = np.random.default_rng(seed).integers(0, 256, (N, 70)).astype(np.uint8)
    layers = ternary.fold(dense.analyze(m, torch.from_numpy(x[:2]).float() / 255.0, cls=DenseTernary))
    ternary.write_header(layers, path)
    return x, verify_bit_exact(layers, x)["_pred"]


CASES = {"boolean_thermometer": lambda p: _boolean(p, "thermometer"),
         "boolean_int8_stem": lambda p: _boolean(p, "int8"),
         "boolean_cells": lambda p: _boolean(p, "thermometer", cells=True),
         "boolean_pool_head": lambda p: _boolean(p, "int8", pool=2),
         "bool1d_binary": lambda p: _bool1d(p, "binary"),
         "bool1d_int8": lambda p: _bool1d(p, "int8"),
         "dense_and": lambda p: _dense(p, "and"),
         "dense_xnor": lambda p: _dense(p, "xnor"),
         "ternary": _ternary}


# ---------------------------------------------------------------------------------------------------
# build in C and C++ with -Werror, run, compare
# ---------------------------------------------------------------------------------------------------

def _run(header, x_u8, d, lang):
    """the header + lib_symbex built with warnings as errors; main in C or C++ (the library is always C).
    Returns the results it prints, or raises with the compiler's message."""
    inc = os.path.join(LIB, "include")
    shutil.copy(header, os.path.join(d, "model_symbex.h"))
    main = os.path.join(d, "main.c" if lang == "c" else "main.cpp")
    open(main, "w").write(dense._TEST_MAIN)
    objs = []
    for s in SOURCES:
        o = os.path.join(d, os.path.basename(s) + ".o")
        r = subprocess.run(["gcc", "-O2", "-std=gnu11", *WARNINGS, "-I", inc, "-c", s, "-o", o], capture_output=True, text=True)
        assert r.returncode == 0, f"lib_symbex ({os.path.basename(s)}):\n{r.stderr[-2000:]}"
        objs.append(o)
    exe = os.path.join(d, f"check_{lang}")
    cc = ["gcc", "-std=gnu11"] if lang == "c" else ["g++", "-std=gnu++17"]
    r = subprocess.run([*cc, "-O2", *WARNINGS, "-I", inc, "-I", d, main, *objs, "-o", exe], capture_output=True, text=True)
    assert r.returncode == 0, f"the header did not build in {lang} with -Werror:\n{r.stderr[-3000:]}"
    data = os.path.join(d, "x.u8")
    np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(data)
    r = subprocess.run([exe, data, str(len(x_u8))], capture_output=True, text=True)
    assert r.returncode == 0, f"the {lang} program failed while running"
    return np.array([int(v) for v in r.stdout.split()], dtype=np.int64)


_STRUCT = re.compile(r"(?:const\s+)?struct\s+Symbex\w+\s+\w+\s*=\s*\{(.*?)\};", re.S)


def _positional_structs(text):
    """the struct initializers of a header that are not written by field name"""
    return [m.group(0)[:80] for m in _STRUCT.finditer(text) if not m.group(1).strip().startswith(".")]


@needs_compilers
@pytest.mark.parametrize("case", list(CASES))
def test_current_header_builds_in_c_and_cpp_and_equals_the_simulation(case, tmp_path):
    header = str(tmp_path / "model_symbex.h")
    x, expected = CASES[case](header)
    text = open(header).read()
    assert f"SYMBEX_FORMAT < {c_header.FORMAT}" in text and "#pragma GCC diagnostic pop" in text
    assert not _positional_structs(text), f"structs written by position: {_positional_structs(text)}"
    for lang in ("c", "cpp"):
        d = tmp_path / lang
        d.mkdir()
        got = _run(header, x, str(d), lang)
        assert np.array_equal(got, np.asarray(expected, dtype=np.int64)), f"{lang}: {int((got == expected).sum())}/{N}"
    if os.environ.get("SYMBEX_FREEZE_HEADERS"):
        out = os.path.join(FROZEN, f"format{c_header.FORMAT}")
        os.makedirs(out, exist_ok=True)
        shutil.copy(header, os.path.join(out, f"{case}.h"))
        np.savez_compressed(os.path.join(out, f"{case}.npz"), x_u8=x, expected=np.asarray(expected, dtype=np.int64))


def test_the_format_numbers_agree():
    text = open(os.path.join(LIB, "include", "core", "symbex_format.h")).read()
    fmt = int(re.search(r"#define SYMBEX_FORMAT (\d+)", text).group(1))
    fmt_min = int(re.search(r"#define SYMBEX_FORMAT_MIN (\d+)", text).group(1))
    assert fmt == c_header.FORMAT, "the CDT writes a format the library does not call its own"
    assert fmt_min <= fmt
    frozen = sorted(int(os.path.basename(p)[6:]) for p in glob.glob(os.path.join(FROZEN, "format*")))
    assert c_header.FORMAT in frozen, f"no frozen headers of format {c_header.FORMAT}: freeze them (see the docstring)"


FROZEN_HEADERS = sorted(glob.glob(os.path.join(FROZEN, "format*", "*.h")))


@needs_compilers
@pytest.mark.parametrize("header", FROZEN_HEADERS, ids=[os.path.relpath(h, FROZEN) for h in FROZEN_HEADERS])
def test_frozen_header_still_builds_and_gives_its_results(header, tmp_path):
    fmt = int(os.path.basename(os.path.dirname(header))[6:])
    fmt_min = int(re.search(r"#define SYMBEX_FORMAT_MIN (\d+)",
                            open(os.path.join(LIB, "include", "core", "symbex_format.h")).read()).group(1))
    if fmt < fmt_min:
        pytest.skip(f"format {fmt} is older than SYMBEX_FORMAT_MIN {fmt_min}: the header stops with #error, by design")
    data = np.load(header[:-2] + ".npz")
    for lang in ("c", "cpp"):
        d = tmp_path / lang
        d.mkdir()
        got = _run(header, data["x_u8"], str(d), lang)
        assert np.array_equal(got, data["expected"]), f"{lang}: {int((got == data['expected']).sum())}/{len(got)}"
