"""c_build.check honours $CC (split like a shell), else gcc, else cc, and says which compiler it looked for."""
import os
import shlex
import shutil

import numpy as np
import torch
import torch.nn as nn

from compiler_core.export import c_build
from compiler_core.export.dense import analyze
from compiler_core.export.ternary import fold, validate_c, write_header
from compiler_core.methods import apply_plan
from compiler_core.methods.dense.ternary import DenseTernary
from compiler_core.validate.ternary import verify_bit_exact

LIB_SYMBEX = os.path.join(os.path.dirname(__file__), "..", "Symbex1_PIO", "lib", "lib_symbex")


def _tiny_ternary_case(tmp_path):
    """Folded 70-45-10 ternary MLP written to tmp_path: (header, inputs, expected classes)."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Flatten(), nn.Linear(70, 45, bias=False), nn.BatchNorm1d(45), nn.ReLU(), nn.Linear(45, 10))
    m = apply_plan(m, {"1": "bin", "4": "bin"}, bin_method="ternary")
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.running_mean.normal_()
            mod.running_var.uniform_(0.5, 2)
            mod.weight.data.normal_()
        if hasattr(mod, "beta"):
            mod.beta.data.normal_(0, 0.5)
    m.eval()
    xu8 = np.random.default_rng(0).integers(0, 256, (50, 70)).astype(np.uint8)
    layers = fold(analyze(m, torch.from_numpy(xu8[:2]).float() / 255.0, cls=DenseTernary))
    header = str(tmp_path / "tern.h")
    write_header(layers, header)
    return header, xu8, verify_bit_exact(layers, xu8)["_pred"]


def _wrapper(tmp_path):
    """A compiler wrapper in a folder with a space: logs its argv, then runs the real gcc. Returns (path, log)."""
    folder = tmp_path / "dir with space"
    folder.mkdir()
    log = tmp_path / "argv.log"
    wrapper = folder / "cc-wrap"
    wrapper.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{log}'\nexec '{shutil.which('gcc')}' \"$@\"\n")
    wrapper.chmod(0o755)
    return wrapper, log


def _dummy_check():
    return c_build.check("unused.h", "/nonexistent", np.zeros((1, 4), np.uint8), np.zeros(1), [])


def test_cc_wrapper_is_used_and_gives_the_same_result_as_gcc(tmp_path, monkeypatch):
    header, xu8, expected = _tiny_ternary_case(tmp_path)
    monkeypatch.delenv("CC", raising=False)
    baseline = validate_c(header, LIB_SYMBEX, xu8, expected)
    wrapper, log = _wrapper(tmp_path)
    monkeypatch.setenv("CC", shlex.quote(str(wrapper)))
    result = validate_c(header, LIB_SYMBEX, xu8, expected)
    assert result == baseline == (len(xu8), len(xu8), "")
    args = log.read_text().splitlines()
    assert "-O2" in args and "-std=c99" in args
    assert any(a.endswith("main.c") for a in args)


def test_cc_with_several_words_is_split_like_a_shell(tmp_path, monkeypatch):
    header, xu8, expected = _tiny_ternary_case(tmp_path)
    wrapper, log = _wrapper(tmp_path)
    monkeypatch.setenv("CC", f"{shutil.which('sh')} {shlex.quote(str(wrapper))}")
    assert validate_c(header, LIB_SYMBEX, xu8, expected) == (len(xu8), len(xu8), "")
    assert log.read_text().strip()


def test_missing_cc_program_returns_none_and_names_it(monkeypatch):
    monkeypatch.setenv("CC", "symbex-no-such-cc-xyz --flag")
    equal, total, msg = _dummy_check()
    assert equal is None and total == 0
    assert "symbex-no-such-cc-xyz" in msg and "CC" in msg


def test_unsplittable_cc_returns_none(monkeypatch):
    monkeypatch.setenv("CC", '"unterminated')
    equal, total, msg = _dummy_check()
    assert equal is None and total == 0
    assert "CC" in msg


def test_without_cc_gcc_is_used_unchanged(tmp_path, monkeypatch):
    header, xu8, expected = _tiny_ternary_case(tmp_path)
    monkeypatch.delenv("CC", raising=False)
    assert validate_c(header, LIB_SYMBEX, xu8, expected) == (len(xu8), len(xu8), "")
    monkeypatch.setenv("CC", "  ")
    assert validate_c(header, LIB_SYMBEX, xu8, expected) == (len(xu8), len(xu8), "")


def test_without_cc_falls_back_to_cc(tmp_path, monkeypatch):
    header, xu8, expected = _tiny_ternary_case(tmp_path)
    real_which = shutil.which
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(c_build.shutil, "which", lambda n, *a, **k: None if n == "gcc" else real_which(n, *a, **k))
    assert validate_c(header, LIB_SYMBEX, xu8, expected) == (len(xu8), len(xu8), "")


def test_without_cc_and_no_compiler_names_both(monkeypatch):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(c_build.shutil, "which", lambda *a, **k: None)
    equal, total, msg = _dummy_check()
    assert equal is None and total == 0
    assert "gcc" in msg and "cc" in msg
