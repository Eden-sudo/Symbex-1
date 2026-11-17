"""The host exactness check selects its compiler once per call."""
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from compiler_core.export import c_build

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "Symbex1_PIO" / "lib" / "lib_symbex"
FROZEN = LIB / "tests" / "headers" / "format1" / "dense_xnor"
SOURCES = ["layers/dense/symbex_dense.c", "ops/symbex_and.c", "hal/generic/symbex_hal_impl.c"]


@pytest.fixture
def model():
    with np.load(str(FROZEN) + ".npz") as data:
        return data["x_u8"], data["expected"]


def check(model, **kwargs):
    x, expected = model
    return c_build.check(str(FROZEN) + ".h", str(LIB), x, expected, SOURCES, std="c99", **kwargs)


def wrapper(tmp_path):
    assert shutil.which("sh"), "needs sh on PATH"
    compiler = shutil.which("gcc")
    assert compiler, "needs gcc on PATH"
    log = tmp_path / "compiler.log"
    script = tmp_path / "compiler wrapper.sh"
    script.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" >> " + shlex.quote(str(log)) + "\nexec \"$@\"\n")
    script.chmod(0o755)
    command = " ".join(map(shlex.quote, (str(script), compiler, "-DCC_ENV_TEST=1")))
    return command, compiler, log


def test_wrapper_compiles_frozen_model(monkeypatch, tmp_path, model):
    command, compiler, log = wrapper(tmp_path)
    monkeypatch.delenv("CC", raising=False)
    baseline = check(model)
    monkeypatch.setenv("CC", command)
    result = check(model)
    total = len(model[1])
    assert total > 0
    assert baseline == result == (total, total, "")
    args = log.read_text().splitlines()
    assert args[:5] == [compiler, "-DCC_ENV_TEST=1", "-O2", "-std=c99", "-I"]


def test_explicit_missing_does_not_fallback(monkeypatch, model):
    monkeypatch.setenv("CC", "/nonexistent/cc")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    assert check(model) == (None, 0, "/nonexistent/cc not found: the C exactness check was skipped")


def test_fallback_to_cc(monkeypatch, model):
    monkeypatch.delenv("CC", raising=False)
    which = shutil.which
    assert which("cc"), "needs cc on PATH"
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "gcc" else which(name))
    run = subprocess.run
    commands = []

    def spy(args, **kwargs):
        commands.append(args)
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    total = len(model[1])
    assert total > 0 and check(model) == (total, total, "")
    assert commands[0][0] == "cc"


@pytest.mark.parametrize("value", [None, ""])
def test_default_prefers_gcc(monkeypatch, model, value):
    if value is None:
        monkeypatch.delenv("CC", raising=False)
    else:
        monkeypatch.setenv("CC", value)
    which = shutil.which
    assert which("gcc"), "needs gcc on PATH"
    lookups = []

    def lookup(name):
        lookups.append(name)
        return which(name)

    monkeypatch.setattr(shutil, "which", lookup)
    run = subprocess.run
    commands = []

    def spy(args, **kwargs):
        commands.append(args)
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    total = len(model[1])
    assert total > 0 and check(model) == (total, total, "")
    assert lookups == ["gcc"]
    assert commands[0][0] == "gcc"


def test_both_defaults_missing(monkeypatch, model):
    monkeypatch.delenv("CC", raising=False)
    lookups = []
    monkeypatch.setattr(shutil, "which", lambda name: lookups.append(name))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    assert check(model) == (None, 0, "cc not found: the C exactness check was skipped")
    assert lookups == ["gcc", "cc"]


def test_non_executable_compiler(monkeypatch, tmp_path, model):
    path = tmp_path / "cc"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o644)
    monkeypatch.setenv("CC", str(path))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    assert check(model) == (None, 0, f"{path} not found: the C exactness check was skipped")


@pytest.mark.parametrize("value", ["   ", "'unterminated"])
def test_invalid_cc(monkeypatch, model, value):
    monkeypatch.setenv("CC", value)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    with pytest.raises(ValueError, match="CC must contain a compiler command" if value.isspace() else "No closing quotation"):
        check(model)


def test_compiler_selection_lasts_through_after(monkeypatch, tmp_path, model):
    command, compiler, log = wrapper(tmp_path)
    monkeypatch.setenv("CC", command)

    def after(build, folder):
        monkeypatch.setenv("CC", "/nonexistent/cc")
        Path(folder, "extra.c").write_text("int main(void) { return 0; }\n")
        assert build("extra.c", "extra") is None
        return ""

    total = len(model[1])
    assert total > 0 and check(model, after=after) == (total, total, "")
    assert log.read_text().splitlines().count(compiler) == 2
    assert check(model) == (None, 0, "/nonexistent/cc not found: the C exactness check was skipped")


def test_build_failure_keeps_stderr_tail(monkeypatch, tmp_path, model):
    assert shutil.which("sh"), "needs sh on PATH"
    script = tmp_path / "failing-cc"
    diagnostic = "discarded" + "x" * 2100 + "distinctive failure\n"
    script.write_text("#!/bin/sh\nprintf '%s' " + shlex.quote(diagnostic) + " >&2\nexit 1\n")
    script.chmod(0o755)
    monkeypatch.setenv("CC", str(script))
    assert check(model) == (None, 0, "the C did not compile:\n" + diagnostic[-2000:])
