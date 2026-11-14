"""symbex_bool.c builds with the host compiler and -Wall -Wextra -Werror, with and without the SMLAD path.

stem3_ld32 is only called where __ARM_FEATURE_SIMD32 or SYMBEX_HAL_TEST_SMLAD is defined; defined anywhere else it is an unused
function, which clang (macOS, where gcc is clang) turns into an error under -Werror.
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")
SOURCE = os.path.join(LIB, "src", "layers", "conv", "symbex_bool.c")
WARNINGS = ["-Wall", "-Wextra", "-Werror"]


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not found")
@pytest.mark.parametrize("defines", [[], ["-DSYMBEX_HAL_TEST_SMLAD"]], ids=["plain", "smlad"])
def test_symbex_bool_builds_with_warnings_as_errors(defines, tmp_path):
    r = subprocess.run(["gcc", "-O2", "-std=gnu11", *WARNINGS, *defines, "-I", os.path.join(LIB, "include"), "-c", SOURCE,
                        "-o", str(tmp_path / "symbex_bool.o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
