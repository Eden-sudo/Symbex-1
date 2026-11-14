"""Build and exercise the standalone host example."""
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "Symbex1_PIO/lib/lib_symbex"
SRC = REPO / "Symbex1_PIO/src"
MAIN = LIB / "examples/host_mlp/main.c"


def _compile(tmp_path, include=None):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    executable = tmp_path / "host_mlp"
    cmd = ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror"]
    if include is not None:
        cmd += ["-I", str(include)]
    cmd += ["-I", str(LIB / "include"), "-I", str(SRC), str(MAIN),
            str(LIB / "src/layers/conv/symbex_bool.c"),
            str(LIB / "src/hal/generic/symbex_hal_impl.c"),
            "-o", str(executable)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


def _run(executable):
    return subprocess.run([str(executable)], capture_output=True, text=True,
                          timeout=30)


def _synthetic_headers(tmp_path, mismatch):
    predictions = [4, -3, 7, 9, 12]
    expected = predictions.copy()
    if mismatch is not None:
        expected[mismatch] = 42
    (tmp_path / "demo_input.h").write_text(
        "#define DEMO_N_IMAGES 5\n"
        "static const unsigned char demo_images[5][1] = {{0}, {1}, {2}, {3}, {4}};\n"
        f"static const int demo_expected[5] = {{{', '.join(map(str, expected))}}};\n"
    )
    limit = mismatch + 1 if mismatch is not None else len(predictions)
    (tmp_path / "model_symbex.h").write_text(
        "#include <stdlib.h>\n"
        "static const char *const symbex_mlp_steps[] = {\"test\"};\n"
        "static int symbex_mlp_infer(const unsigned char *image) {\n"
        f"    static const int predictions[5] = {{{', '.join(map(str, predictions))}}};\n"
        "    static int calls = 0;\n"
        f"    if (calls >= {limit} || *image != calls) exit(2);\n"
        "    return predictions[calls++];\n"
        "}\n"
    )
    return predictions, expected


def test_host_mlp_demo(tmp_path):
    result = _run(_compile(tmp_path))
    assert (result.returncode, result.stdout, result.stderr) == (0, "OK 8/8\n", "")


@pytest.mark.parametrize("mismatch", [0, 1, 4])
def test_host_mlp_first_mismatch(tmp_path, mismatch):
    predictions, expected = _synthetic_headers(tmp_path, mismatch)
    result = _run(_compile(tmp_path, tmp_path))
    assert (result.returncode, result.stdout, result.stderr) == (
        1, "", f"Mismatch image {mismatch}: expected {expected[mismatch]}, "
               f"got {predictions[mismatch]}\n")


def test_host_mlp_synthetic_success(tmp_path):
    _synthetic_headers(tmp_path, None)
    result = _run(_compile(tmp_path, tmp_path))
    assert (result.returncode, result.stdout, result.stderr) == (0, "OK 5/5\n", "")
