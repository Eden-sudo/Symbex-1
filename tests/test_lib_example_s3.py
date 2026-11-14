"""The ESP32-S3 Arduino example of lib_symbex: sketch contents and a PlatformIO CI build."""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")
EXAMPLE = os.path.join(LIB, "examples", "s3_mlp")
SRC = os.path.join(REPO, "Symbex1_PIO", "src")
PIO = "/Users/obed/.platformio/penv/bin/pio"


@pytest.mark.parametrize("name", ["model_symbex.h", "demo_input.h"])
def test_example_headers_are_identical_copies_of_src(name):
    with open(os.path.join(EXAMPLE, name), "rb") as f:
        copy = f.read()
    with open(os.path.join(SRC, name), "rb") as f:
        original = f.read()
    assert copy == original


def test_sketch_prints_computed_exactness_and_time():
    with open(os.path.join(EXAMPLE, "s3_mlp.ino"), encoding="utf-8") as f:
        text = f.read()
    for needle in ["#include <symbex.h>", "Serial.begin(115200)", "micros()",
                   "symbex_mlp_infer(demo_images[", "demo_expected[", "DEMO_N_IMAGES"]:
        assert needle in text, needle
    for literal in re.findall(r'"([^"\\]*)"', text):
        assert not re.search(r"\d+\s*/\s*\d+", literal), literal
        assert not re.search(r"\d+\s*(us|µs|ms)\b", literal), literal


@pytest.mark.skipif(not os.path.exists(PIO), reason=f"{PIO} not found")
def test_example_compiles_for_esp32_s3_with_pio_ci():
    r = subprocess.run(
        [PIO, "ci", "--lib", LIB, "--board", "esp32-s3-devkitc-1",
         "--project-option=platform=espressif32",
         "--project-option=framework=arduino", EXAMPLE],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
