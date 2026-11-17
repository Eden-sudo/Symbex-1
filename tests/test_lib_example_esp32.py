import os
from pathlib import Path
import shutil
import subprocess


REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "Symbex1_PIO/lib/lib_symbex/examples/esp32_mlp"
SOURCE = REPO / "Symbex1_PIO/src"


def test_esp32_example_compiles():
    pio = shutil.which("pio")
    if not pio:
        candidate = os.path.expanduser("~/.platformio/penv/bin/pio")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            pio = candidate
    assert pio, "needs PlatformIO"
    result = subprocess.run(
        [
            pio, "ci", "--lib", "Symbex1_PIO/lib/lib_symbex",
            "--board", "esp32dev", "--project-option=platform=espressif32",
            "--project-option=framework=arduino",
            "Symbex1_PIO/lib/lib_symbex/examples/esp32_mlp",
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"PlatformIO exited {result.returncode}\n{result.stdout}\n{result.stderr}"
    )


def test_example_headers_match_sources():
    names = {"esp32_mlp.ino", "model_symbex.h", "demo_input.h"}
    assert {p.name for p in EXAMPLE.iterdir()} == names
    for name in names:
        assert not (EXAMPLE / name).is_symlink(), name
    for name in ("model_symbex.h", "demo_input.h"):
        assert (EXAMPLE / name).read_bytes() == (SOURCE / name).read_bytes(), name


def test_sketch_contract():
    sketch = (EXAMPLE / "esp32_mlp.ino").read_text()
    for include in ('<Arduino.h>', '<symbex.h>', '"model_symbex.h"', '"demo_input.h"'):
        assert f"#include {include}" in sketch
    assert "pio ci" in sketch and "temporary project" in sketch
    assert "Serial.begin(115200)" in sketch
    assert sketch.count("symbex_hal_name()") == 1
    setup = sketch.split("void setup()", 1)[1].split("void loop()", 1)[0]
    assert setup.index("symbex_hal_name()") < setup.index("run_demo()")
    assert "i < DEMO_N_IMAGES" in sketch
    assert "const uint32_t t0 = micros();\n        const int cls = symbex_mlp_infer(demo_images[i]);\n        const uint32_t us = micros() - t0;" in sketch
    assert "if (us < best_us) best_us = us;" in sketch
    assert "equal += (cls == demo_expected[i]);" in sketch
    assert "equal to the PC simulation: OK" in sketch
    assert "equal to the PC simulation: [X] MISMATCH" in sketch
    assert 'Serial.print("fastest inference: ");\n    Serial.print(best_us);' in sketch
