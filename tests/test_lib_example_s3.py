"""Compile the standalone Arduino example and check its copied headers."""
import os
from pathlib import Path
import shutil
import subprocess


REPO = Path(__file__).resolve().parents[1]
LIB = Path("Symbex1_PIO/lib/lib_symbex")
EXAMPLE = LIB / "examples/s3_mlp"
SOURCE = Path("Symbex1_PIO/src")
HEADERS = ("model_symbex.h", "demo_input.h")


def test_s3_mlp_compiles():
    pio = shutil.which("pio")
    if pio is None:
        pio = os.path.expanduser("~/.platformio/penv/bin/pio")
        assert os.path.isfile(pio) and os.access(pio, os.X_OK), "needs PlatformIO"
    result = subprocess.run(
        [pio, "ci", "--lib", str(LIB), "--board", "esp32-s3-devkitc-1",
         "--project-option=platform=espressif32", "--project-option=framework=arduino",
         str(EXAMPLE)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"PlatformIO compilation failed ({result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_s3_mlp_headers_match_source():
    example = REPO / EXAMPLE
    assert {path.name for path in example.iterdir()} == {"s3_mlp.ino", *HEADERS}
    for path in example.iterdir():
        assert path.is_file() and not path.is_symlink(), f"not a normal file: {path.name}"
    for name in HEADERS:
        assert (example / name).read_bytes() == (REPO / SOURCE / name).read_bytes(), name
