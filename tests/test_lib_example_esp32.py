"""The standalone ESP32 example builds with the application's demo data."""
from pathlib import Path
import subprocess

import pytest


def test_esp32_example_compiles():
    root = Path(__file__).resolve().parents[1]
    pio = Path('/Users/obed/.platformio/penv/bin/pio')
    if not pio.exists():
        pytest.skip('PlatformIO is not installed')
    cmd = [
        str(pio), 'ci',
        '--lib', 'Symbex1_PIO/lib/lib_symbex',
        '--board', 'esp32dev',
        '--project-option=platform=espressif32',
        '--project-option=framework=arduino',
        'Symbex1_PIO/lib/lib_symbex/examples/esp32_mlp',
    ]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr


def test_example_demo_headers_match_sources():
    root = Path(__file__).resolve().parents[1] / 'Symbex1_PIO'
    example = root / 'lib/lib_symbex/examples/esp32_mlp'
    for name in ('model_symbex.h', 'demo_input.h'):
        assert (example / name).read_bytes() == (root / 'src' / name).read_bytes()
