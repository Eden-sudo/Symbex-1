"""Every host HAL configuration builds all library sources with warnings as errors."""
from pathlib import Path
import shutil
import subprocess

import pytest


LIB = Path(__file__).resolve().parents[1] / "Symbex1_PIO" / "lib" / "lib_symbex"
CONFIGS = {
    "generic": [],
    "generic-native": ["SYMBEX_POPCOUNT_FORCE_NATIVE"],
    "generic-bytewise": ["SYMBEX_U8_BYTEWISE", "SYMBEX_POPCOUNT_FORCE_LUT8", "SYMBEX_PLANES_PLANE_OUTER",
                         "SYMBEX_PLANES_PER_NEURON"],
    "esp32s3-c": ["CONFIG_IDF_TARGET_ESP32S3", "SYMBEX_HAL_TEST_NO_PIE"],
    "rp2350-planes": ["SYMBEX_HAL_TEST_RP2350", "SYMBEX_CODE_IN_RAM"],
    "rp2350-usada8-smlad": ["SYMBEX_HAL_TEST_RP2350", "SYMBEX_RP2350_U8_USADA8", "SYMBEX_HAL_TEST_SMLAD"],
    "generic-swar": ["SYMBEX_POPCOUNT_FORCE_SWAR"],
    "generic-lut16": ["SYMBEX_POPCOUNT_FORCE_LUT16"],
    "generic-no-lut16": ["SYMBEX_POPCOUNT_NO_LUT16"],
    "esp32s3-c-no-lut16": ["CONFIG_IDF_TARGET_ESP32S3", "SYMBEX_HAL_TEST_NO_PIE", "SYMBEX_POPCOUNT_NO_LUT16"],
}


@pytest.mark.parametrize("config", CONFIGS)
def test_all_sources_build_with_warnings_as_errors(config, tmp_path):
    cc = shutil.which("cc")
    assert cc, "needs cc on PATH"
    sources = sorted((LIB / "src").rglob("*.c"))
    assert sources, "no lib_symbex sources found"
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "esp_heap_caps.h").write_text(
        "#pragma once\n#include <stdlib.h>\n#define MALLOC_CAP_INTERNAL 1\n#define MALLOC_CAP_8BIT 2\n"
        "static inline void* heap_caps_malloc(size_t n, int c) { (void)c; return malloc(n); }\n")
    for source in sources:
        relative = source.relative_to(LIB / "src")
        output = tmp_path / "objects" / relative.with_suffix(".o")
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-I", str(LIB / "include"), "-I", str(stubs),
               *[f"-D{define}" for define in CONFIGS[config]], "-c", str(source), "-o", str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (f"{config}: {relative}\ncommand: {cmd}\nreturncode: {result.returncode}\n"
                                        f"{result.stdout}{result.stderr}")
