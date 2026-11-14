"""tools/header_info.py reads an exported header without a compiler (no gcc needed, nothing here skips)."""
import glob
import json
import os
import re
import subprocess
import sys

import pytest

import header_info

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex", "tests", "headers")
HEADERS = sorted(glob.glob(os.path.join(FROZEN, "format*", "*.h")))
TOOL = os.path.join(REPO, "tools", "header_info.py")
DENSE_AND = os.path.join(FROZEN, "format1", "dense_and.h")
SIZES = {"int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4, "int64_t": 8,
         "uint64_t": 8}


def _cli(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _int(pattern, text):
    found = re.search(pattern, text)
    return int(found.group(1)) if found else None


def test_there_are_frozen_headers():
    assert HEADERS


@pytest.mark.parametrize("path", HEADERS, ids=[os.path.relpath(h, FROZEN) for h in HEADERS])
def test_frozen_header_reports_its_own_macros(path):
    text = _read(path)
    info = header_info.read_header_info(path)
    assert info["format"] == _int(r"SYMBEX_FORMAT\s*<\s*(\d+)", text)
    assert info["format_min"] == _int(r"SYMBEX_FORMAT_MIN\s*>\s*(\d+)", text)
    assert info["n_inputs"] == _int(r"#define SYMBEX_MLP_N_INPUTS\s+(\d+)", text)
    assert info["n_classes"] == _int(r"#define SYMBEX_MLP_N_CLASSES\s+(\d+)", text)
    assert info["n_steps"] == _int(r"#define SYMBEX_MLP_N_STEPS\s+(\d+)", text)
    assert info["bytes_buffers"] == _int(r"#define SYMBEX_MLP_BYTES_BUFFERS\s+(\d+)", text)
    assert info["engine"] is not None
    assert isinstance(info["flash_bytes"], int) and info["flash_bytes"] > 0


def test_engine_of_each_exporter():
    expected = {"dense_and.h": "dense_and", "dense_xnor.h": "dense_xnor", "ternary.h": "ternary",
                "bool1d_int8.h": "bool1d", "boolean_thermometer.h": "boolean"}
    for name, engine in expected.items():
        path = os.path.join(FROZEN, "format1", name)
        if os.path.exists(path):
            assert header_info.read_header_info(path)["engine"] == engine, name


def test_flash_bytes_equal_an_independent_count():
    text = _read(DENSE_AND)
    total = 0
    for ctype, body in re.findall(r"static SYMBEX_ROM (\w+) \w+\[[^\]]*\][^=;]*=\s*\{(.*?)\};", text, re.S):
        elements = [item for item in body.split(",") if item.strip()]
        total += len(elements) * SIZES[ctype]
    assert total == 1412
    assert header_info.read_header_info(DENSE_AND)["flash_bytes"] == total


def test_cli_text_and_json_agree():
    path = HEADERS[0]
    info = header_info.read_header_info(path)
    as_json = _cli(path, "--json")
    assert as_json.returncode == 0
    assert tuple(json.loads(as_json.stdout)) == header_info.KEYS
    assert json.loads(as_json.stdout) == info
    as_text = _cli(path)
    assert as_text.returncode == 0
    lines = dict(line.split(": ", 1) for line in as_text.stdout.splitlines())
    assert lines == {key: "null" if info[key] is None else str(info[key]) for key in header_info.KEYS}


def test_missing_macro_is_null_not_guessed(tmp_path):
    original = header_info.read_header_info(DENSE_AND)
    lines = [line for line in _read(DENSE_AND).splitlines() if "SYMBEX_MLP_N_STEPS" not in line]
    copy = tmp_path / "no_steps.h"
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info = header_info.read_header_info(str(copy))
    assert info["n_steps"] is None
    for key in header_info.KEYS:
        if key not in ("header", "n_steps"):
            assert info[key] == original[key], key


def test_non_symbex_file_exits_nonzero_with_a_clear_message(tmp_path):
    other = tmp_path / "other.h"
    other.write_text("int x[4] = {1,2,3,4};\n", encoding="utf-8")
    result = _cli(str(other))
    assert result.returncode != 0
    assert "not a SYMBEX header" in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_missing_file_exits_nonzero(tmp_path):
    result = _cli(str(tmp_path / "absent.h"))
    assert result.returncode != 0
    assert "cannot read" in result.stderr
    assert "Traceback" not in result.stderr
