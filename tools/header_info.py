"""
header_info.py -- prints what a SYMBEX-1 exported header holds, without a compiler.

Reads the header the exporter wrote and reports its format (SYMBEX_FORMAT / SYMBEX_FORMAT_MIN), the engine
(from the lib_symbex layer headers it includes), SYMBEX_MLP_N_INPUTS, N_CLASSES, N_STEPS, BYTES_BUFFERS and
the flash bytes of its const arrays (element count x element size, default build where SYMBEX_ROM = const).
Anything it cannot read exactly is reported as null, never guessed.

Usage (run from tools/):
    python header_info.py path/to/model.h            # one "key: value" line per field
    python header_info.py path/to/model.h --json     # the same as one JSON object

Exit code: 0 on success, 1 when the file is not a SYMBEX header or cannot be read, 2 on a usage error.
"""
import argparse
import json
import re
import sys

KEYS = ("header", "format", "format_min", "engine", "n_inputs", "n_classes", "n_steps", "bytes_buffers",
        "flash_bytes")

ELEMENT_BYTES = {"int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4,
                 "int64_t": 8, "uint64_t": 8}

# Ordered: the first include found wins (ternary and AND headers also include the dense one, bool1d the bool one).
ENGINE_INCLUDES = (
    ("layers/dense/symbex_ternary.h", "ternary"),
    ("ops/symbex_and.h", "dense_and"),
    ("layers/dense/symbex_dense.h", "dense_xnor"),
    ("layers/conv/symbex_bool1d.h", "bool1d"),
    ("layers/conv/symbex_bool.h", "boolean"),
)
FORMAT_INCLUDE = "core/symbex_format.h"

INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
FLASH_ARRAY = re.compile(r"\b(?:static\s+)?(?:SYMBEX_ROM|const)\s+(?:const\s+)?(u?int(?:8|16|32|64)_t)\s+(\w+)\s*\[([^\]]*)\]")


class NotSymbexHeader(ValueError):
    """The file is not a SYMBEX header, or it cannot be read."""


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _macro(text, name):
    """The integer a macro is defined as; None when absent, not a plain literal, or defined with two values."""
    defined = re.findall(r"^\s*#\s*define\s+%s\b(.*)$" % name, text, re.M)
    parsed = [re.fullmatch(r"\s+(\d+)[uUlL]*\s*", rest) for rest in defined]
    if not defined or None in parsed:
        return None
    numbers = {int(found.group(1)) for found in parsed}
    return numbers.pop() if len(numbers) == 1 else None


def _format(text, stripped):
    fmt = _macro(stripped, "SYMBEX_FORMAT")
    fmt_min = _macro(stripped, "SYMBEX_FORMAT_MIN")
    for line in re.findall(r"^\s*#\s*if\b[^\n]*", text, re.M):
        found = re.search(r"SYMBEX_FORMAT\s*<\s*(\d+)", line)
        found_min = re.search(r"SYMBEX_FORMAT_MIN\s*>\s*(\d+)", line)
        if fmt is None and found:
            fmt = int(found.group(1))
        if fmt_min is None and found_min:
            fmt_min = int(found_min.group(1))
    return fmt, fmt_min


def _includes(stripped):
    return set(INCLUDE.findall(stripped))


def _engine(stripped):
    included = _includes(stripped)
    found = [engine for path, engine in ENGINE_INCLUDES if path in included]
    dense = {"ternary", "dense_and", "dense_xnor"}
    conv = {"bool1d", "boolean"}
    if set(found) & dense and set(found) & conv:
        return None
    return found[0] if found else None


def _flash_bytes(stripped):
    total = 0
    for ctype, _name, size in FLASH_ARRAY.findall(stripped):
        size = size.strip()
        if not size.isdigit():
            return None
        total += int(size) * ELEMENT_BYTES[ctype]
    return total


def parse_header_text(text):
    """The fields of a header's text, in KEYS order without "header"."""
    stripped = _strip_comments(text)
    included = _includes(stripped)
    known = FORMAT_INCLUDE in included or any(path in included for path, _ in ENGINE_INCLUDES)
    if not known and not re.search(r"^\s*#\s*define\s+SYMBEX_MLP_", stripped, re.M):
        raise NotSymbexHeader("not a SYMBEX header (no core/symbex_format.h include, no SYMBEX_MLP_ macro, "
                              "no lib_symbex layer include)")
    fmt, fmt_min = _format(text, stripped)
    return {
        "format": fmt,
        "format_min": fmt_min,
        "engine": _engine(stripped),
        "n_inputs": _macro(stripped, "SYMBEX_MLP_N_INPUTS"),
        "n_classes": _macro(stripped, "SYMBEX_MLP_N_CLASSES"),
        "n_steps": _macro(stripped, "SYMBEX_MLP_N_STEPS"),
        "bytes_buffers": _macro(stripped, "SYMBEX_MLP_BYTES_BUFFERS"),
        "flash_bytes": _flash_bytes(stripped),
    }


def read_header_info(path):
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError) as error:
        raise NotSymbexHeader(f"cannot read {path}: {error}")
    try:
        fields = parse_header_text(text)
    except NotSymbexHeader as error:
        raise NotSymbexHeader(f"{path} is {error}")
    return {"header": path, **fields}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print what an exported SYMBEX header holds.")
    parser.add_argument("header", help="header written by the exporter")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of lines")
    args = parser.parse_args(argv)
    try:
        info = read_header_info(args.header)
    except NotSymbexHeader as error:
        print(f"header_info: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(info))
    else:
        for key in KEYS:
            print(f"{key}: {'null' if info[key] is None else info[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
