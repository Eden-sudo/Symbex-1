---
name: symbex-exactness-check
description: Proves a SYMBEX-1 kernel, export or model is bit-exact, first on the host and then on a board. Use before claiming a change to `lib_symbex`, the exporter or a model is correct, and before any latency measurement.
---

# SYMBEX-1 exactness check

The project's rule: the board gives the same output as the integer simulation on 100 % of the inputs
(`docs/methodology.md`, "On the chip"). Report host and board results separately; a host pass is not a board pass.

## 1. Host

From the repository root (needs gcc and numpy; benchmarks use `benchmarks/env_bench`):

```sh
python3 Symbex1_PIO/lib/lib_symbex/tests/run_tests.py            # every HAL configuration
python3 Symbex1_PIO/lib/lib_symbex/tests/run_tests.py --only <regex> --cases 200
python3 Symbex1_PIO/lib/lib_symbex/tests/run_tests.py --stress   # sanitizers
python -m pytest -q
```

Known blind spots: the host replaces the ESP32-S3 PIE assembly with C (#18), and random cases can miss sentinel
thresholds such as `INT_MIN` (#11); add those cases to the test when the change touches thresholds.

## 2. Board

A CDT output folder (with `model_symbex.h` and the validation files) on a board:

```sh
python3 tools/chip_test.py <cdt output folder> --board <board> --port <port>
```

Options that change what is measured: `--weights {flash,ram}`, `--popcount {lut16,swar}`,
`--first-layer {mask,planes,simple}`, `--engine {standard,vertical,vertical_prepared,vertical_compact}`,
`--code {flash,ram}`, `--flash-config {default,fast}`, `--samples N` (default 1000), `--usb` for the S3 native port,
`--no-upload` when the firmware is already flashed, `--no-int8` to skip the INT8 reference. `--kernels` measures
only the arithmetic of the chip. The board choices are in `tools/chip_test.py` (`BOARDS`).

Flashing a board needs your human's approval; the shared S3 also needs a board request
([symbex-sync](../symbex-sync/SKILL.md), section 2).

## 3. Report

Per board: inputs checked, equal count (must be all), options used, commit. Anything not run is listed as pending,
with the reason.
