# tools

## For users: from your model to your board

| Tool | What it does |
|---|---|
| `cdt.py` | **the CDT**: your model, your data and your hardware limits in a TOML file → a binary network trained, folded into integers, written as a C header (`model_symbex.h`) and checked bit by bit against its integer simulation |
| `chip_test.py` | runs an exported model on a board: equal to the simulation (must be 100 %), accuracy, latency per layer, RAM |
| `configs/example.toml` | every CDT option, commented: start here |

```
python cdt.py my_model.toml                                     # train -> export -> check
python cdt.py my_model.toml --board esp32 --port /dev/ttyUSB0   # ... and measure on the board
python chip_test.py <output folder> --board s3 --port /dev/ttyUSB0
```

The header runs with `lib_symbex` in any project: see `Symbex1_PIO/src/main.cpp` for the minimal example.

## For development: measuring the chips and the library

| Tool | What it does |
|---|---|
| `count_bench.py` | the raw arithmetic of a chip on the same convolution pattern: every bit counting method against int8 multiply-accumulate; saves `chip_profiles/count_<chip>.json` |
| `fine_profile.py` | time per library function on the chip (CPU cycle counters inside lib_symbex); `--plain` gives the latency per layer without counters |
| `chip_profiles/` | the measured profiles of each chip; the sizing studies of the benchmarks predict latency from them |
| `compiler_core/` | the library behind the CDT: methods, training, export, validation, chip harness |

```
python count_bench.py --board s3 --port /dev/ttyUSB0
python fine_profile.py <export folder> --board esp32 --port /dev/ttyUSB0 --popcount lut16
```

The tests: `python ../Symbex1_PIO/lib/lib_symbex/tests/run_tests.py` (the C library, three hardware layers) and `pytest`
from the repository root (the Python export and the C engine against numpy references).
