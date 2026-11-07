# HAR reference model (UniMiB-SHAR, 1D convolutions)

One ready model, so that anyone can build it into a project or measure it on a board **without training**. It is
seed 42 of the largest network of Daghero et al.'s Table 1 ("max"), fully binary, next to the INT8 model of the same
network trained with the same recipe ([README](../README.md)).

This benchmark is the one exception to the common method: its networks are Daghero's, so the comparison follows their
paper (same networks, same data split) rather than a network sized for binary.

## The model

| | SYMBEX (`model_symbex.h`) | INT8 (`har_int8_max.tflite`) |
|---|---|---|
| Network | Daghero's "max" 1D CNN, every layer 1 bit, fused OR pool | the same network, INT8 |
| Training | 100 epochs, lr 0.005, labels only, seed 42 (best epoch 85) | the same recipe, then INT8 |
| Validation accuracy (from the training data) | 69.37 % | — |
| Test accuracy (used once) | 59.47 % | 67.48 % |
| Weights + thresholds | 5.6 KB (first layer in bits) / 6.5 KB (±1 bytes) | 42.5 KB |
| Exact | the compiled C equals the integer simulation on 100 % of the windows, every board | — |

The three seeds together (the result, not this one model): SYMBEX 58.8 ± 0.9 %, TFLite INT8 67.9 ± 0.9 %. INT8 is
more accurate here; SYMBEX's case is speed and memory.

## On the boards (seed 42, weights in flash, median of 1000 windows)

The first layer runs in one of two forms with the same weights and outputs: "bits" (`model_symbex.h`) or "±1 bytes"
(`model_symbex_bytes.h`, the int8 kernel).

| Board | Counting | SYMBEX bits ms | SYMBEX ±1 bytes ms | RAM KB | INT8 TFLite Micro ms |
|---|---|---|---|---|---|
| ESP32-S3 | swar | 4.83 | 3.84 | 3.2 | 7.11 (RAM: 6.16) |
| ESP32 | swar | 14.77 | 14.08 | 2.9 | 115.62 (RAM: 102.70) |
| Pico 2 Cortex-M33 | lut16 | 11.60 | 11.71 | 67.2 | — |
| Pico 2 Hazard3 | default (cpop) | 10.90 | 8.59 | 3.4 | — |

RAM is what the board reports for SYMBEX (2.0 KB of buffers, input, stack; on the M33 the 64 KB lut16 table). The
INT8 arena is 10.3 KB (ESP32) / 25.5 KB (S3). Every SYMBEX row was 100 % equal to the simulation.

## Files

| File | What it is | sha256 (first 12) |
|---|---|---|
| `model_symbex.h` | the network for lib_symbex, first layer in bits | `705c4728a39b` |
| `model_symbex_bytes.h` | the same network, first layer as ±1 bytes | `0c386811adf9` |
| `model.pth` | the trained weights (PyTorch) | `129c6a9ad78a` |
| `train.json` | the training settings and its accuracy | `0fff6658f42f` |
| `export.json`, `export_bytes.json` | what the export measured (sizes, exactness) | — (rewritten by step 1) |
| `har_int8_max.tflite` | the INT8 model, same network and recipe, seed 42 | `b5c39623c4bc` |

## Use it

To build it into a project, copy `model_symbex.h` (or `model_symbex_bytes.h`) next to your code and follow the
[library guide](../../../../docs/library.md).

To measure it on a board (environment `env_bench`, UniMiB-SHAR prepared once with `python har_data.py`):

```
cd benchmarks/conv1d/har_ultracompact
# 1. rebuild the 1000 validation windows and the integer simulation's answers (no training)
python har_export.py reference                  # -> reference/
python har_export.py reference --first-bytes    # -> reference_bytes/
# 2. on a board (check the port first), from tools/
cd ../../../tools
python chip_test.py ../benchmarks/conv1d/har_ultracompact/reference --board s3 --port /dev/ttyACM0 --usb --popcount swar
python chip_test.py ../benchmarks/conv1d/har_ultracompact/reference_bytes --board esp32 --port /dev/ttyUSB0 --popcount swar
python chip_test.py ../benchmarks/conv1d/har_ultracompact/reference --board pico2_riscv --port /dev/ttyACM1
```

The INT8 model on a board: `python har_tflite.py --arch max --seed 42 --board s3 --port /dev/ttyACM0 --usb` from the
benchmark's folder.
