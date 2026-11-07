# mlp_digits reference model (scikit-learn Digits)

One ready model, so that anyone can build it into a project or measure it on a board **without training**: seed 42 of
the benchmark's main result, a network binary from end to end ([README](../README.md#main-result-fully-binary-labels-only)).

## The model

| | SYMBEX (`model_symbex.h`, `model_symbex_xnor.h`) | INT8 (`digits_int8.tflite`) |
|---|---|---|
| Network | 64-256-10: input, weights and activations all 1 bit | the FP32 original 64-128-10 on grey levels, INT8 |
| Training | the CDT, method `plain`, width x2, labels only, seed 42 | post-training quantization of `model_init.pth` |
| Validation accuracy (288 images from the training data) | 92.01 % | — |
| Test accuracy (360 images, used once) | 95.56 % | 98.33 % |
| Weights + thresholds | 5.5 KB (AND) / 3.4 KB (XNOR) | 14.7 KB |
| Exact | the compiled C equals the integer simulation on 100 % of the images, every board | TFLite Micro equals the PC on 100 % |

The two headers are the same network with two arithmetics: AND (`model_symbex.h`, rows padded for the SIMD multiplier;
fastest on the ESP32-S3 and the Cortex-M33) and XNOR (`model_symbex_xnor.h`; fastest on the ESP32, Hazard3 and the
Arduino Uno). The three seeds together: test 94.54 ± 1.31 %. The INT8 model is not the same network: it is the FP32
original on 17 grey levels, the accuracy ceiling (on 1-bit input an FP32 MLP reaches 94.35 %).

## On the boards (seed 42, median of the 360 test images, counting mode `swar`, weights in flash)

| Board | Arithmetic | SYMBEX ms | RAM KB | INT8 TFLite Micro ms |
|---|---|---|---|---|
| ESP32-S3 | AND | **0.109** | 2.05 | 0.110 |
| ESP32 | XNOR | **0.232** | 0.66 | 0.763 |
| Pico 2 Cortex-M33 | AND | **0.396** | 2.05 | — |
| Pico 2 Hazard3 | XNOR | **0.378** | 1.13 | — |
| Arduino Uno (benchmark README, not re-measured) | XNOR | 5.34 | 0.21 | does not fit |

Every SYMBEX row was 100 % equal to the simulation (measured Nov 6, 2026, from this folder and its XNOR rebuild).

## Files

| File | What it is | sha256 (first 12) |
|---|---|---|
| `model_symbex.h` | the network for lib_symbex, AND arithmetic | `718269dbfeec` |
| `model_symbex_xnor.h` | the same network, XNOR arithmetic | `10d69968fa38` |
| `model_binarized.pth` | the trained weights (PyTorch) | `ee91a1f0c910` |
| `model_init.pth` | the FP32 starting model (`../models/digits_mlp.pth`) | `9dd155035e5f` |
| `train.json`, `estimate.json` | what the CDT chose and measured | `f0ab2e19b648` (train) |
| `export.json`, `export_xnor.json` | what each export measured | — |
| `config.toml` | the CDT configuration, paths relative to `tools/` | — |
| `digits_int8.tflite` | the INT8 model of the FP32 original | `8403755793cc` |

## Use it

To build it into a project, copy the header for your chip next to your code (rename `model_symbex_xnor.h` to
`model_symbex.h`) and follow the [library guide](../../../../docs/en/library.md).

To measure it on a board, from `tools/` (environment `env_bench`, Digits in `benchmarks/data/digits`):

```
# 1. rebuild the 360 test images and the integer simulation's answers (no training; same bytes as the run)
python cdt.py ../benchmarks/mlp/mlp_digits/reference/config.toml --from export --to export
# 2. on a board (check the port first); for XNOR, copy the folder and put model_symbex_xnor.h as model_symbex.h
python chip_test.py ../benchmarks/mlp/mlp_digits/reference --board s3 --port /dev/ttyACM0 --usb --popcount swar
python chip_test.py ../benchmarks/mlp/mlp_digits/reference --board pico2_arm --port /dev/ttyACM1 --popcount swar
```
