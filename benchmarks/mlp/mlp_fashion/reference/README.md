# mlp_fashion reference model (Fashion-MNIST)

One ready model, so that anyone can build it into a project or measure it on a board **without training**: seed 42 of
the MLP sized for binary, 784-128-512-10 ([README](../README.md#3-sizing-the-mlp-for-binary)), next to the INT8 model of
the same network.

## The model

| | SYMBEX (`model_symbex.h`) | INT8 (`fashion_int8_h128_512.tflite`) |
|---|---|---|
| Network | 784-128-512-10, every layer 1 bit (the first takes the 8-bit pixels) | 784-128-512-10, INT8 |
| Training | the CDT, method `plain`, 15 epochs, labels only, seed 42, from `model_init.pth` | post-training quantization of `model_init.pth` (the FP32 model, test 88.92 %) |
| Validation accuracy (6000 images from the training data) | 87.60 % | — |
| Test accuracy (10000 images, used once) | 87.50 % | 89.60 % (first 2000 test images) |
| Weights + thresholds | 27.3 KB | 187.0 KB |
| Exact | the compiled C equals the integer simulation on 100 % of the images, every board | TFLite Micro equals the PC on 100 % |

The three seeds together (the result, not this one model): test 87.47 ± 0.09 %.

## On the boards (seed 42, median of 1000 images, counting mode `swar`: no lookup table)

| Board | Weights | SYMBEX ms | RAM KB | INT8 TFLite Micro ms |
|---|---|---|---|---|
| ESP32-S3 | flash | **0.420** | 4.7 | 11.67 |
| ESP32-S3 | RAM | **0.413** | 31.9 | 0.89 |
| ESP32 | flash | **3.64** | 4.4 | 31.11 |
| ESP32 | RAM | **2.19** | 31.6 | — |
| Pico 2 Cortex-M33 | flash | **4.65** | 4.7 | 31.35 (RAM: 2.89) |
| Pico 2 Hazard3 | flash | **4.41** | 4.8 | 37.53 (RAM: 9.61) |

Every SYMBEX row was 100 % equal to the simulation (measured Nov 6, 2026, from this folder). RAM is what the board
reports (buffers, input, stack, and the weights when in RAM); the INT8 arena is 6.7 KB.

## Files

| File | What it is | sha256 (first 12) |
|---|---|---|
| `model_symbex.h` | the network for lib_symbex, ready to compile | `ffb1b7f403ce` |
| `model_binarized.pth` | the trained weights (PyTorch) | `b91c7a1d0985` |
| `model_init.pth` | the FP32 starting model (`../models/fashion_mlp_128_512.pth`) | `86a2c40513ed` |
| `train.json`, `estimate.json` | what the CDT chose and measured | `34a365b7b24e` (train) |
| `export.json` | what the export measured (rewritten by step 1) | — |
| `config.toml` | the CDT configuration, paths relative to `tools/` | — |
| `fashion_int8_h128_512.tflite` | the INT8 model of the same network | `9bd8175b3064` |

## Use it

To build it into a project, copy `model_symbex.h` next to your code and follow the
[library guide](../../../../docs/library.md).

To measure it on a board, from `tools/` (environment `env_bench`, Fashion-MNIST in `benchmarks/data/fashionmnist`):

```
# 1. rebuild the 2000 validation images and the integer simulation's answers (no training; same bytes as the run)
python cdt.py ../benchmarks/mlp/mlp_fashion/reference/config.toml --from export --to export
# 2. on a board (check the port first)
python chip_test.py ../benchmarks/mlp/mlp_fashion/reference --board s3 --port /dev/ttyACM0 --usb --popcount swar
python chip_test.py ../benchmarks/mlp/mlp_fashion/reference --board s3 --port /dev/ttyACM0 --usb --popcount swar --weights ram
python chip_test.py ../benchmarks/mlp/mlp_fashion/reference --board esp32 --port /dev/ttyUSB0 --popcount swar
python chip_test.py ../benchmarks/mlp/mlp_fashion/reference --board pico2_arm --port /dev/ttyACM1 --popcount swar
```

The INT8 model on a board: `python fashion_tflite.py --board s3 --port /dev/ttyACM0` from the benchmark's folder (see
its README for the 784-128-512 shape).
