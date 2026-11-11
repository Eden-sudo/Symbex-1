# ResNet-8 reference model (CIFAR-10)

One ready model, so that anyone can build it into a project or measure it on a board **without training**. It is
seed 42 of the most accurate recipe in this benchmark ([README](../README.md#the-most-accurate-recipe-int8-stem--bop--pooled-int8-classifier)),
next to the official MLPerf Tiny INT8 model it is compared with.

## The model

| | SYMBEX (`model_symbex.h`) | INT8 (`resnet8_int8.tflite`) |
|---|---|---|
| Network | ResNet-8 16-32-64, int8 first layer, every other convolution 1 bit, pooled int8 classifier | MLPerf Tiny ResNet-8, INT8 |
| Training | from scratch, 150 epochs, labels only, Bop for the binary weights, seed 42 | official, as published |
| Validation accuracy (5000 images from the training data) | 77.72 % | — |
| Test accuracy (10000 images, used once) | 76.87 % | 86.97 % |
| Weights + thresholds | 29.0 KB | 96.2 KB |
| Exact | the compiled C equals the integer simulation on 100 % of the images, every board and engine | TFLite Micro equals the PC on 98.1 % |

The three seeds together (the result, not this one model): validation 77.30 ± 0.45 %, test 76.68 ± 0.34 %.

## On the boards (seed 42, median of 1000 images)

| Board | Engine | Weights | SYMBEX ms | RAM KB | INT8 TFLite Micro ms |
|---|---|---|---|---|---|
| ESP32-S3 | standard (swar) | flash | 32.33 | 15.2 | 51.96 |
| ESP32-S3 | standard (swar) | RAM | 32.32 | 44.2 | — |
| ESP32-S3 | vertical | flash | 9.72 | 54.2 | — |
| ESP32-S3 | vertical_compact | flash | 7.09 | 137.7 | — |
| ESP32-S3 | vertical_prepared | flash | 6.02 | 216.7 | — |
| ESP32 | standard (lut16) | flash | 62.52 | 79.0 | 717.76 |
| ESP32 | standard (lut16) | RAM | 58.23 | 108.0 | — |
| Pico 2 Cortex-M33 | standard (lut16) | flash | 67.77 | 79.9 | 453.02 (RAM: 211.44) |
| Pico 2 Cortex-M33 | vertical_prepared (lut16) | flash | 59.20 | 279.4 | — |
| Pico 2 Hazard3 | standard (lut16) | flash | 57.94 | 15.5 | 5701.88 (reference kernels) |
| Pico 2 Hazard3 | standard (lut16) | RAM | 53.46 | 44.4 | — |

RAM is what the board reports for SYMBEX (buffers, input, stack, the 64 KB lut16 table where used, the weights when
in RAM). The INT8 model runs from flash unless marked; its tensor arena is 53-80 KB. Every SYMBEX row was 100 % equal
to the simulation. On the S3 the counting mode matters: `--popcount swar` (PIE); `chip_test.py`'s default, lut16, gave
33.56 ms instead of 32.33.

## Files

| File | What it is | sha256 (first 12) |
|---|---|---|
| `model_symbex.h` | the network for lib_symbex, ready to compile | `1d408a666057` |
| `model_binarized.pth` | the trained weights (PyTorch) | `54f19bff1586` |
| `model_init.pth` | the starting weights the training began from | `1e297baaed3f` |
| `train.json` | what the training chose and its accuracy | `16282c190b9e` |
| `export.json` | what the export measured (accuracy, exactness per engine, sizes) | — (rewritten by step 1) |
| `config.toml` | the CDT configuration, paths relative to `tools/` | — |
| `resnet8_int8.tflite` | the official MLPerf Tiny INT8 model ([source](https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite)) | `3c002613d1b2` |

## Use it

To build it into a project, copy `model_symbex.h` next to your code and follow the
[library guide](../../../../docs/library.md).

To measure it on a board, from `tools/` (environment `env_bench`, CIFAR-10 in `benchmarks/data/cifar10`):

```
# 1. rebuild the 2000 validation images and the integer simulation's answers (no training; same bytes as the run)
python cdt.py ../benchmarks/conv2d/cnn_cifar_resnet8/reference/config.toml --from export --to export
# 2. on a board (check the port first)
python chip_test.py ../benchmarks/conv2d/cnn_cifar_resnet8/reference --board s3 --port /dev/ttyACM0 --usb --popcount swar
python chip_test.py ../benchmarks/conv2d/cnn_cifar_resnet8/reference --board s3 --port /dev/ttyACM0 --usb --popcount swar --engine vertical_prepared
python chip_test.py ../benchmarks/conv2d/cnn_cifar_resnet8/reference --board esp32 --port /dev/ttyUSB0 --popcount lut16
python chip_test.py ../benchmarks/conv2d/cnn_cifar_resnet8/reference --board pico2_arm --port /dev/ttyACM1 --popcount lut16
```

The INT8 model on a board: `python resnet8_tflite.py --board s3 --port /dev/ttyACM0` from the benchmark's folder.
