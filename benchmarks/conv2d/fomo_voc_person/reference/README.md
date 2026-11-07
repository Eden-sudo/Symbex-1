# FOMO reference model (person detection, Pascal VOC)

One ready model, so that anyone can build it into a project or measure it on a board **without training**. It is
seed 42 of the detector resized for binary (32-32-128, blocks 1-2-2), fully binary, next to the INT8 model of the same
architecture ([README](../README.md)).

## The model

| | SYMBEX (`model_symbex.h`) | INT8 (`fomo_int8_c32-32-128.tflite`) |
|---|---|---|
| Network | FOMO 32-32-128 / 1-2-2, binary end to end (8-level thermometer input) | the same architecture, INT8 |
| Training | 100 epochs, person weight 40, seed 42 | the floating-point recipe: 60 epochs, person weight 20 |
| Validation F1 | 31.78 % | — |
| Test F1 (VOC 2007 test) | 31.8 % (precision 35.6, recall 28.8) | 30.34 % |
| Weights + thresholds | 41.2 KB | 236.1 KB |
| Exact | the compiled C equals the integer simulation on 100 % of the images, every board | TFLite Micro equals the PC on 88 % |

The three seeds together (the result, not this one model): F1 31.2 ± 0.7 % SYMBEX vs 30.1 ± 0.2 % INT8.

> [!NOTE]
> This model was trained on the earlier split (its threshold calibrated on the last 1,000 training images, Oct 23).
> It will be replaced when FOMO is retrained on the current validation split.

## On the boards (seed 42, weights in flash, median of 1000 images)

| Board | Counting | SYMBEX ms | RAM KB | INT8 TFLite Micro ms |
|---|---|---|---|---|
| ESP32-S3 | swar | 126.33 | 79.0 | 256.89 |
| ESP32 | lut16 | 226.35 | 142.8 | does not fit |
| Pico 2 Cortex-M33 | lut16 | 416.40 | 143.2 | — |
| Pico 2 Hazard3 | lut16 | 338.35 | 79.3 | — |

RAM is what the board reports for SYMBEX (50.8 KB of buffers, input, stack, the 64 KB lut16 table where used). The INT8
arena is 222.6 KB, too large for the classic ESP32. Every SYMBEX row was 100 % equal to the simulation.

## Files

| File | What it is | sha256 (first 12) |
|---|---|---|
| `model_symbex.h` | the network for lib_symbex, ready to compile | `6d7f59e3e262` |
| `model.pth` | the trained weights (PyTorch) | `eb208fc4b5d3` |
| `train.json` | the training settings and the detection threshold | `949b25fb1dd6` |
| `export.json` | what the export measured (sizes, exactness) | — (rewritten by step 1) |
| `fomo_int8_c32-32-128.tflite` | the INT8 model, same architecture | `5541bc85a22a` |

## Use it

To build it into a project, copy `model_symbex.h` next to your code and follow the
[library guide](../../../../docs/library.md).

To measure it on a board (environment `env_bench`, Pascal VOC prepared once with `python fomo_data.py`):

```
cd benchmarks/conv2d/fomo_voc_person
# 1. rebuild the 1000 test images and the integer simulation's answers (no training)
python fomo_export.py reference
# 2. on a board (check the port first), from tools/
cd ../../../tools
python chip_test.py ../benchmarks/conv2d/fomo_voc_person/reference --board s3 --port /dev/ttyACM0 --usb --popcount swar
python chip_test.py ../benchmarks/conv2d/fomo_voc_person/reference --board esp32 --port /dev/ttyUSB0 --popcount lut16
python chip_test.py ../benchmarks/conv2d/fomo_voc_person/reference --board pico2_riscv --port /dev/ttyACM1 --popcount lut16
```

The INT8 model on a board, this exact file (no training): `python fomo_tflite.py --model
reference/fomo_int8_c32-32-128.tflite --board s3 --port /dev/ttyACM0` from the benchmark's folder.
