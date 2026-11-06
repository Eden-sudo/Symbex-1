# cnn_cifar_resnet8 — a fully binary ResNet-8 against the official MLPerf Tiny INT8 model

The reference is the **official INT8 ResNet-8 of MLPerf Tiny** (image classification, CIFAR-10), measured with the official
TensorFlow Lite Micro on the same boards. The SYMBEX network has **the same architecture** — a 3×3 stem, three stages of two
3×3 convolutions with 16-32-64 channels, a linear classifier — and only what binarization imposes changes:

| | MLPerf Tiny ResNet-8 | SYMBEX binary ResNet-8 |
|---|---|---|
| Input | 8-bit pixels | thermometer: 8 bits per color (FracBNN); optionally an int8 first layer on the pixels (below) |
| Weights and activations | 8 bits | **1 bit** |
| Shortcuts | sum | OR of bits |
| Arithmetic on the chip | 8-bit multiply-accumulate | **counting bits: no multiplication at all** |

Every SYMBEX layer is folded into one integer threshold per channel, and the chip gives **exactly** the same class as the
integer simulation on every image.

## Results (3 seeds, trained from scratch, labels only)

| | **SYMBEX binary** | MLPerf Tiny INT8 | |
|---|---|---|---|
| **ESP32 (240 MHz)** | **59.7 ms** (lut16, weights in RAM) · **63.3 ms** (lut16, weights in flash) · **71.0 ms** (no table) | 717.8 ms | **12.0× / 11.3× / 10.1× faster** |
| **ESP32-S3 (240 MHz)** | **39.2 ms** | 52.0 ms | **1.33× faster** |
| Raspberry Pi Pico 2 (150 MHz), Cortex-M33 | **74.5 ms** (lut16, weights in flash) | 453.0 ms (CMSIS-NN, flash) · 211.4 ms (CMSIS-NN, model in RAM) | **6.1× / 2.8× faster** |
| Raspberry Pi Pico 2 (150 MHz), Hazard3 RISC-V | **56.1 ms** (native popcount, weights in flash) | 5702 ms (reference kernels, flash; 100 images) | **102× faster** |
| RAM | **15 KB** (79 KB with the lut16 table; 103 KB with the weights in RAM too) | 53 KB (ESP32) / 80.5 KB (S3) of arena | **3.6× / 5.4× less** (ESP32 no table / S3) |
| Flash (weights) | **24.3 KB** | 96.2 KB | **4× less** |
| Same result as the reference on the PC | **100%** (by design) | 98.1% | |
| CIFAR-10 accuracy (10,000 test images) | 71.3 ± 0.8 % | 87.0 % | |

The three seeds give the same latency to 0.01 ms (42, 43, 44: 70.60 %, 72.18 %, 71.06 %). The chip gives the same class as
the integer simulation on 100 % of the images on every board. On the Pico 2 the INT8 side is TensorFlow Lite Micro built
from source at a pinned commit: with Arm's CMSIS-NN kernels on the M33 (DSP on) and with the reference kernels on
Hazard3, the only ones TFLite Micro has for it (so that row shows the gap without vendor kernels, not a tuned INT8).
SYMBEX is 6.1× faster with both in flash, 2.8× against INT8 with the whole model in RAM. The chip gives the PC
interpreter's class on 98.1 % of the images, as on the ESP32 boards.

## Why: what each chip does best

`tools/dev/count_bench.py` measures, on each chip, the raw arithmetic of both families on the same layer pattern: how many
**binary operations** per cycle against how many **8-bit multiply-accumulates** per cycle.

| Chip | Best binary method | Binary operations per int8 multiply-accumulate |
|---|---|---|
| **ESP32** (no SIMD) | 64 KB table, AND form | **11-17×** |
| **ESP32-S3** (8-bit SIMD, no popcount) | PIE `pie_swar`, XOR form | **~1.05-2.1×** |
| x86, i9-13950HX (POPCNT + AVX2, against AVX-VNNI) | `POPCNT` 64 / AVX2, XOR form | 1.5-3.8× |

On chips without SIMD — most low-cost microcontrollers — binary arithmetic wins by an order of magnitude. On the S3, whose
SIMD unit multiplies 16 bytes per instruction, SYMBEX counts bits **with that same multiplier** (`pie_swar`: SWAR inside
every byte, then one multiply-accumulate adds the 16 counts), and still comes out ahead.

## Where the time goes (ESP32-S3, 39.2 ms)

| Step | ms | % |
|---|---|---|
| thermometer + stem 24→16, 32×32 | 7.84 | 20.0 |
| block 0: conv 16→16, 32×32 | 7.73 | 19.7 |
| block 1: conv 16→16, 32×32 | 7.73 | 19.7 |
| block 2: conv 16→32, stride 2 + OR pool + 1×1 | 5.40 | 13.8 |
| block 3: conv 32→32, 16×16 | 4.28 | 10.9 |
| block 4: conv 32→64, stride 2 + OR pool + 1×1 | 3.01 | 7.7 |
| block 5: conv 64→64, 8×8 | 3.13 | 8.0 |
| classifier | 0.06 | 0.1 |

Inside the library (`tools/dev/fine_profile.py`, CPU cycle counters per function): counting bits is ~61 % of the time and runs
at exactly the speed the benchmarks measure; thresholds and output bits ~17 %; building the windows ~13 %; the rest is the
thermometer, OR pooling, OR shortcuts and the classifier.

## How it got here

The same model, the same predictions; only the library changed, each step checked by the profile and by the tests:

| | ESP32-S3 | ESP32 (lut16) |
|---|---|---|
| First version | 67.1 ms | 100.5 ms * |
| `pie_swar`: one multiply per 128 bits instead of eight; XOR form | 48.3 ms | — |
| Windows, thresholds and counts without padding | 45.9 ms * | 77.9 ms * |
| Packed windows for 16 and 24 channels | 40.7 ms | 64.8 ms |
| Current library (Oct 28; the gain is in the stem: 9.3 → 7.8 ms) | **39.2 ms** | **63.3 ms** |

\* measured with the fine profile (CPU cycle counters inside the library: about 1-2 ms more than a normal run).

## Sizing the network for binary

The ResNet-8 shape was designed for INT8. `resnet8_sizing.py` trains the same structure with other numbers of blocks and
channels per stage (150 epochs each, one seed) and predicts the latency of every variant from the chip profiles of
`tools/dev/count_bench.py`, calibrated with the measured ResNet-8.

**Blocks** (16-32-64 channels): the second block of stage 1, the most expensive one, adds nothing.

| Blocks per stage | Accuracy | S3 ms (predicted) |
|---|---|---|
| 2-2-2 (ResNet-8) | 70.60 % | 40.7 |
| **1-2-2** | **70.59 %** | **34.4** (1.18× faster) |
| 1-1-2 | 69.03 % | 29.1 |
| 2-2-1 | 69.29 % | 36.9 |

**Channels** (1-1-2 blocks, 28 variants): channels in the last stage (8×8, cheap) buy accuracy; channels in the first
stage (32×32, expensive) buy little. The interesting variants, **measured on the chip with 3 seeds**:

| Channels / blocks | Accuracy | ESP32-S3 | ESP32 (weights in RAM) | RAM (S3) | Flash |
|---|---|---|---|---|---|
| 16-32-64 / 2-2-2 (ResNet-8) | 71.28 ± 0.81 % | 39.2 ms | 59.7 ms | 14.8 KB | 24.3 KB |
| 16-32-64 / 1-1-2 | 68.80 ± 0.22 % | **28.7 ms** | | 20.2 KB | 21.3 KB |
| **16-32-128 / 1-1-2** | **73.10 ± 0.23 %** | **39.6 ms** | 60.9 ms | 15.4 KB | 46.4 KB |
| **16-64-128 / 1-1-2** | **74.45 ± 0.23 %** | 46.6 ms | | 16.4 KB | 52.8 KB |

**A binary network shaped for binary is more accurate at the same speed:** 16-32-128 / 1-1-2 gains 1.8 points over the
ResNet-8 shape in the same time, with less RAM; 16-64-128 / 1-1-2 gains 3.2 points and is still faster than INT8 on the S3.

- **The predictions hold:** 29.1 / 41.4 / 49.9 ms predicted, 28.7 / 39.6 / 46.6 ms measured on the S3.
- **Choosing on clean seeds:** the variants were ranked on one seed; the two seeds added afterwards were never used to
  choose: 72.97 % and 74.58 % for them, against 73.10 % and 74.45 % over all three. Ranking on a validation split carved
  out of the training data is the cleaner method, and the next version of the study uses it.
- **The best shape depends on the chip.** On the ESP32, the 128-channel last stage holds 18 KB of weights, more than the
  flash cache keeps: in flash its last block takes 43 ms, in RAM 21 ms (the whole network: 88.8 against 60.9 ms). With the
  weights in RAM, 16-32-128 runs at the speed of the ResNet-8 shape with 1.8 points more.

## An int8 first layer (optional)

The thermometer turns every pixel into 8 bits before a binary stem. lib_symbex can instead run the first layer on the
8-bit pixels themselves: int8 weights, a multiply-accumulate per pixel and one integer threshold per channel
(`symbex_bool_stem_i8`), and every layer after it stays binary. It is an option of the `boolean` method; the
thermometer stays the default. Same network otherwise, same recipe (150 epochs, labels only, seeds 42, 43, 44, the
validation split carved out of the training data):

| First layer | Validation | Test (10,000 images) | Flash |
|---|---|---|---|
| thermometer + binary stem | 71.17 ± 0.32 % | 71.34 ± 0.89 % | 24.3 KB |
| **int8 stem** | **75.63 ± 0.50 %** | **75.41 ± 0.02 %** | **24.0 KB** |

**4.1 points more accurate**, with slightly less flash and RAM (14.3 against 14.8 KB on the S3: its output map has the
size of the thermometer's). Measured on every board, 3 seeds, 1,000 images, the chip equal to the integer simulation on 100 % of them
(thermometer: seed 42, measured the same day; latency does not depend on the seed, 0.01 ms apart):

| Chip | Thermometer | int8 stem | int8 stem vs thermometer | int8 stem vs MLPerf INT8 |
|---|---|---|---|---|
| **ESP32-S3** (swar, flash or RAM) | 39.19 ms | **35.54 ms** | **1.10× faster** | **1.46× faster** (52.0 ms) |
| ESP32-S3 (lut16, flash or RAM) | 40.70 ms | 36.75 ms | 1.11× faster | |
| Pico 2 Cortex-M33 (lut16, flash / RAM) | 74.58 / 69.62 ms | 72.21 / 68.54 ms | 1.03× / 1.02× faster | 6.3× / 3.1× faster |
| ESP32 (lut16, flash / RAM) | 63.25 / 59.74 ms | 68.07 / 63.52 ms | 0.93× / 0.94× (slower) | 10.5× / 11.3× faster |
| Pico 2 Hazard3 (native popcount, flash / RAM) | 56.15 / 52.55 ms | 64.34 / 59.45 ms | 0.87× / 0.88× (slower) | 89× faster |

The stem alone explains it: on the S3 the int8 stem takes 4.19 ms against 7.84 ms for the thermometer and its binary
stem (8-bit SIMD multiplies 16 bytes per instruction); the M33 has 2-way multiply-accumulate (SMLAD), 15.1 against
17.7 ms; the ESP32 and Hazard3 have no SIMD, 18.6 against 14.0 ms and 20.0 against 12.7 ms. `../../experiments/chip_kernels`
(`int8_first_bench.py`) measures the stem alone with random weights and gives the same picture.

**When to use it:** always for accuracy; on the S3 and the M33 it is also faster. On the ESP32 and Hazard3 it costs
5-8 ms (7-15 %) for the 4 points (before Nov 6: the windowless stem below makes it faster). Shortcuts stay OR: added
shortcuts, tried in the same study, were less accurate (69.58 % alone, 72.34 % with the int8 stem, test) and are not
in lib_symbex. The vertical engine (`docs/vertical.md`) runs the int8 stem too (`../../experiments/training_gap`).

## The most accurate recipe: int8 stem + Bop + pooled int8 classifier

`../../experiments/training_gap` found that two more changes add accuracy on top of the int8 stem: Bop for the 1-bit
weights (`binary_optimizer = "bop"`) and a small int8 classifier on the ones counted per 2×2 window of the last map
(`pool2`, lib_symbex's `symbex_bool_pool_head`). The recipe now runs through the CDT
(`--variant int8stem_pool2`, class `BoolResNet8Int8StemPool2` in `model.py`), same 150 epochs, labels only, seeds
42, 43, 44:

| Trained by | Validation | Test (10,000 images) | Flash |
|---|---|---|---|
| the study (`train_gap_study.py`) | 77.48 ± 0.39 % | 77.26 ± 0.09 % | 29.6 KB |
| **the CDT** | **77.30 ± 0.45 %** | **76.68 ± 0.34 %** | **29.0 KB** |

The CDT's network is the same on validation; on test it is 0.6 points lower (its test is printed by every CDT run and
used for no choice). Every seed exact: the compiled C equal to the integer simulation on 100 % of the images, the
standard engine and the three vertical modes.

**A faster int8 stem on every board but the S3 (Nov 6).** On the Pico 2 the int8 stem was ~20 % of the network
(M33 14.6 ms, Hazard3 19.1 ms). lib_symbex now runs 3×3 stems without building a window per pixel: the image kept as 3
padded rows, the weights of one channel in registers while 16 output pixels stream past (SMLAD on the M33, plain C
elsewhere; `stem3_rows` in `symbex_bool.c`, rows up to 64 wide, the window path past that). The stem alone: M33
14.6 → 9.2 ms, Hazard3 19.1 → 12.3 ms (`../../experiments/chip_kernels`, `stem_i8_pico_bench.py`). Whole network,
3 seeds × 1,000 images, every image equal to the simulation, lut16, ms:

| Chip | weights | before (Nov 5) | **now** | |
|---|---|---|---|---|
| Pico 2 Cortex-M33 | flash | 73.57 | **67.77** | 1.09× faster |
| Pico 2 Cortex-M33 | RAM | 69.14 | **63.43** | 1.09× faster |
| Pico 2 Hazard3 | flash | 65.52 | **57.94** | 1.13× faster |
| Pico 2 Hazard3 | RAM | 60.16 | **53.46** | 1.13× faster |
| ESP32 | flash | 69.31 | **62.52** | 1.11× faster |
| ESP32 | RAM | 63.98 | **58.23** | 1.10× faster |

"Before" is the study's network on the library of Nov 5 (`../../experiments/training_gap/results/train_gap/pool_head_chip.json`);
"now" is the CDT's network (latency does not depend on the weights; the three seeds agree to 0.01 ms). With it, the
most accurate network is also faster than the thermometer network on the ESP32 (62.52 vs 63.25 ms, flash) and the M33
(67.77 vs 74.58); on Hazard3 it is 1.8 ms slower (57.94 vs 56.15) for 5.3 more points (test). The S3 keeps its PIE kernel
(32.33 ms standard, 9.73 ms vertical).

## Reproduce

From this folder (environment `env_bench`, with PyTorch on CUDA and TensorFlow):

```
python resnet8_symbex.py --width 1 --seeds 3                                   # train + export (PC)
python resnet8_symbex.py --width 1 --seeds 3 --board s3 --port /dev/ttyUSB0
python resnet8_symbex.py --width 1 --seeds 3 --board esp32 --port /dev/ttyUSB0 --popcount lut16
python resnet8_symbex.py --width 1 --seeds 3 --board esp32 --port /dev/ttyUSB0 --popcount lut16 --weights ram
python resnet8_symbex.py --width 1 --seeds 3 --board esp32 --port /dev/ttyUSB0
python resnet8_symbex.py --spec 16-32-128/1-1-2 --seeds 3 --board s3 --port /dev/ttyUSB0     # a sized variant
python resnet8_symbex.py --width 1 --seeds 3 --board pico2_riscv --port /dev/ttyACM0 --popcount lut16   # Pico 2 (Hazard3)
python resnet8_symbex.py --width 1 --seeds 3 --board pico2_arm --port /dev/ttyACM0 --popcount lut16     # Pico 2 (M33)
python ../../../tools/chip_test.py runs/w1_s42 --board s3 --port /dev/ttyACM0 --usb   # a trained seed, the S3's native USB
python resnet8_symbex.py --width 1 --seeds 3 --board s3 --summary              # summary of the chip_test.py runs
python resnet8_symbex.py --variant int8stem --seeds 3                          # the int8 stem: train + export (PC)
python resnet8_symbex.py --variant int8stem --seeds 3 --board s3 --port /dev/ttyACM0   # ... on a board
python resnet8_symbex.py --variant int8stem_pool2 --seeds 3                    # the most accurate recipe (Bop + pool2)
python resnet8_symbex.py --variant int8stem_pool2 --seeds 3 --board pico2_arm --port /dev/ttyACM1 --popcount lut16
python resnet8_symbex.py --variant base --seeds 3                              # the stem / shortcut study:
python resnet8_symbex.py --variant add --seeds 3                               #   base | int8stem | add | int8stem_add
python resnet8_sizing.py --phase depth --epochs 150                            # the sizing study
python resnet8_sizing.py --phase width --blocks 1-1-2 --epochs 150
python resnet8_tflite.py                                                       # INT8: accuracy on the PC
python resnet8_tflite.py --board s3 --port /dev/ttyUSB0                         # INT8 on the chip
python resnet8_tflite.py --board esp32 --port /dev/ttyUSB0
python resnet8_tflite.py --board pico2_arm --port /dev/ttyACM1                  # Pico 2 M33: TFLite Micro + CMSIS-NN
python resnet8_tflite.py --board pico2_arm --port /dev/ttyACM1 --model-ram
python resnet8_tflite.py --board pico2_riscv --port /dev/ttyACM1 --images 100   # Hazard3: reference kernels
```

Profiles: `python ../../../tools/dev/fine_profile.py runs/w1_s44 --board s3` and `python ../../../tools/dev/count_bench.py --board s3`.

| File | What it does |
|---|---|
| `model.py` | the binary ResNet-8 with any channels and blocks per stage, the thermometer, CIFAR-10 |
| `symbex.toml` | the CDT configuration (method `boolean`, 150 epochs, labels only) |
| `resnet8_symbex.py` | the CDT per seed and per board; summary in `results/symbex_seeds_<board>_<popcount>_w1.json` (`..._w1_<variant>.json` with `--variant`) (`--summary`: from runs measured with `tools/chip_test.py`; older summaries in `results/earlier/`) |
| `resnet8_tflite.py` | the official MLPerf Tiny INT8 model, on the PC and with TFLite Micro on the boards |
| `resnet8_train.py` | evaluates a trained model on its own (`--check`) |
| `resnet8_sizing.py` | the sizing study: variants, predicted latency, Pareto front (`results/sizing.json`; every variant's quick training in `results/sizing/`) |

## Future work

- **Accuracy:** distillation from the official floating-point MLPerf Tiny ResNet-8 as the teacher; twice the channels
  (`--width 2`, 79-80 % in earlier experiments); better thermometer levels; longer training.
- **Sizing:** rank the variants on a validation split; place each layer's weights in RAM or flash by its size,
  automatically (the flash cache effect above).
- **Speed:** counting is now the main cost, at the speed of the best method of each chip; vector popcount (ARM NEON),
  other chips (RISC-V without multiplier), and ping-pong buffering of the weights for models too large for RAM.
- **Applications:** person detection is now its own benchmark (`../fomo_voc_person`).

## References

MLPerf Tiny (Banbury et al., NeurIPS 2021) · TensorFlow Lite Micro (David et al., MLSys 2021) · Binarized Neural Networks
(Hubara et al., NeurIPS 2016) · XNOR-Net (Rastegari et al., ECCV 2016) · FINN (Umuroglu et al., FPGA 2017: BatchNorm and
sign folded into integer thresholds) · FracBNN (Zhang et al., FPGA 2021: thermometer input encoding).
