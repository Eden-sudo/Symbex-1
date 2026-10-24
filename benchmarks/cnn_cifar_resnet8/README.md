# cnn_cifar_resnet8 — a fully binary ResNet-8 against the official MLPerf Tiny INT8 model

The reference is the **official INT8 ResNet-8 of MLPerf Tiny** (image classification, CIFAR-10), measured with the official
TensorFlow Lite Micro on the same boards. The SYMBEX network has **the same architecture** — a 3×3 stem, three stages of two
3×3 convolutions with 16-32-64 channels, a linear classifier — and only what binarization imposes changes:

| | MLPerf Tiny ResNet-8 | SYMBEX binary ResNet-8 |
|---|---|---|
| Input | 8-bit pixels | thermometer: 8 bits per color (FracBNN) |
| Weights and activations | 8 bits | **1 bit** |
| Shortcuts | sum | OR of bits |
| Arithmetic on the chip | 8-bit multiply-accumulate | **counting bits: no multiplication at all** |

Every SYMBEX layer is folded into one integer threshold per channel, and the chip gives **exactly** the same class as the
integer simulation on every image.

## Results (3 seeds, trained from scratch, labels only)

| | **SYMBEX binary** | MLPerf Tiny INT8 | |
|---|---|---|---|
| **ESP32 (240 MHz)** | **59.7 ms** (lut16, weights in RAM) · **64.8 ms** (lut16, weights in flash) · **72.6 ms** (no table) | 717.8 ms | **12.0× / 11.1× / 9.9× faster** |
| **ESP32-S3 (240 MHz)** | **40.7 ms** | 52.0 ms | **1.28× faster** |
| RAM | **20 KB** (84 KB with the lut16 table; 103 KB with the weights in RAM too) | 53 KB (ESP32) / 80.5 KB (S3) of arena | **2.7-4× less** on the S3 |
| Flash (weights) | **24.3 KB** | 96.2 KB | **4× less** |
| Same result as the reference on the PC | **100%** (by design) | 98.1% | |
| CIFAR-10 accuracy (10,000 test images) | 71.3 ± 0.8 % | 87.0 % | |

The three seeds give the same latency to 0.01 ms (42, 43, 44: 70.60 %, 72.18 %, 71.06 %).

## Why: what each chip does best

`tools/count_bench.py` measures, on each chip, the raw arithmetic of both families on the same layer pattern: how many
**binary operations** per cycle against how many **8-bit multiply-accumulates** per cycle.

| Chip | Best binary method | Binary operations per int8 multiply-accumulate |
|---|---|---|
| **ESP32** (no SIMD) | 64 KB table, AND form | **11-17×** |
| **ESP32-S3** (8-bit SIMD, no popcount) | PIE `pie_swar`, XOR form | **~1.05-2.1×** |
| x86, i9-13950HX (POPCNT + AVX2, against AVX-VNNI) | `POPCNT` 64 / AVX2, XOR form | 1.5-3.8× |

On chips without SIMD — most low-cost microcontrollers — binary arithmetic wins by an order of magnitude. On the S3, whose
SIMD unit multiplies 16 bytes per instruction, SYMBEX counts bits **with that same multiplier** (`pie_swar`: SWAR inside
every byte, then one multiply-accumulate adds the 16 counts), and still comes out ahead.

## Where the time goes (ESP32-S3, 40.7 ms)

| Step | ms | % |
|---|---|---|
| thermometer + stem 24→16, 32×32 | 9.34 | 23.0 |
| block 0: conv 16→16, 32×32 | 7.73 | 19.0 |
| block 1: conv 16→16, 32×32 | 7.73 | 19.0 |
| block 2: conv 16→32, stride 2 + OR pool + 1×1 | 5.40 | 13.3 |
| block 3: conv 32→32, 16×16 | 4.29 | 10.5 |
| block 4: conv 32→64, stride 2 + OR pool + 1×1 | 3.01 | 7.4 |
| block 5: conv 64→64, 8×8 | 3.13 | 7.7 |
| classifier | 0.06 | 0.1 |

Inside the library (`tools/fine_profile.py`, CPU cycle counters per function): counting bits is ~61 % of the time and runs
at exactly the speed the benchmarks measure; thresholds and output bits ~17 %; building the windows ~13 %; the rest is the
thermometer, OR pooling, OR shortcuts and the classifier.

## How it got here

The same model, the same predictions; only the library changed, each step checked by the profile and by the tests:

| | ESP32-S3 | ESP32 (lut16) |
|---|---|---|
| First version | 67.1 ms | 100.5 ms * |
| `pie_swar`: one multiply per 128 bits instead of eight; XOR form | 48.3 ms | — |
| Windows, thresholds and counts without padding | 45.9 ms * | 77.9 ms * |
| Packed windows for 16 and 24 channels | **40.7 ms** | **64.8 ms** |

\* measured with the fine profile (CPU cycle counters inside the library: about 1-2 ms more than a normal run).

## Sizing the network for binary

The ResNet-8 shape was designed for INT8. `resnet8_sizing.py` trains the same structure with other numbers of blocks and
channels per stage (150 epochs each, one seed) and predicts the latency of every variant from the chip profiles of
`tools/count_bench.py`, calibrated with the measured ResNet-8.

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
| 16-32-64 / 2-2-2 (ResNet-8) | 71.28 ± 0.81 % | 40.7 ms | 59.7 ms | 20.2 KB | 24.3 KB |
| 16-32-64 / 1-1-2 | 68.80 ± 0.22 % | **28.7 ms** | | 20.2 KB | 21.3 KB |
| **16-32-128 / 1-1-2** | **73.10 ± 0.23 %** | **39.6 ms** | 60.9 ms | **15.4 KB** | 46.4 KB |
| **16-64-128 / 1-1-2** | **74.45 ± 0.23 %** | 46.6 ms | | 16.4 KB | 52.8 KB |

**A binary network shaped for binary is more accurate at the same speed:** 16-32-128 / 1-1-2 gains 1.8 points over the
ResNet-8 shape in the same time, with less RAM; 16-64-128 / 1-1-2 gains 3.2 points and is still faster than INT8 on the S3.

- **The predictions hold:** 29.1 / 41.4 / 49.9 ms predicted, 28.7 / 39.6 / 46.6 ms measured on the S3.
- **Choosing on clean seeds:** the variants were ranked on one seed; the two seeds added afterwards were never used to
  choose: 72.97 % and 74.58 % for them, against 73.10 % and 74.45 % over all three. Ranking on a validation split carved
  out of the training data is the cleaner method, and the next version of the study uses it.
- **The best shape depends on the chip.** On the ESP32, the 128-channel last stage holds 18 KB of weights, more than the
  flash cache keeps: in flash its last block takes 43 ms, in RAM 21 ms (the whole network: 88.2 against 60.9 ms). With the
  weights in RAM, 16-32-128 runs at the speed of the ResNet-8 shape with 1.8 points more.

## Reproduce

From this folder (environment `env_bench`, with PyTorch on CUDA and TensorFlow):

```
python resnet8_symbex.py --width 1 --seeds 3                                   # train + export (PC)
python resnet8_symbex.py --width 1 --seeds 3 --board s3 --port /dev/ttyUSB0
python resnet8_symbex.py --width 1 --seeds 3 --board esp32 --port /dev/ttyUSB0 --popcount lut16
python resnet8_symbex.py --width 1 --seeds 3 --board esp32 --port /dev/ttyUSB0 --popcount lut16 --weights ram
python resnet8_symbex.py --width 1 --seeds 3 --board esp32 --port /dev/ttyUSB0
python resnet8_symbex.py --spec 16-32-128/1-1-2 --seeds 3 --board s3 --port /dev/ttyUSB0     # a sized variant
python resnet8_sizing.py --phase depth --epochs 150                            # the sizing study
python resnet8_sizing.py --phase width --blocks 1-1-2 --epochs 150
python resnet8_tflite.py                                                       # INT8: accuracy on the PC
python resnet8_tflite.py --board s3 --port /dev/ttyUSB0                         # INT8 on the chip
python resnet8_tflite.py --board esp32 --port /dev/ttyUSB0
```

Profiles: `python ../../tools/fine_profile.py runs/w1_s44 --board s3` and `python ../../tools/count_bench.py --board s3`.

| File | What it does |
|---|---|
| `model.py` | the binary ResNet-8 with any channels and blocks per stage, the thermometer, CIFAR-10 |
| `symbex.toml` | the CDT configuration (method `boolean`, 150 epochs, labels only) |
| `resnet8_symbex.py` | the CDT per seed and per board; summary in `results/symbex_seeds_<board>_<popcount>_w1.json` |
| `resnet8_tflite.py` | the official MLPerf Tiny INT8 model, on the PC and with TFLite Micro on the boards |
| `resnet8_train.py` | evaluates a trained model on its own (`--check`) |
| `resnet8_sizing.py` | the sizing study: variants, predicted latency, Pareto front (`results/sizing.json`) |

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
