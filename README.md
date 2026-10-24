# SYMBEX-1 -- Binary neural networks for microcontrollers

SYMBEX-1 compiles neural networks into **fully binary networks that run on small microcontrollers**, and measures
them on real chips against the official INT8 toolchain (TensorFlow Lite Micro).

It has three parts:

- **`lib_symbex`** -- a portable C inference engine for binary networks: bit counting instead of multiplication,
  integer thresholds, no floating point. One hardware layer per chip family (generic, ESP32, ESP32-S3, AVR).
- **The CDT** (`tools/cdt.py`) -- takes a model, your data and your hardware limits, trains the binary network,
  folds it into integers, writes its C header and checks it bit by bit, on the PC and on the chip.
- **`benchmarks/`** -- reproducible comparisons against INT8 TensorFlow Lite Micro on the same chips.

## Results (ESP32-S3 and ESP32, 240 MHz, one core)

### Image classification: a fully binary ResNet-8 ([`benchmarks/cnn_cifar_resnet8`](benchmarks/cnn_cifar_resnet8/))

CIFAR-10, against the official INT8 ResNet-8 of MLPerf Tiny. Every layer of the SYMBEX network is binary, the first
one included (the image enters as a thermometer of bits); 3 training seeds, labels only.

| | **SYMBEX, binary** | MLPerf Tiny INT8 |
|---|---|---|
| Accuracy | 71.28 ± 0.81 % | 87.0 % |
| ESP32-S3 | **40.7 ms (1.28× faster)** | 52.0 ms |
| ESP32 | **59.7 ms (12.0× faster)**, weights in RAM; 64.8 ms in flash | 717.8 ms |
| Flash | **24.3 KB** | 96.2 KB |
| RAM | **20 KB** | 53-80 KB |
| Same result as its PC reference | **100 %** | 98.1 % |

**Sized for binary.** The best shape of a binary network is not the INT8 one: a sizing study (blocks and channels per
stage, latency predicted from measured chip profiles) found that the first stages can be thinner and the last one
wider. Measured on the chip, 3 seeds (on the ESP32 the gain is accuracy at the same speed; the best shape depends on
the chip):

| Network | Accuracy | ESP32-S3 | ESP32 | RAM |
|---|---|---|---|---|
| ResNet-8 shape (16-32-64, 2-2-2 blocks) | 71.28 ± 0.81 % | 40.7 ms | 59.7 ms (weights in RAM) | 20 KB |
| **16-32-128, 1-1-2 blocks** | **73.10 ± 0.23 %** | **39.6 ms** | 60.9 ms (weights in RAM) | 15 KB |
| **16-64-128, 1-1-2 blocks** | **74.45 ± 0.23 %** | 46.6 ms | | 16 KB |

### Person detection: FOMO, binary against the same detector in INT8 ([`benchmarks/fomo_voc_person`](benchmarks/fomo_voc_person/))

Pascal VOC persons, 96 × 96 images, a 12 × 12 grid of person centers. The INT8 reference is **the same architecture**
trained with the same recipe.

| | **SYMBEX, binary** | TFLite INT8 |
|---|---|---|
| F1 (3 seeds) | **31.2 ± 0.7 %** | 30.1 ± 0.2 % |
| ESP32-S3 | **126.3 ms (2.0× faster)** | 256.9 ms |
| ESP32 | **runs** | does not fit (its arena overflows the RAM) |
| Flash | **41 KB** | 236 KB |
| RAM | **51 KB** | 223 KB |
| Same result as its PC reference | **100 %** | 88 % |

Detection binarizes well: every cell answers a yes/no question, and the binary detector is as accurate as the INT8 one.

### Dense networks ([`benchmarks/mlp_fashion`](benchmarks/mlp_fashion/), [`benchmarks/mlp_digits`](benchmarks/mlp_digits/))

Fashion-MNIST, every layer at 1 bit, twice as wide: 86.80 ± 0.36 % (INT8: 87.20 %) with **69 KB of weights against
242 KB**, and **2.5× faster than INT8 on the ESP32** (16.70 ms against 41.97 ms). These figures are being re-measured
with the current engine and the corrected validation split (see Methodology).

### Why it is faster: counting bits against multiplying

`tools/count_bench.py` measures the raw arithmetic of each chip on the same convolution pattern: binary (bit counting)
against int8 (multiply-accumulate).

| Chip | Binary against int8, raw arithmetic |
|---|---|
| ESP32 (no SIMD) | **11-17× faster** |
| ESP32-S3 (8-bit SIMD, no popcount instruction) | 1.05-2.1× faster (the SIMD multiplier counts the bits: `pie_swar`) |
| x86 (AVX2 / VNNI) | 1.5-3.8× |

## How it works

1. **Every layer is binary.** Weights and activations are one bit. A convolution is a bit count (AND or XOR followed
   by popcount); a residual shortcut is an OR of bits; pooling is an OR.
2. **The input becomes bits too.** Each pixel enters as a thermometer (8 levels per color), so the first layer is binary
   like the rest.
3. **Everything else folds into integers.** Scale, BatchNorm and the activation collapse into one integer threshold per
   output: `bit = (count >= T)`. No floating point at inference.
4. **The counting method is chosen per chip.** A native popcount instruction where there is one, a 64 KB table or SWAR
   on the ESP32, the SIMD multiplier on the ESP32-S3. `count_bench.py` measures them; the library picks the fastest.
5. **Bit-exact, always.** The compiled C, on the PC and on the chip, must reproduce the integer simulation of the model
   100 %: every exported model is checked, every time.

## Methodology

- **Same architecture** whenever SYMBEX is compared with INT8 (the official MLPerf model, or the same network trained
  with the same recipe); a resized binary network is compared with INT8 resized the same way.
- **Independent baseline:** INT8 through the official TensorFlow Lite converter and TensorFlow Lite Micro, in a
  separate environment that imports nothing from SYMBEX.
- **Measured on the chip:** medians over 1000 inferences (`esp_timer`), one core, 240 MHz; RAM measured on the chip.
- **Several seeds** for SYMBEX: accuracy as mean ± standard deviation.
- **Validation split:** model and variant selection must use a validation split carved out of the training data, with
  the test set evaluated once. A review found places where the test set was used for selection (some CDT runs, the
  choice of variants in the sizing studies); the fix is being applied and the affected figures re-measured.

## Current state and next steps

- **1D convolutions** for sensor data (human activity recognition), compared with the closest prior work: the
  binary library of Daghero et al. (2021), in their own simulated RISC-V environment (GVSoC) and on real silicon.
- **More chips:** Arm Cortex-M33 and RISC-V (RP2350).
- **Distillation** from the official floating-point MLPerf ResNet-8.
- **Energy** has not been measured yet; latency and memory are.

## Repository layout

```
Symbex1_PIO/               PlatformIO project
  lib/lib_symbex/          the C engine (include/, src/, hal/), and tests/run_tests.py (bit-exact tests, 3 HALs)
  src/main.cpp             minimal example: a demo network checked against its PC simulation on your board
tools/
  cdt.py                   the CDT: configuration (TOML) -> train -> export -> validate -> chip
  chip_test.py             runs an exported model on a board: exactness, accuracy, latency per layer, RAM
  fine_profile.py          time per library function on the chip
  count_bench.py           raw arithmetic of each chip (binary against int8)
  compiler_core/           the library behind the CDT
  configs/                 example.toml: every option, commented
benchmarks/                reproducible comparisons against TensorFlow Lite Micro, one folder per task
docs/                      documentation (English and Spanish)
```

## Quick start

```
# the library on your board (PlatformIO): the example must print "8/8 equal to the PC simulation"
cd Symbex1_PIO && pio run -t upload && pio device monitor -b 115200

# the library tests (gcc)
python Symbex1_PIO/lib/lib_symbex/tests/run_tests.py

# a model through the CDT (from tools/), then on a board
python cdt.py my_model.toml
python chip_test.py <output folder> --board esp32 --port /dev/ttyUSB0
```

Requirements: Python with PyTorch; gcc; PlatformIO and pyserial for the chip; TensorFlow for the INT8 baselines.

## Previous versions

Version 2 reported results on the scikit-learn Digits dataset with the `gated` and `bitslice` methods. During the
version 3 review, errors were found in those methods; those results are not reported until the methods are rebuilt and
re-measured. The first convolutional engine of version 3 (with a learned int8 first layer) has been replaced by the fully
binary engine above.

## Citation

A preprint describing SYMBEX-1 is in preparation; the reference will be added here.

## Contributors

Carlos Duarte (Edén) -- Panama -- GitHub: [Eden-sudo](https://github.com/Eden-sudo)
Obed -- GitHub: [Obed0101](https://github.com/Obed0101) -- test suite, low-bit baselines, validation split review

Spanish version: [README.es.md](README.es.md) -- License: see [LICENSE](LICENSE).
