<p align="center">
  <img src="docs/assets/symbex-1-banner.jpg" alt="SYMBEX-1, binary neural networks for microcontrollers" width="100%">
</p>

<p align="center"><b>Binary neural networks for microcontrollers.</b></p>

<p align="center">
  <a href="https://github.com/Eden-sudo/Symbex-1/releases"><img src="https://img.shields.io/github/v/release/Eden-sudo/Symbex-1?color=14b8a6&label=release" alt="release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Eden-sudo/Symbex-1?color=0d9488&label=license" alt="license"></a>
  <a href="https://symbex-1.pages.dev/"><img src="https://img.shields.io/badge/website-symbex--1.pages.dev-0f766e" alt="website"></a>
  <a href="benchmarks/"><img src="https://img.shields.io/badge/chips-ESP32%20%7C%20ESP32--S3%20%7C%20RP2350%20%7C%20AVR-115e59" alt="chips"></a>
  <a href="https://github.com/Eden-sudo/Symbex-1/pulls"><img src="https://img.shields.io/badge/PRs-welcome-f97316" alt="PRs"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/leer-en%20espa%C3%B1ol-334155" alt="leer"></a>
</p>

SYMBEX-1 compiles neural networks into **fully binary networks that run on small microcontrollers**, and measures
them on real chips against the official INT8 toolchain (TensorFlow Lite Micro).

It has three parts:

- **`lib_symbex`** -- a portable C inference engine for binary networks: bit counting instead of multiplication,
  integer thresholds, no floating point. One hardware layer per chip family (generic, ESP32, ESP32-S3, AVR,
  RP2350).
- **The CDT** (`tools/cdt.py`) -- takes a model, your data and your hardware limits, trains the binary network,
  folds it into integers, writes its C header and checks it bit by bit, on the PC and on the chip.
- **`benchmarks/`** -- reproducible comparisons against INT8 TensorFlow Lite Micro on the same chips.

## Results (measured on the chips, one core)

INT8 here is the deployment path almost everyone uses: TensorFlow Lite Micro with Espressif's optimized ESP-NN
kernels. It is not the fastest INT8 possible: on the ESP32-S3, Espressif's ESP-DL library runs ResNet-8 in 14.8 ms,
2.6× faster than SYMBEX's standard engine. SYMBEX's [vertical engine](docs/vertical.md) runs it in 10.4 ms, 1.43× faster
than ESP-DL with the weights in flash; with its optional compact mode, 7.70 ms with 151 KB of RAM, it is also 1.27×
faster than ESP-DL with its weights in RAM (147 KB there), and 6.44 ms with the prepared mode (238 KB), while on the classic ESP32 SYMBEX stays 8-11× faster than ESP-DL too.
[docs/results.md](docs/results.md#how-fast-could-int8-be) has the numbers and what is not measured yet.

### Image classification: a fully binary ResNet-8 ([`benchmarks/conv2d/cnn_cifar_resnet8`](benchmarks/conv2d/cnn_cifar_resnet8/))

CIFAR-10, against the official INT8 ResNet-8 of MLPerf Tiny. Every layer of the SYMBEX network is binary, the first
one included (the image enters as a thermometer of bits); 3 training seeds, labels only.

| | **SYMBEX, binary** | MLPerf Tiny INT8 |
|---|---|---|
| Accuracy | 71.28 ± 0.81 % | 87.0 % |
| ESP32-S3 | **39.2 ms (1.33× faster)** | 52.0 ms |
| ESP32 | **59.7 ms (12.0× faster)**, weights in RAM; 63.3 ms in flash | 717.8 ms |
| Raspberry Pi Pico 2, Cortex-M33 | **74.5 ms (6.1× faster)** | 453.0 ms (CMSIS-NN) |
| Raspberry Pi Pico 2, Hazard3 RISC-V | **56.1 ms** | 5702 ms (reference kernels, the only ones TFLite Micro has for it) |
| Flash | **24.3 KB** | 96.2 KB |
| RAM | **15 KB** | 53-80 KB |
| Same result as its PC reference | **100 %** | 98.1 % |

**The most accurate recipe.** With an int8 first layer on the pixels and a pooled int8 classifier (every other layer
binary, Bop for the binary weights; `--variant int8stem_pool2` in the CDT), the same ResNet-8 reaches **76.7 ± 0.3 %**
and runs faster: ESP32-S3 32.3 ms (1.61×), 6.0-9.7 ms with the vertical engine; ESP32 62.5 ms (11.5×); Pico 2 M33
67.8 ms (6.7×), Hazard3 57.9 ms; 29 KB of flash, exact on every board. Seed 42 is ready to use in
[`reference/`](benchmarks/conv2d/cnn_cifar_resnet8/reference/README.md).

**Sized for binary.** The best shape of a binary network is not the INT8 one: a sizing study (blocks and channels per
stage, latency predicted from measured chip profiles) found that the first stages can be thinner and the last one
wider. Measured on the chip, 3 seeds (on the ESP32 the gain is accuracy at the same speed; the best shape depends on
the chip):

| Network | Accuracy | ESP32-S3 | ESP32 | RAM |
|---|---|---|---|---|
| ResNet-8 shape (16-32-64, 2-2-2 blocks) | 71.28 ± 0.81 % | 39.2 ms | 59.7 ms (weights in RAM) | 15 KB |
| **16-32-128, 1-1-2 blocks** | **73.10 ± 0.23 %** | **39.6 ms** | 60.9 ms (weights in RAM) | 15 KB |
| **16-64-128, 1-1-2 blocks** | **74.45 ± 0.23 %** | 46.6 ms | | 16 KB |

### Person detection: FOMO, binary against the same detector in INT8 ([`benchmarks/conv2d/fomo_voc_person`](benchmarks/conv2d/fomo_voc_person/))

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

### Dense networks ([`benchmarks/mlp/mlp_fashion`](benchmarks/mlp/mlp_fashion/), [`benchmarks/mlp/mlp_digits`](benchmarks/mlp/mlp_digits/))

Fashion-MNIST, sized for binary (784-128-512-10: a narrow first layer, wide binary layers), against INT8 of the same
network; 3 seeds, labels only, weights in flash unless stated:

| | **SYMBEX, binary** | INT8 (same network) |
|---|---|---|
| Accuracy | 87.47 ± 0.09 % | 89.81 ± 0.11 % (QAT) |
| ESP32-S3 | **0.42 ms (28× faster)** | 11.7 ms (TFLite Micro) |
| ESP32-S3, weights in RAM | **0.41 ms (2.1× faster)** | 0.89 ms |
| ESP32 | **3.5 ms (8.8× faster)** | 31.1 ms |
| Raspberry Pi Pico 2, Cortex-M33 | **4.6 ms (6.9× faster)**; 1.5 ms with the weights in RAM (1.9×) | 31.4 ms (CMSIS-NN); 2.9 ms in RAM |
| Raspberry Pi Pico 2, Hazard3 | **4.4 ms (8.5× faster)**; 1.4 ms with the weights in RAM (6.8×) | 37.5 ms; 9.6 ms in RAM (reference kernels) |
| Flash | **27 KB** | 187 KB |
| RAM | **4.6 KB** | 6.7 KB |

INT8 is 2.3 points more accurate; SYMBEX wins speed and memory. On the smaller Digits MLP the two tie on the ESP32-S3
(0.109 against 0.110 ms), SYMBEX is 2-3.3× faster on the ESP32, and it is the only one that runs on an Arduino Uno.

### Activity recognition: binary 1D convolutions ([`benchmarks/conv1d/har_ultracompact`](benchmarks/conv1d/har_ultracompact/))

UniMiB-SHAR (accelerometer, 17 activities), the four networks of the closest prior work (Daghero et al. 2021,
ultracompactBNN), 3 seeds:

- **Against their binary library, in their own simulated RISC-V environment (GVSoC):** SYMBEX needs **1.07-1.37×
  fewer instructions** on all four networks, also with their own trained weights.
- **Against INT8 TFLite Micro, same architectures, weights in flash:** **5.3-10.4× faster** on the ESP32 and
  **1.47-1.90× faster** on the ESP32-S3, with 2-10× less RAM and 6.5-10× less flash. INT8 is more accurate on these
  small networks (about 9 points on the large ones).
- **Where fully binary stops paying (a working theory):** with an 8-bit input like the accelerometer, a first layer
  with int8 weights is faster and more accurate than a binary one on every board measured.

### Why it is faster: counting bits against multiplying

`tools/dev/count_bench.py` measures the raw arithmetic of each chip on the same convolution pattern: binary (bit counting)
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

The full standard, the references and how to repeat a benchmark: [docs/methodology.md](docs/methodology.md). In short:

- **Same architecture** whenever SYMBEX is compared with INT8 (the official MLPerf model, or the same network trained
  with the same recipe); a resized binary network is compared with INT8 resized the same way.
- **Independent baseline:** INT8 through the official TensorFlow Lite converter and TensorFlow Lite Micro, in a
  separate environment that imports nothing from SYMBEX.
- **Measured on the chip:** medians over 1000 inferences, one core, at the stock clock (240 MHz on the ESP32 chips,
  150 MHz on the Pico 2); RAM measured on the chip.
- **Several seeds** for SYMBEX: accuracy as mean ± standard deviation.
- **Validation split:** every choice (checkpoint, threshold, variant, shape) is made on a validation split carved out
  of the training data; the test set is evaluated once. A review found places where the test set had been used for
  selection; they were fixed and the affected figures re-measured (October 2026).

## Current state and next steps

Done in October 2026: 1D convolutions (against Daghero et al. 2021, in GVSoC and on real boards), the Raspberry
Pi Pico 2 (both cores, Arm Cortex-M33 and Hazard3 RISC-V, exact), and every benchmark re-measured with the current engine
and, all but FOMO, with the corrected validation split. Next:

- **The arithmetic of the first layer chosen per chip** (bits or bytes), from measurements, by the CDT.
- **Energy** has not been measured yet; latency and memory are.
- **FOMO re-trained** on the new fixed validation split.

## Repository layout

```
Symbex1_PIO/               PlatformIO project
  lib/lib_symbex/          the C engine (include/, src/, hal/), and tests/run_tests.py (bit-exact tests, every HAL)
  src/main.cpp             minimal example: a demo network checked against its PC simulation on your board
tools/
  cdt.py                   the CDT: configuration (TOML) -> train -> export -> validate -> chip
  chip_test.py             runs an exported model on a board: exactness, accuracy, latency per layer, RAM
  compiler_core/           the library behind the CDT
  configs/                 example.toml: every option, commented
  lowbit/                  INT8 / INT4 / INT2 baselines (Brevitas, ESP-NN kernels) for any benchmark
  dev/                     developer tools: fine_profile.py (time per library function), count_bench.py (raw
                           arithmetic of each chip), train_bench.py
benchmarks/                reproducible comparisons against TensorFlow Lite Micro: mlp/, conv1d/, conv2d/, experiments/
tests/                     Python tests (pytest -q from the root)
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
Abrahan -- GitHub: [abrahandejesusaparicio-code](https://github.com/abrahandejesusaparicio-code) -- FOMO person
detection on the classic ESP32

Spanish version: [README.es.md](README.es.md) -- License: see [LICENSE](LICENSE).
