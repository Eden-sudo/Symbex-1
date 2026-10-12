# SYMBEX-1 -- Binary neural networks for microcontrollers

SYMBEX-1 turns a trained neural network into a **binarized network that runs on small microcontrollers**,
and measures it on real chips against the official INT8 toolchain.

**Version 3 achieves binarized convolutional networks running on the ESP32**, faster than the official INT8
models of the MLPerf Tiny benchmark, and a compiler (the CDT) that takes any FP32 model from training to a
validated binary model on the chip.

It has three parts:

- **`lib_symbex`** -- a portable C inference engine for binary layers (XNOR + popcount, integer thresholds).
- **The CDT** (`tools/cdt.py`) -- takes an FP32 model, your data and your hardware limits, and produces a
  binarized model, its C header, and a bit-exact validation of it, ready for the chip.
- **`benchmarks/`** -- reproducible comparisons against TensorFlow Lite / TensorFlow Lite Micro INT8, starting
  from the same FP32 model, on the same images, on the same chip.

## Why binary networks

A 1-bit weight is **8 times smaller** than an INT8 weight and 32 times smaller than an FP32 one, and a 1-bit
multiply-accumulate is an XNOR followed by a bit count. On a microcontroller, with kilobytes of RAM and a small
flash cache, that changes what fits and how fast it runs:

- **The model fits where INT8 does not.** A binary model can live in RAM, or fit in the flash cache, where the
  INT8 model of the same network cannot.
- **Accuracy can be bought back with width.** A binary layer is so cheap that the network can be made wider to
  recover the accuracy lost to binarization, and still be smaller and faster than INT8. INT8 cannot do the
  same without paying full price in memory.
- **The arithmetic is cheap even on hardware that was not designed for it.** Current microcontrollers are built
  around multiplication (int8 SIMD) and have no popcount instruction; even so, a proof of concept on the
  ESP32-S3 shows that its own SIMD multiplier can do the bit counting, and then a binary weight costs
  **less than an INT8 weight** on the same hardware (see Results).

## Results

### Binarized convolutional networks (SYMBEX v3 boolean engine)

Measured on the chip against the official INT8 models of MLPerf Tiny (TensorFlow Lite Micro), with several
training seeds. These results come from the v3 research engine; their benchmark folder is being prepared for
publication in `benchmarks/`.

| Task | SYMBEX model | Accuracy | ESP32 (no SIMD) | ESP32-S3 |
|---|---|---|---|---|
| CIFAR-10 | x1 | 71.44 +- 0.22% | **17.5x faster than INT8** (40.5 ms vs. 710 ms) | 1.52x faster (34.2 ms vs. 52 ms) |
| CIFAR-10 | x2 | 82.90 +- 0.29% | 3.7x faster than INT8 (189 ms) | -- |
| Visual Wake Words | x2 | 84.19 +- 0.47% (INT8: 86.71%) | 3.5x faster than INT8 (147 ms vs. 521 ms) | -- |

The v3 convolutional engine keeps a small **learned int8 encoder** as its first layer (the image enters there);
every layer after it is binary. Its intermediate RAM goes down to 16 KB with the software popcount.

### MLP on Fashion-MNIST, end to end through the CDT ([`benchmarks/mlp_fashion`](benchmarks/mlp_fashion/))

Same FP32 model (88.17%), official TFLite Micro INT8, same 1000 images, every layer of SYMBEX at 1 bit:

| | Accuracy | Weights | ESP32 (no SIMD) | ESP32-S3 |
|---|---|---|---|---|
| INT8, TensorFlow Lite Micro | 87.20% | 242 KB | 41.97 ms (does not fit in RAM) | 0.88 ms (model in RAM) |
| **SYMBEX, 1 bit, x2 width (3 seeds)** | 86.80 +- 0.36% | **69 KB** | **16.70 ms (2.5x faster)**, 25.54 ms with weights in flash (1.6x faster) | 14.78 ms |

The CDT made the binary network twice as wide to keep its accuracy within 0.4 points of INT8, and it is still
3.5x smaller and faster on the ESP32. On the ESP32-S3 the current engine does not use the SIMD multiplier yet
(next result).

### The cost of one weight on the ESP32-S3 (proof of concept, same hardware for both)

One layer of 256 x 512 weights in internal RAM, both sides using the native PIE SIMD multiply instruction
(`ee.vmulas`): INT8 through Espressif's ESP-NN kernel; binary as a 128-bit XOR followed by the bit count done
by the same multiplier against a vector of ones.

| Kernel | Cycles per weight | Speed | Weight memory |
|---|---|---|---|
| INT8, ESP-NN (PIE SIMD) | 0.531 | 1.00x | 128 KB |
| **Binary, PIE XOR + PIE multiplier** | **0.322** | **1.65x** | **16 KB** |

## How it works

1. **Binarization.** Weights and activations become -1 / +1 (one bit). A dot product of two 1-bit vectors of
   length n is `n - 2 * popcount(a XOR b)`: no multiplications.
2. **Everything else folds into integers.** The weight scale, BatchNorm, the activation and the binarization
   threshold of the next layer collapse into **one integer threshold per neuron**: `bit = (raw >= T)`.
   Inference has no floating point.
3. **The input stays as it comes.** The data (pixels, sensor values) enters the first layer as 8-bit integers;
   the weights of that layer are 1 bit too, and it is computed without multiplications (masks).
4. **The CDT decides how far to go.** It measures the sensitivity of every layer, starts with every layer at
   1 bit, widens the network or raises the bits of the most sensitive layer only if the accuracy drop is larger
   than you accept, within your hardware budget. It retrains with distillation from the original model, folds
   the result into integers, writes the C header, and checks it.
5. **Bit-exact validation.** The compiled C (on the PC and on the chip) must reproduce an integer simulation of
   the model **100%**: every exported model is checked, every time.

## Methodology

- **Same starting point:** one FP32 model, trained once; SYMBEX and INT8 both start from that file.
- **Independent baseline:** the INT8 model is produced by the official TensorFlow Lite converter and runs with
  the official TensorFlow Lite Micro, in a separate environment that imports nothing from SYMBEX.
- **Same images, same chip:** medians of 1000 inferences measured on the chip (`esp_timer`), one core, 240 MHz.
- **Checked results:** SYMBEX against its integer simulation, TFLite Micro against the TFLite interpreter on
  the PC; both must agree 100%.
- **Several seeds** for the SYMBEX training; accuracy as mean +- standard deviation.

## Current state and next steps

- **The engine does not use the S3 SIMD multiplier yet.** The proof of concept above shows the gain; it goes
  into the S3 hardware layer next. Until then, on the ESP32-S3 with the model in RAM, INT8 is faster.
- **The CDT does not export convolutional networks yet.** They are binarized and trained by the CDT, and they
  run on the chip with the v3 engine; joining both (CDT export to the v3 engine) is in progress.
- **The first layer receives 8-bit data**, and in the MLP it carries most of the operations (96% of the time).
  Splitting the input into bit planes is the next step.
- **Accuracy depends on the task**: within 0.4 points of INT8 on Fashion-MNIST; on CIFAR-10 the gap closes with
  width (x2) at a still lower cost than INT8 on the ESP32.
- **Energy has not been measured** yet; latency and memory are.

## Repository layout

```
lib_symbex/        C inference engine and its hardware abstraction layer (generic, ESP32, ESP32-S3, AVR)
                   (currently in Symbex1_PIO/lib/lib_symbex)
tools/
  cdt.py           the CDT: configuration (TOML) -> inspect -> estimate -> train -> export -> validate -> chip
  probar_chip.py   tests an exported model on a board: exactness, accuracy, latency per layer, RAM
  compiler_core/   the library behind the CDT (sensitivity, methods, training, export)
  configs/         example.toml: every option, commented
  tests/           tests of the CDT pieces (run from tools/: python -m tests.test_mezclas)
benchmarks/        reproducible comparisons against TensorFlow Lite Micro (see benchmarks/README.md)
docs/              documentation (English and Spanish)
```

## Quick start

```
# 1. configure: copy tools/configs/example.toml and fill in your model, data and hardware
# 2. run the CDT (from tools/)
python cdt.py my_model.toml
# 3. test the result on a board
python probar_chip.py <output folder> --board esp32 --port /dev/ttyUSB0
```

Requirements: Python with PyTorch; gcc (exactness check on the PC); PlatformIO and pyserial for the chip.

## Previous versions

Version 2 reported results on the scikit-learn Digits dataset with the `gated` and `bitslice` methods. During
the version 3 review, errors were found in those methods (the gates of `gated` did not receive a training
signal; the `bitslice` simulator did not correspond to the C engine). Those results are not reported here
until the methods are rebuilt and re-measured.

## Citation

A preprint describing SYMBEX-1 is in preparation; the reference will be added here.

## Author

Carlos Duarte (Edén) -- Panama -- GitHub: [Eden-sudo](https://github.com/Eden-sudo)

Spanish version: [README.es.md](README.es.md) -- License: see [LICENSE](LICENSE).
