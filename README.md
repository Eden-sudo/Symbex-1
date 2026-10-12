# SYMBEX-1 -- Binary neural networks for microcontrollers

SYMBEX-1 turns a trained neural network into a **1-bit network that runs on small microcontrollers**, and measures,
on real chips, what that gains and what it costs against the official INT8 toolchain.

It has three parts:

- **`lib_symbex`** -- a portable C inference engine for binary layers (XNOR + popcount, integer thresholds).
- **The CDT** (`tools/cdt.py`) -- takes an FP32 model, your data and your hardware limits, and produces a
  binarized model, its C header, and a bit-exact validation of it, ready for the chip.
- **`benchmarks/`** -- reproducible comparisons against TensorFlow Lite / TensorFlow Lite Micro INT8, starting
  from the same FP32 model, on the same images, on the same chip.

## The claim, stated precisely

On the hardware that needs it most -- **microcontrollers without SIMD and with little memory** -- a binary
network reaches an accuracy close to INT8 with **several times less memory** and **faster inference**.
On hardware built around multiply-accumulate units (SIMD for int8), the binary network still saves memory,
but loses speed: not because its arithmetic is less efficient, but because that hardware has instructions
for multiplication and **none for the operation binary networks need (popcount)**.

## Why

Microcontrollers have kilobytes, not gigabytes: tens to hundreds of KB of RAM, a few MB of flash, often no
FPU and no vector unit. A standard model does not fit, and an INT8 model often fits only in flash, where every
weight read goes through a small cache. A 1-bit weight is 8 times smaller than an INT8 one and 32 times
smaller than an FP32 one, and a 1-bit multiply-accumulate is an XNOR followed by a bit count.

## How it works

1. **Binarization.** Weights and activations become -1 / +1 (one bit). A dot product of two 1-bit vectors of
   length n is `n - 2 * popcount(a XOR b)`: no multiplications.
2. **Everything else folds into integers.** The weight scale, BatchNorm, the activation and the binarization
   threshold of the next layer collapse into **one integer threshold per neuron**: `bit = (raw >= T)`.
   Inference has no floating point and no multiplications in the binary layers.
3. **The first layer keeps 8-bit input.** Pixels (or sensor values) enter as uint8 against 1-bit weights,
   computed exactly in integers.
4. **The CDT decides how far to go.** It measures the sensitivity of every layer, starts with every layer at
   1 bit, and only widens the network or raises the bits of the most sensitive layer when the accuracy drop
   is larger than you accept -- with your hardware budget as a ceiling. It retrains with distillation from
   the original model, folds the result into integers, writes the C header, and checks it.
5. **Bit-exact validation.** The compiled C (on the PC and on the chip) must reproduce an integer simulation
   of the model **100%**: every exported model is checked, every time.

## Methodology

Every comparison in `benchmarks/` follows the same rules:

- **Same starting point:** one FP32 model, trained once; both SYMBEX and INT8 start from that file.
- **Independent baseline:** the INT8 model is produced by the official TensorFlow Lite converter and runs with
  the official TensorFlow Lite Micro, in a separate environment that imports nothing from SYMBEX.
- **Same images, same chip:** both models receive the same validation images over serial; timings are
  medians of 1000 inferences measured on the chip (`esp_timer`), one core, 240 MHz.
- **Checked results:** SYMBEX against its integer simulation, TFLite Micro against the TFLite interpreter on
  the PC. Both must agree 100%.
- **Several seeds** for the SYMBEX training; accuracy reported as mean +- standard deviation.

## Results

**MLP on Fashion-MNIST** ([`benchmarks/mlp_fashion`](benchmarks/mlp_fashion/)) -- FP32 original: 88.17%.

| | Accuracy | Flash | ESP32 (no SIMD) | ESP32-S3 (int8 SIMD) |
|---|---|---|---|---|
| INT8, TensorFlow Lite Micro | 87.20% | 242 KB | 41.97 ms (model in flash; does not fit in RAM) | **0.88 ms** (model in RAM) |
| SYMBEX, 1 bit (3 seeds) | 86.80 +- 0.36% | **69 KB** | **16.70 ms** (weights in RAM) / 25.54 ms (flash) | 14.78 ms (weights in RAM) |

Further results of the SYMBEX v3 boolean engine on **CIFAR-10** and **Visual Wake Words** (convolutional,
ESP32 and ESP32-S3, against the official MLPerf Tiny INT8 models) are being moved into `benchmarks/`.

## Limitations

- **Hardware built for multiplication wins on speed.** On the ESP32-S3, the INT8 model in RAM runs 17x faster
  than SYMBEX: its SIMD does 16 int8 multiply-adds per instruction, and the chip has no popcount instruction,
  so SYMBEX counts bits in software.
- **On the classic ESP32, part of the speed advantage comes from memory**, not arithmetic: the INT8 weights do
  not fit in the flash cache or in RAM; the binary ones do.
- **Accuracy costs depend on the task.** Small on Fashion-MNIST (about 0.4 points against INT8); several
  points on CIFAR-10.
- **The first layer is the bottleneck** of the current engine (8-bit input, 96-98% of the time in the MLP).
- **Not yet supported by the CDT:** exporting convolutional networks (they train, but do not reach the chip
  yet), k-bit layers on the chip (mixed precision trains in Python only), residual networks for widening.
- **Energy has not been measured.** Only latency and memory are reported.

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

## Status

Active development (version 3). Working end to end: MLPs at 1 bit, from FP32 model to measured chip.
In progress: convolutional export, integrating the boolean v3 engine into `lib_symbex`, k-bit layers on the chip.

## Previous versions

Version 2 reported results on the scikit-learn Digits dataset with the `gated` and `bitslice` methods. During
the version 3 review, errors were found in those methods (the gates of `gated` did not receive a training
signal; the `bitslice` simulator did not correspond to the C engine). Those results are not reported here
until the methods are rebuilt and re-measured.

## Citation

A preprint describing SYMBEX-1 is in preparation; the reference will be added here.

## Author

Carlos Duarte (Edén) -- Panama -- GitHub: [Eden-sudo](https://github.com/Eden-sudo)

Spanish version: [README.es.md](README.es.md)

License: see [LICENSE](LICENSE).
