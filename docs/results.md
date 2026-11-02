# Results: every benchmark in one place

Each benchmark compares a fully binary SYMBEX network against INT8 on the same chip and with the same architecture.
The INT8 side is the official TensorFlow Lite Micro model or, where no official model exists, the same network trained
with the same recipe. All the rules come from the [methodology](../README.md#methodology):

- the results are means over 3 seeds (42, 43, 44);
- every choice is made on a validation split, and the test set is used once;
- the weights are in flash unless the table says otherwise;
- on every board, the chip gives **the same output as the integer simulation on 100 % of the inputs**.

**What INT8 means here.** The INT8 side is the deployment path almost everyone uses: TensorFlow Lite Micro with
Espressif's optimized ESP-NN kernels. It is not the fastest INT8 possible. Calling the kernels directly, or other
libraries, can be faster, and [How fast could INT8 be?](#how-fast-could-int8-be) shows what is measured and what is not.

This page gives the headline of each benchmark. The details, the commands and the raw files are in each benchmark's
README.

## Speed, flash and RAM

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speedup_dark.svg">
  <img alt="SYMBEX speed-up over TensorFlow Lite Micro INT8 per benchmark and chip" src="assets/speedup.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/flash_dark.svg">
  <img alt="Flash used by each model, SYMBEX vs. INT8" src="assets/flash.svg">
</picture>

Every run on one core, at 240 MHz on both ESP32s. SYMBEX and INT8 always have their weights in the same place: flash
on every row but one.

| Benchmark | Chip | Weights in | Latency, SYMBEX / INT8 | Speed-up | Flash, SYMBEX / INT8 | RAM, SYMBEX / INT8 |
|---|---|---|---|---|---|---|
| MLP (Fashion-MNIST, 784-128-512-10) | ESP32-S3 | flash | 0.42 / 11.67 ms | **27.8×** | 27.3 / 187.0 KB | 4.6 / 6.7 KB |
| MLP (Fashion-MNIST, 784-128-512-10) | ESP32-S3 | **RAM** | 0.414 / 0.888 ms | **2.14×** | 27.3 / 187.0 KB | 4.6 / 6.7 KB + the weights (21 / 187 KB) |
| MLP (Fashion-MNIST, 784-128-512-10) | ESP32 | flash | 3.64 / 31.11 ms | **8.56×** | 27.3 / 187.0 KB | 4.4 / 6.7 KB |
| ResNet-8 (CIFAR-10) | ESP32-S3 | flash | 39.2 / 52.0 ms | **1.33×** | 24.3 / 96.2 KB | 14.8 / 80.5 KB |
| ResNet-8 (CIFAR-10) | ESP32 | flash | 63.3 / 717.8 ms | **11.3×** | 24.3 / 96.2 KB | 78.6 / 53.1 KB ¹ |
| FOMO person detection (32-32-128) | ESP32-S3 | flash | 126.3 / 256.9 ms | **2.03×** | 41.2 / 236.1 KB | 79.0 / 222.6 KB |
| FOMO person detection (32-32-128) | ESP32 | flash | 226.3 ms / does not fit | **only SYMBEX runs** | 41.2 / 236.1 KB | 142.8 KB ¹ / does not fit |
| Activity recognition (1D), 4 networks | ESP32-S3 | flash | 0.49-3.83 / 0.72-7.11 ms | **1.47-1.90×** | 0.7-6.5 / 7.3-42.5 KB | 1.3-2.7 / 2.7-25.5 KB |
| Activity recognition (1D), 4 networks | ESP32 | flash | 0.78-14.08 / 4.17-115.62 ms | **5.3-10.4×** | 0.7-6.5 / 7.3-42.5 KB | 1.3-2.7 / 2.7-10.3 KB |

- **Flash** is the whole model: weights, thresholds and scales for SYMBEX, the `.tflite` file for INT8.
- **RAM** is what the model uses while it runs. For SYMBEX, that is the buffers, the input and the stack, measured on
  the chip. For INT8, it is TFLite Micro's arena, which also holds the input.
- ¹ On the classic ESP32, SYMBEX counts bits with a 64 KB table in RAM, the fastest method there. Without it, ResNet-8
  needs 15 KB of RAM and runs in 71.0 ms (10.1×), and FOMO needs 79 KB and runs in 274.5 ms.
- The 1D rows are the ranges over Daghero's four networks, from the smallest to the largest.

The charts are drawn from the result files by `python tools/dev/docs_charts.py`, so a new measurement updates them.

The gap is set by the chip. The classic ESP32 has no SIMD, and there counting bits is an order of magnitude cheaper
than 8-bit multiply-accumulate. The ESP32-S3 has a SIMD unit (PIE) that speeds up INT8, so the gap is smaller. Where
INT8 can use that unit fully, the gap is narrowest: ResNet-8 at 1.33× and the MLP with its weights in RAM at 2.14×.
[How it works](how-it-works.md) explains why counting bits is cheaper.

## How it is measured

- **Clock.** The firmware fixes the clock in its build settings: 240 MHz on the ESP32 and the ESP32-S3
  (`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240`), 150 MHz on the Pico 2, 16 MHz on the Arduino Uno. Nobody measures the MHz;
  they are the setting both sides are built with.
- **Time.** The firmware reads the chip's microsecond timer (`esp_timer_get_time()`, `micros()` on the Uno) right
  before and right after one inference. Each input is timed separately, and the median over every input is reported
  (1,000 inputs in most runs). One core, nothing else running.
- **Same build for both sides.** The SYMBEX and TFLite firmwares use the same ESP-IDF settings: optimized for speed
  (`CONFIG_COMPILER_OPTIMIZATION_PERF`), the same clock and the same serial protocol.
- **Correctness.** Each SYMBEX output is checked against the integer simulation on the PC. Each TFLite output is
  checked against the TFLite interpreter on the PC.

## How fast could INT8 be?

The INT8 side is the official deployment path, already optimized: TensorFlow Lite Micro from Espressif
(`esp-tflite-micro`) with Espressif's ESP-NN kernels turned on (`CONFIG_NN_OPTIMIZED`). On the ESP32-S3, those kernels
use the PIE SIMD unit. The arena is sized to what each model needs.

Where INT8 could still go faster:

| Option | What we know |
|---|---|
| Calling ESP-NN directly, without the TFLite interpreter | **Measured** on the x1 MLP (784-256-128-10) on the ESP32-S3, weights in RAM: 0.556 ms instead of 0.876 ms (1.58× faster). SYMBEX x1 runs in 0.525 ms, so in RAM the two are close (`tools/lowbit`, [mlp_fashion](../benchmarks/mlp/mlp_fashion/README.md)). |
| Weights in RAM | **Measured** where the model fits: the MLP, and the 1D networks. In flash, INT8 is slow because it reads 8× more bytes through the flash cache. RAM removes that, if the chip has room. |
| ESP-DL (Espressif's other library, tuned for the S3) | **Not measured.** It may be faster than TFLite Micro on the S3; how much is unknown until it is measured. |
| CMSIS-NN on the Pico 2's Cortex-M33 | **Not measured yet** (future work). |

So the headline numbers compare SYMBEX with the INT8 path almost everyone deploys. They do not compare it with the
fastest INT8 possible. The fairest point today is the x1 MLP with its weights in RAM against ESP-NN called directly, and
there SYMBEX is only 1.06× faster. SYMBEX's clear wins are flash, RAM, chips without SIMD, and weights that stay in
flash.

## Accuracy: where binary costs

Binary weights and activations cost accuracy. How much depends on the task:

| Benchmark | SYMBEX | INT8 (same network) | Difference |
|---|---|---|---|
| FOMO person detection, F1 | **31.2 ± 0.7 %** | 30.1 ± 0.2 % | binary as good as INT8 |
| Digits on an Arduino Uno (1-bit input) | 94.5 ± 1.3 % | FP32 on the same input: 94.4 % | binarizing the network costs nothing |
| MLP, Fashion-MNIST (784-128-512-10) | 87.5 % | 89.8 % (QAT) | −2.3 points |
| ResNet-8, CIFAR-10 | 71.3 ± 0.8 % (73.1 % sized for binary) | 87.0 % (MLPerf Tiny) | −14 to −16 points |
| Activity recognition, large networks | 56.8-58.8 % | 66.2-67.9 % (TFLite) | about −9 points |

Detection, and small tasks with a binary input, binarize well. Classes that differ only in fine detail, like CIFAR-10's
ten, lose the most. With a byte-wide input such as an accelerometer, an int8 first layer is both faster and more
accurate than a binary one (see the 1D benchmark).

## The benchmarks

### ResNet-8 on CIFAR-10: against the official MLPerf Tiny model

[`benchmarks/conv2d/cnn_cifar_resnet8`](../benchmarks/conv2d/cnn_cifar_resnet8/README.md)

- The architecture is the same as the official INT8 ResNet-8 of MLPerf Tiny, measured with TFLite Micro on the same
  boards.
- **ESP32-S3:** 39.2 ms vs 52.0 ms (1.33×).
- **ESP32:** 63.3 ms with the weights in flash vs 717.8 ms (11.3×); 59.7 ms with the weights in RAM.
- **Pico 2:** 56.1 ms on Hazard3 and 74.5 ms on the M33.
- **Memory:** 24.3 KB of flash vs 96.2 KB, and 15 KB of RAM vs a 53-80 KB arena.
- **Accuracy:** 71.3 ± 0.8 % vs 87.0 %. With the network shaped for binary (16-32-128 / 1-1-2) it reaches 73.1 % at the
  same speed.

### FOMO person detection on Pascal VOC

[`benchmarks/conv2d/fomo_voc_person`](../benchmarks/conv2d/fomo_voc_person/README.md)

- The comparison is the same detector, trained the same way, against its INT8 version.
- **Detection:** F1 31.2 % vs 30.1 %, so the two detect alike.
- **ESP32-S3:** binary is 2.03× faster.
- **Flash:** 41 KB vs 236 KB.
- **Classic ESP32:** the binary detector runs in 226.3 ms; the INT8 one does not fit, because its arena overflows the
  RAM by 88 KB.
- **Pico 2:** 338.5 ms on Hazard3 and 416.4 ms on the M33.

### MLP on Fashion-MNIST: binary against INT8, INT4 and INT2

[`benchmarks/mlp/mlp_fashion`](../benchmarks/mlp/mlp_fashion/README.md)

- **Sized for binary** (784-128-512-10): 0.42 ms on the ESP32-S3 with the weights in flash vs 11.7 ms for TFLite INT8
  of the same network (28×), 27 KB of flash and 4.6 KB of RAM.
- **Accuracy per byte:** low-bit integers win. INT2 QAT is 1.0-1.6 points more accurate than a binary model of about
  the same size.
- **Speed and memory:** SYMBEX wins. The width works best when the first layer is narrow and the binary layers are wide.

### Digits on an Arduino Uno: binary from end to end

[`benchmarks/mlp/mlp_digits`](../benchmarks/mlp/mlp_digits/README.md)

- Input, weights and activations are all 1 bit.
- **Arduino Uno:** 94.5 % on handwritten digits in 5.3 ms, with 0.21 KB of RAM and 3.4 KB of flash.
- **The same network** runs in 0.11 ms on the ESP32-S3, 0.23 ms on the ESP32 and 0.38-0.40 ms on the Pico 2.

### Activity recognition with 1D convolutions: against ultracompactBNN and INT8

[`benchmarks/conv1d/har_ultracompact`](../benchmarks/conv1d/har_ultracompact/README.md)

- **Against ultracompactBNN** (Daghero et al. 2021), in its own environment: their processor (RI5CY on GVSoC), their
  four networks and their code unchanged. SYMBEX needs **1.07-1.37× fewer instructions**, and both are exact.
- **Against TFLite INT8** on the boards: 5.3-10.4× faster on the ESP32 and 1.47-1.90× on the ESP32-S3, with 6.5-10×
  less flash.
- **Where fully binary stops paying (a working theory):** with an 8-bit input, an int8 first layer is faster than the
  binary one and more accurate.

## Future work

- TFLite Micro with CMSIS-NN on the Pico 2's Cortex-M33, the INT8 side the Pico results are still missing.
- Energy per inference, measured with a current meter.
- FOMO's decision threshold calibrated again on the new validation split.
