# Results: every benchmark in one place

*[Leer en español](results.es.md)*

Each benchmark compares a fully binary SYMBEX network against INT8 on the same chip and with the same architecture.
The INT8 side is the official TensorFlow Lite Micro model or, where no official model exists, the same network trained
with the same recipe. All the rules come from the [methodology](../README.md#methodology):

- the results are means over 3 seeds (42, 43, 44);
- every choice is made on a validation split, and the test set is used once;
- the weights are in flash unless the table says otherwise;
- on every board, the chip gives **the same output as the integer simulation on 100 % of the inputs**.

**What INT8 means here.** The INT8 side is the deployment path almost everyone uses: TensorFlow Lite Micro with
Espressif's optimized ESP-NN kernels. It is not the fastest INT8 possible: on the ESP32-S3, Espressif's ESP-DL library
runs ResNet-8 2.6× faster than SYMBEX's standard engine. SYMBEX's vertical engine ([The vertical engine](vertical.md))
runs it in 10.4 ms, 1.43× faster than ESP-DL with the weights in flash; with the weights in RAM, ESP-DL stays 1.06×
faster, unless one of the engine's optional modes is used: compact (7.70 ms, 1.27× faster than ESP-DL in RAM, with
151 KB of RAM against ESP-DL's 147 KB) or prepared (6.44 ms, 1.52× faster, 238 KB of RAM). [How fast could INT8 be?](#how-fast-could-int8-be) shows what is measured and
what is not.

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
| ResNet-8, int8 first layer (optional) ² | ESP32-S3 | flash | 35.5 / 52.0 ms | **1.46×** | 24.0 / 96.2 KB | 14.3 / 80.5 KB |
| ResNet-8, int8 first layer (optional) ² | ESP32 | flash | 68.1 / 717.8 ms | **10.5×** | 24.0 / 96.2 KB | 78.1 / 53.1 KB ¹ |
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
- ² The same ResNet-8 with lib_symbex's optional int8 first layer on the pixels instead of the thermometer: 75.4 % on the
  test set against 71.3 % (3 seeds), faster on the S3 and the Pico 2's M33, slower on the ESP32 and Hazard3
  ([the benchmark](../benchmarks/conv2d/cnn_cifar_resnet8/README.md#an-int8-first-layer-optional)).

The charts are drawn from the result files by `python tools/dev/docs_charts.py`, so a new measurement updates them.

The gap is set by the chip. The classic ESP32 has no SIMD, and there counting bits is an order of magnitude cheaper
than 8-bit multiply-accumulate. The ESP32-S3 has a SIMD unit (PIE) that speeds up INT8, so the gap is smaller. Where
INT8 can use that unit fully, the gap is narrowest: ResNet-8 at 1.33× and the MLP with its weights in RAM at 2.14×.
There, Espressif's ESP-DL library is faster than SYMBEX's standard engine; on ResNet-8, [the vertical engine](vertical.md)
beats it with the weights in flash, and in RAM with its optional compact and prepared modes (see [How fast could INT8 be?](#how-fast-could-int8-be)).
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
| ESP-DL (Espressif's other library, tuned for the S3) | **Measured** in a quick check, table below. On the ESP32-S3 it is the fastest INT8: faster than SYMBEX's standard engine on ResNet-8 (the vertical engine beats it with the weights in flash, and in RAM with its compact mode at about the same RAM), and on the MLP when the weights are in RAM. |
| CMSIS-NN on the Pico 2's Cortex-M33 | **Measured** (TFLite Micro with Arm's CMSIS-NN kernels, DSP on). It does not catch SYMBEX: ResNet-8 453.0 ms in flash and 211.4 ms with the model in RAM, against SYMBEX's 74.5 ms (6.1× / 2.8×); MLP 784-128-512-10 31.35 / 2.89 ms against 4.57 / 1.53 ms (6.9× / 1.9×). |

**ESP-DL, quick check.** ESP-DL v3 (`espressif/esp-dl`, ESP-IDF 6.1), models quantized to INT8 with `esp-ppq`
(post-training, 512 calibration images; `target="esp32s3"` on the S3, `"c"` on the ESP32). ResNet-8 uses the MLPerf
Tiny float weights; the MLP is the same FP32 784-128-512-10 that the INT8 rows start from. Same clock and build
settings as every other run, median of 100 (ResNet-8) or 200 (MLP) test inputs, one build per row. "Flash" keeps the
weights in flash (`param_copy = false`); "RAM" copies them to RAM when the model loads.

| Model | Chip | Weights in | ESP-DL | SYMBEX | TFLite Micro | SYMBEX vs. ESP-DL |
|---|---|---|---|---|---|---|
| ResNet-8 | ESP32-S3 | flash | **14.8 ms** | 39.2 ms | 52.0 ms | ESP-DL **2.6× faster** |
| ResNet-8, SYMBEX vertical engine | ESP32-S3 | flash | 14.8 ms | **10.4 ms** (63 KB of RAM) | 52.0 ms | SYMBEX **1.43× faster** |
| ResNet-8, SYMBEX vertical engine | ESP32-S3 | RAM | **9.8 ms** (147 KB of RAM) | 10.4 ms (87 KB of RAM) | — | ESP-DL **1.06× faster** |
| ResNet-8, SYMBEX vertical engine, compact (optional) | ESP32-S3 | flash | 14.8 ms | **7.70 ms** (151 KB of RAM) | 52.0 ms | SYMBEX **1.92× faster** |
| ResNet-8, SYMBEX vertical engine, compact (optional) | ESP32-S3 | RAM | 9.8 ms (147 KB of RAM) | **7.70 ms** (175 KB of RAM) | — | SYMBEX **1.27× faster** |
| ResNet-8, SYMBEX vertical engine, prepared (optional) | ESP32-S3 | flash | 14.8 ms | **6.44 ms** (238 KB of RAM) | 52.0 ms | SYMBEX **2.3× faster** |
| ResNet-8, SYMBEX vertical engine, prepared (optional) | ESP32-S3 | RAM | 9.8 ms (147 KB of RAM) | **6.44 ms** (262 KB of RAM) | — | SYMBEX **1.52× faster**, with more RAM |
| ResNet-8 | ESP32 | flash | 712.4 ms | **63.3 ms** | 717.8 ms | SYMBEX **11.3× faster** |
| ResNet-8 | ESP32 | RAM | 603.4 ms | **59.7 ms** | — | SYMBEX **10.1× faster** |
| MLP 784-128-512-10 | ESP32-S3 | flash | 11.38 ms | **0.42 ms** | 11.67 ms | SYMBEX **27× faster** |
| MLP 784-128-512-10 | ESP32-S3 | RAM | **0.317 ms** (177 KB of RAM) | 0.414 ms | 0.888 ms | ESP-DL **1.31× faster** |
| MLP 784-128-512-10 | ESP32 | flash | 28.49 ms | **3.64 ms** | 31.11 ms | SYMBEX **7.8× faster** |
| MLP 784-128-512-10 | ESP32 | RAM | 4.84 ms (177 KB of RAM) | **3.64 ms** (in flash) | — | SYMBEX 1.33× faster, from flash |

Accuracy on those inputs: ResNet-8 86 / 100, the MLP 179 / 200, the same on both chips. This check is one run on a
subset of the test set, not yet a full benchmark with 3 seeds; it is enough to show where ESP-DL stands.

So the headline numbers compare SYMBEX with the INT8 path almost everyone deploys, not with the fastest INT8 possible.
On the ESP32-S3, ESP-DL is faster than SYMBEX's standard engine on ResNet-8 (2.6×) and on the MLP with its weights in
RAM (1.31×). [the vertical engine](vertical.md) turns ResNet-8 around with the weights in flash (10.4 ms, 1.43× faster than ESP-DL);
with the weights in RAM, ESP-DL is still 1.06× faster, unless an optional mode is used: compact (7.70 ms, 1.27× faster
than ESP-DL with 151 KB of RAM against 147 KB) or prepared (6.44 ms, 1.52× faster, 238 KB). SYMBEX's clear wins are chips without SIMD (the classic ESP32: 7.8-11.3× faster than ESP-DL
too), weights that stay in flash on dense layers, flash size (4-10× smaller) and RAM.

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
- **ESP32-S3:** 39.2 ms vs 52.0 ms (1.33×). Espressif's ESP-DL runs the same network in 14.8 ms, 2.6× faster than
  SYMBEX's standard engine (see [How fast could INT8 be?](#how-fast-could-int8-be)).
- **ESP32-S3, vertical engine:** the same network and bits in 10.36 ± 0.05 ms (3 seeds), 5.0× faster than TFLite Micro
  and 1.43× faster than ESP-DL with the weights in flash, 63 KB of RAM ([The vertical engine](vertical.md)). Its
  optional modes build lists and thresholds once into RAM: compact 7.70 ± 0.05 ms with 151 KB of RAM (1.27× faster than
  ESP-DL in RAM), prepared 6.44 ± 0.02 ms with 238 KB (8.1× faster than TFLite Micro, 1.52× faster than ESP-DL in RAM).
- **ESP32:** 63.3 ms with the weights in flash vs 717.8 ms (11.3×); 59.7 ms with the weights in RAM.
- **Pico 2:** 56.1 ms on Hazard3 and 74.5 ms on the M33; on the M33 the vertical engine's prepared mode runs it in
  57.4 ms (seed 42, a ranking; [The vertical engine](vertical.md#other-chips-pico-2)). TFLite Micro INT8 with CMSIS-NN
  on the M33: 453.0 ms (6.1×), 211.4 ms with the model in RAM; on Hazard3, with the reference kernels: 5702 ms.
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

- Energy per inference, measured with a current meter.
- ESP-DL as a full INT8 baseline (3 seeds, the whole test set, every benchmark), with its scripts in the repository.
- Beating ESP-DL on the ESP32-S3 with the weights in RAM with clearly less RAM (default vertical engine: 10.4 ms vs
  9.8 ms with 63 KB; the compact mode wins at 7.70 ms with about the same RAM, 151 KB against 147 KB), and the
  vertical engine for the MLP and the 1D networks.
- FOMO's decision threshold calibrated again on the new validation split.
