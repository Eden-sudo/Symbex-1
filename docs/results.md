# Results: every benchmark in one place

Each benchmark compares a fully binary SYMBEX network against INT8 on the same chip and with the same architecture.
The INT8 side is the official TensorFlow Lite Micro model or, where no official model exists, the same network trained
with the same recipe. All the rules come from the [methodology](../README.md#methodology):

- the results are means over 3 seeds (42, 43, 44);
- every choice is made on a validation split, and the test set is used once;
- the weights are in flash unless the table says otherwise;
- on every board, the chip gives **the same output as the integer simulation on 100 % of the inputs**.

This page gives the headline of each benchmark. The details, the commands and the raw files are in each benchmark's
README.

## Speed and flash

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speedup_dark.svg">
  <img alt="SYMBEX speed-up over TensorFlow Lite Micro INT8 per benchmark and chip" src="assets/speedup.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/flash_dark.svg">
  <img alt="Flash used by each model, SYMBEX vs. INT8" src="assets/flash.svg">
</picture>

| Benchmark | Chip | SYMBEX | INT8 | SYMBEX vs INT8 |
|---|---|---|---|---|
| MLP (Fashion-MNIST), weights in flash | ESP32-S3 | 0.42 ms | 11.67 ms | **27.8×** |
| MLP (Fashion-MNIST), weights in RAM | ESP32-S3 | 0.414 ms | 0.888 ms | **2.14×** |
| MLP (Fashion-MNIST) | ESP32 | 3.64 ms | 31.11 ms | **8.56×** |
| ResNet-8 (CIFAR-10) | ESP32-S3 | 39.2 ms | 52.0 ms | **1.33×** |
| ResNet-8 (CIFAR-10) | ESP32 | 63.3 ms | 717.8 ms | **11.3×** |
| FOMO person detection | ESP32-S3 | 126.3 ms | 256.9 ms | **2.03×** |
| FOMO person detection | ESP32 | 226.3 ms | does not fit | **only SYMBEX runs** |
| Activity recognition (1D), 4 networks | ESP32-S3 | 0.49-3.83 ms | 0.72-7.11 ms | **1.47-1.90×** |
| Activity recognition (1D), 4 networks | ESP32 | 0.78-14.08 ms | 4.17-115.62 ms | **5.3-10.4×** |

SYMBEX needs 4-7× less flash: 27 KB vs 187 KB for the MLP, 24 KB vs 96 KB for ResNet-8 and 41 KB vs 236 KB for FOMO.
The charts are drawn from the result files by `python tools/dev/docs_charts.py`, so a new measurement updates them.

The gap is set by the chip. The classic ESP32 has no SIMD, and there counting bits is an order of magnitude cheaper
than 8-bit multiply-accumulate. The ESP32-S3 has a SIMD unit (PIE) that speeds up INT8, so the gap is smaller. Where
INT8 can use that unit fully, the gap is narrowest: ResNet-8 at 1.33× and the MLP with its weights in RAM at 2.14×.
[How it works](how-it-works.md) explains why counting bits is cheaper.

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
