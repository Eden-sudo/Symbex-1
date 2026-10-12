# mlp_fashion -- 1-bit MLP vs. official INT8 on the ESP32 and the ESP32-S3

**Question:** starting from the same FP32 Fashion-MNIST classifier, how does the SYMBEX 1-bit model compare
with the INT8 model of the official toolchain (TensorFlow Lite + TensorFlow Lite Micro), on the same
microcontroller and the same images?

## Model

`modelo.py` -- FashionMLP, FP32: 784 -> 256 -> 128 -> 10, hidden layers Linear -> BatchNorm1d -> ReLU.
Input: 28x28 grayscale, normalized as (x / 255 - 0.2860) / 0.3530. Trained 5 epochs with Adam (seed 42).
FP32 accuracy: **88.17%** (10,000 test images). Both sides start from this same `models/fashion_mlp.pth`.

## The two sides

| | SYMBEX (`fashion_symbex.py`) | Official INT8 (`fashion_tflite.py`) |
|---|---|---|
| Quantization | CDT: every layer 1 bit, width search, 15 epochs with distillation from the FP32 model | `tf.lite.TFLiteConverter`, full-integer INT8, 1000 training images for calibration, no retraining |
| Model chosen | width x2 (784 -> 512 -> 256 -> 10) in all 3 seeds | the original 784 -> 256 -> 128 -> 10 |
| On the chip | `lib_symbex` (`dense` layer; first layer with masks, SWAR popcount) | TensorFlow Lite Micro (esp-tflite-micro, Espressif's ESP-NN kernels) |
| Checked against | its own integer simulation: **100%** equal on every run | the TFLite interpreter on the PC: **100%** equal on every run |
| Uses SYMBEX code | yes | **no** (independent environment, `env_bench`) |

The same 1000 validation images (the first Fashion-MNIST test images) are sent to both models on the chip.

## Configuration (`symbex.toml`)

- `options = ["bin"]`: every layer at 1 bit. The CDT widened the hidden layers to x2 (`widths = [1.0, 1.5, 2.0]`)
  because the 1-bit drop at x1 was above `max_accuracy_drop` (2 points) in its 5-epoch quick trial.
- `[hardware]`: ESP32 (320 KB RAM, 4 MB flash), `budget = 0.25` as a ceiling.
- On the chip: `--popcount swar` (no lookup table), `--first-layer mascara` (the fastest variant measured on
  both boards, now the default), `--weights flash` or `ram`.

## Results

### Accuracy (3 seeds for SYMBEX, all from the same FP32 model)

| Model | 10,000 validation images | the 1000 images sent to the chip |
|---|---|---|
| FP32 original | 88.17% | -- |
| INT8, official TFLite | -- | 87.20% |
| **SYMBEX, 1 bit, x2 (seeds 42, 43, 44)** | **87.02 +- 0.29%** | **86.80 +- 0.36%** |

The chip computes exactly the integer simulation, so the accuracy of a SYMBEX model is the same on every board.

### Classic ESP32 (ESP32-D0WD-V3, 240 MHz, one core, no SIMD)

| Model | Latency | vs. TFLite | Flash | RAM |
|---|---|---|---|---|
| INT8, official TFLite Micro (model in flash) | 41.97 ms | 1.00x | 242.2 KB | 4.8 KB (arena) |
| INT8, official TFLite Micro (model in RAM) | does not fit: 242 KB of weights > available DRAM | -- | -- | -- |
| **SYMBEX, weights in flash** | **25.54 ms** | **1.64x** | **69.4 KB** | 4.1 KB |
| **SYMBEX, weights in RAM** | **16.70 ms** | **2.51x** | 69.4 KB | 73.6 KB |

### ESP32-S3 (240 MHz, one core, PIE SIMD)

| Model | Latency | vs. TFLite (RAM) | Flash | RAM |
|---|---|---|---|---|
| **INT8, official TFLite Micro (model in RAM)** | **0.876 ms** | **1.00x** | 242.2 KB | 4.8 KB + 242 KB model copy |
| INT8, official TFLite Micro (model in flash) | 15.88 ms | 0.06x | 242.2 KB | 4.8 KB |
| SYMBEX, weights in RAM | 14.78 ms | 0.06x | 69.4 KB | 73.8 KB |
| SYMBEX, weights in flash | 16.92 ms | 0.05x | 69.4 KB | 4.4 KB |

### Where the time goes (per layer, measured on the chip)

| | layer 0 (784 inputs) | layer 1 | layer 2 |
|---|---|---|---|
| SYMBEX, ESP32-S3, weights in RAM | 14.26 ms (uint8 x 1 bit, 784 x 512) | 0.49 ms (1 bit x 1 bit, 512 x 256) | 0.02 ms |
| INT8 (CDT reference, ESP-NN), ESP32-S3, weights in flash | 13.45 ms (784 x 256) | 2.19 ms (256 x 128) | 0.09 ms |

## Interpretation

- **On the classic ESP32, SYMBEX is faster in every configuration**: 1.6x with its weights in flash and 2.5x with
  them in RAM. The binary model is 3.5x smaller: it fits in RAM and in the flash cache, where the INT8 model does
  not. That is the advantage of binarization working as intended.
- **Width buys the accuracy back.** The CDT made the binary network twice as wide; it still is 3.5x smaller and
  faster than the INT8 network, within 0.4 points of its accuracy.
- **Every layer of SYMBEX is binary, the first one included.** Only the input data is 8-bit (the pixels); the
  first layer has 1-bit weights and is computed with masks, without multiplications. It is the layer with the
  most operations (784 x 512) and each of them adds an 8-bit value, so it takes 96-98% of the time.
- **On the ESP32-S3, INT8 with its model in RAM is faster (0.876 ms)** because it uses the PIE SIMD multiplier and
  the current SYMBEX engine does not. The S3 has no popcount instruction, but its multiplier can do the counting:
  `../poc_s3_bin_vs_int8.py` measures a binary weight at 0.322 cycles against 0.531 for an ESP-NN INT8 weight
  (1.65x faster, 8x less memory). That kernel goes into the engine next, and this benchmark will be re-measured.

## How to reproduce

```
# main environment (.venv), from benchmarks/mlp_fashion/
python fashion_symbex.py --semillas 3 --board esp32 --port /dev/ttyUSB0 --weights ram
python fashion_symbex.py --board s3 --port /dev/ttyUSB0 --barrido          # every chip configuration
# env_bench, from benchmarks/mlp_fashion/
python fashion_tflite.py --board esp32 --port /dev/ttyUSB0
python fashion_tflite.py --board s3 --port /dev/ttyUSB0
python fashion_tflite.py --board s3 --port /dev/ttyUSB0 --modelo-ram
python fashion_tflite.py --board s3 --solo-tabla                            # the comparison table, no board
```

Results land in `results/` (`symbex_*.json`, `tflite_*.json`); everything generated goes to `runs/`.
Measured on 2026-10-11. Timings are medians of 1000 inferences, measured on the chip with `esp_timer`.
