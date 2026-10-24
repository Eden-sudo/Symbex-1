# mlp_digits -- a fully binary network on an Arduino Uno

**A network that is binary from end to end -- input, weights and activations, all 1 bit -- classifies handwritten
digits at 93.7% on an Arduino Uno, in 9.9 ms, with 0.27 KB of RAM.** Every result on the chip is identical, bit for
bit, to the integer simulation on the PC.

The task: scikit-learn Digits (1,797 images of 8x8 pixels, 17 grey levels, 10 classes), 20% stratified test split
(360 images). The original model is an FP32 MLP 64 -> 128 -> 10 (98.61%, 38.9 KB of weights: it does not even fit
the Uno's 32 KB of flash). SYMBEX turns it into a binary network with the CDT (`tools/cdt.py`), trained on the same
data, and runs it with `lib_symbex`.

## Main result: fully binary, no distillation

Every pixel becomes one bit, every weight is one bit, every activation is one bit. Network 64 -> 512 -> 10 (x4),
trained on the labels only, 3 seeds.

| Board | Accuracy | Latency | RAM | Flash | Equal to the simulation |
|---|---|---|---|---|---|
| **Arduino Uno** (ATmega328P, 8-bit, 16 MHz) | **93.70 +- 0.42%** | **9.92 ms** | **0.27 KB** | 6.7 KB | 100% |
| **ESP32** (240 MHz, XNOR) | 93.70 +- 0.42% | **0.444 ms** | 0.70 KB | 6.7 KB | 100% |
| **ESP32-S3** (240 MHz, AND with the PIE SIMD unit) | 93.70 +- 0.42% | **0.209 ms** | 3.0 KB | 10.8 KB | 100% |

Each chip runs the arithmetic that is fastest on it, chosen by the CDT from the chip's kernel profile
(`[output] arithmetic = "auto"`): XNOR on the Uno and the ESP32; AND on the ESP32-S3, where its SIMD multiplier does
the counting (1.9x faster than XNOR there; its rows are padded to 16 bytes, hence the larger flash).

How close is that to the limit? With the input reduced to one bit per pixel, even an FP32 network only reaches about
94.4% (measured with scikit-learn on the same split). The binary network keeps **99% of what its own input allows**:
binarizing the network costs almost nothing; what costs is binarizing the input.

## Additional results

### More bits at the input (everything else binary)

The first layer can take each pixel with k bits instead of 1 (k bit planes, still AND + count). Network x2
(64 -> 256 -> 10), Arduino Uno, 3 seeds:

| Bits per pixel | Accuracy | Latency | RAM | Flash |
|---|---|---|---|---|
| 1 | 87.69 +- 1.16% | 5.41 ms | 0.21 KB | 3.4 KB |
| 2 | 96.11 +- 0.28% | 11.36 ms | 0.22 KB | 3.4 KB |
| **4** | **98.24 +- 0.70%** | **21.88 ms** | 0.23 KB | 3.4 KB |
| 8 | 98.06 +- 0.56% | 25.80 ms | 0.21 KB | 3.4 KB |

4 bits reach the accuracy of 8 bits: Digits has only 17 grey levels. With 4 bits, the binary network is 0.4 points
below the FP32 original.

### Against TensorFlow Lite Micro (INT8), 4-bit input

Same images, weights in flash, 3 seeds for SYMBEX; TFLite Micro is the official INT8 conversion of the FP32 model.

| Board | INT8, TFLite Micro | SYMBEX | Speed | Flash: INT8 / SYMBEX |
|---|---|---|---|---|
| Arduino Uno | not supported (AVR); the FP32 model does not fit | **21.88 ms** | only SYMBEX runs | -- / 3.4 KB |
| ESP32 | 0.763 ms, 98.61% | **0.381 ms**, 98.24% | **2.0x faster** | 14.7 / 5.5 KB |
| ESP32-S3 | 0.116 ms, 98.61% | **0.110 ms**, 98.24% | **1.05x faster** | 14.7 / 5.5 KB |

SYMBEX picks its arithmetic per chip and per model (`[output] arithmetic = "auto"`, from the chip's kernel profile):
AND on both ESP32 chips for this model, XNOR on the Uno.

### Width of the fully binary network

| Hidden neurons | Accuracy | Uno latency | Flash |
|---|---|---|---|
| 256 (x2) | 87.69 +- 1.16% | 5.41 ms | 3.4 KB |
| **512 (x4)** | **93.70 +- 0.42%** | **9.92 ms** | **6.7 KB** |
| 1024 (x8) | 89.63 +- 1.85% * | 19.04 ms | 13.4 KB |

\* x2 and x8 were trained with distillation; x4 without it. See below why that matters.

### Where to cut the pixel (binary input)

The 1-bit input keeps a pixel when it is at least a threshold (on the original 0-16 scale). Network x4, no
distillation, 3 seeds:

| Threshold | Accuracy |
|---|---|
| any ink (> 0) | 91.39 +- 1.11% |
| >= 6 | 93.98 +- 0.58% |
| **>= 8 (the most significant bit; no extra step on the chip)** | **93.70 +- 0.42%** |

The default (the most significant bit) is within noise of the best threshold.

### Distillation: only with a teacher better than the student

Training the binary network to imitate the FP32 original (distillation) helps only when the original, seeing the
same input as the binary network, is clearly better than what the binary network reaches alone. Here it is not: the
original was trained on grey levels, and on binary input it drops to 71-91%.

| Configuration | Distillation | Labels only | The original on that input |
|---|---|---|---|
| 1-bit input (> 0), x4 | 74.91 +- 2.10% | **91.39 +- 1.11%** | 71.39% |
| 1-bit input (>= 6), x4 | 89.81 +- 0.98% | **93.98 +- 0.58%** | 90.56% |
| 1-bit input (>= 8), x4 | 89.81 +- 1.12% | **93.70 +- 0.42%** | 88.89% |
| 4-bit input, x2 | 98.24 +- 0.70% | 98.24 +- 0.16% | 98.61% |

With a strong teacher the picture flips: see `cnn_cifar_resnet20`, where distillation from a published ResNet-20 adds
4.8 points. The same rule explains both.

## Reproduce

From `benchmarks/mlp_digits/` (main environment `.venv`; the TFLite side in `env_bench`):

```
python digits_symbex.py --seeds 3 --input-bits 1 --widths 4 --no-distill                                  # PC
python digits_symbex.py --seeds 3 --board uno --port /dev/ttyUSB0 --input-bits 1 --widths 4 --no-distill   # main result
python digits_symbex.py --seeds 3 --board esp32 --port /dev/ttyUSB0 --input-bits 1 --widths 4 --no-distill --compare-arithmetic
python digits_symbex.py --seeds 3 --board s3 --port /dev/ttyUSB0 --input-bits 1 --widths 4 --no-distill --compare-arithmetic
python digits_symbex.py --seeds 3 --board uno --port /dev/ttyUSB0 --input-bits 4                          # 4-bit input
python digits_symbex.py --seeds 3 --input-bits 1 --widths 4 --no-distill --input-threshold 6              # thresholds
python digits_tflite.py --board esp32 --port /dev/ttyUSB0                                                 # TFLite (env_bench)
```

The first run prepares the data (`../data/digits/`) and trains the FP32 original (`models/digits_mlp.pth`).
Trained models stay in `runs/` (the chip stage re-exports in seconds, without retraining).

## Files

| File | What it is |
|---|---|
| `model.py` | the FP32 original (`DigitsMLP`) and the data (Digits, scaled to 0-255, the same split everywhere) |
| `symbex.toml` | the CDT configuration (Arduino Uno: 2 KB of RAM, 32 KB of flash, 50% budget) |
| `digits_symbex.py` | runs the CDT per seed and per board; summary tables |
| `digits_tflite.py` | the INT8 reference with TensorFlow Lite Micro (ESP32, ESP32-S3) |
| `results/symbex_seeds_<board>_b1_w4_nd.json` | main result (fully binary, x4, no distillation) |
| `results/symbex_seeds_<board>_b<k>.json` | k-bit input (x2) |
| `results/symbex_seeds_pc_*.json` | thresholds (`_t1`, `_t6`) and distillation (`_nd`) on the PC |
| `results/tflite_<board>.json` | TensorFlow Lite Micro |
| `results/symbex_s<seed>_*` | every seed: training, estimate, export and chip measurements |

Measured in October 2026 on an Arduino Uno (ATmega328P), an ESP32 dev board (PlatformIO `esp32dev`) and an
ESP32-S3-DevKitC-1, all at their stock clocks; weights in flash, the minimum-RAM counting mode (`popcount = "swar"`,
no lookup table in RAM).
