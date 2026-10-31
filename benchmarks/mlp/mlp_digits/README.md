# mlp_digits -- a fully binary network on an Arduino Uno

**A network that is binary from end to end -- input, weights and activations, all 1 bit -- classifies handwritten
digits at 94.5% on an Arduino Uno, in 5.3 ms, with 0.21 KB of RAM and 3.4 KB of flash.** That is the accuracy an FP32
network reaches on the same 1-bit input (94.4%). Every result on the chip is identical, bit for bit, to the integer
simulation on the PC.

The task: scikit-learn Digits (1,797 images of 8x8 pixels, 17 grey levels, 10 classes). The original model is an FP32
MLP 64 -> 128 -> 10 (98.33% on test, 38.9 KB of weights: it does not even fit the Uno's 32 KB of flash). SYMBEX turns
it into a binary network with the CDT (`tools/cdt.py`), trained on the same data, and runs it with `lib_symbex`.

**Data split** (fixed for every seed and every method, `model.py`): test = 20% stratified (360 images), used once, at
the end; validation = 20% of the rest (288 images), where every choice is made (width, threshold, recipe); training =
the other 1,149 images. The FP32 original is trained on the training images only.

## Main result: fully binary, labels only

Every pixel becomes one bit (its most significant bit), every weight is one bit, every activation is one bit.
Network 64 -> 256 -> 10 (x2), trained on the labels only, 3 seeds. Test accuracy 94.54 +- 1.31% (validation
92.25 +- 0.20%), the same on every chip.

| Board | Arithmetic | Latency | RAM | Flash | Equal to the simulation |
|---|---|---|---|---|---|
| **Arduino Uno** (ATmega328P, 8-bit, 16 MHz) | XNOR | **5.34 ms** | **0.21 KB** | 3.4 KB | 100% |
| **ESP32** (240 MHz) | XNOR | **0.232 ms** | 0.66 KB | 3.4 KB | 100% |
| **ESP32-S3** (240 MHz, PIE SIMD) | AND | **0.109 ms** | 2.05 KB | 5.5 KB | 100% |
| **Raspberry Pi Pico 2**, Cortex-M33 (150 MHz) | AND | **0.396 ms** | 2.05 KB | 5.5 KB | 100% |
| **Raspberry Pi Pico 2**, Hazard3 RISC-V (150 MHz) | XNOR | **0.378 ms** | 1.13 KB | 3.4 KB | 100% |

Weights in flash, no lookup table in RAM (`popcount = "swar"`). Each chip runs the arithmetic that is fastest on it,
both measured: the other one takes 0.382 ms on the ESP32, 0.206 ms on the S3, 0.475 ms on the M33 and 0.475 ms on the
Hazard3. On the S3 the SIMD multiplier does the AND count (rows padded to 16 bytes, hence the larger flash).

How close is that to the limit? With the input reduced to one bit per pixel, an FP32 MLP (scikit-learn, 256 hidden
neurons, same split, 3 seeds) reaches 94.35 +- 0.16% on test. **The binary network matches it**: binarizing the network
costs nothing here; what costs is binarizing the input (FP32 on grey levels: 98.33%).

## Additional results

### Width of the fully binary network

1-bit input, labels only, 3 seeds. The width is chosen on validation: the three tie, so the smallest wins.

| Hidden neurons | Validation | Test | Uno latency | ESP32 latency | Flash (XNOR) |
|---|---|---|---|---|---|
| **256 (x2)** | **92.25 +- 0.20%** | **94.54 +- 1.31%** | **5.34 ms** | **0.232 ms** | **3.4 KB** |
| 512 (x4) | 92.13 +- 0.53% | 94.44 +- 0.73% | 9.93 ms | 0.444 ms | 6.7 KB |
| 1024 (x8) | 92.13 +- 0.40% | 94.54 +- 0.16% | 19.16 ms | -- | 13.4 KB |

The Uno runs the three widths with the current weights, 100% equal to the simulation. x4 on the ESP32: the October
model of that shape, remeasured with the current library.

### More bits at the input (everything else binary)

The first layer can take each pixel with k bits instead of 1 (k bit planes, still AND + count). Network x2
(64 -> 256 -> 10), trained with distillation from the FP32 original, 3 seeds:

| Bits per pixel | Validation | Test | Uno latency | Flash |
|---|---|---|---|---|
| 1 | 85.76 +- 1.39% | 88.52 +- 0.16% | 5.34 ms | 3.4 KB |
| 2 | 96.41 +- 0.40% | 96.57 +- 0.32% | 11.20 ms | 3.4 KB |
| **4** | **98.15 +- 0.53%** | **97.41 +- 0.42%** | **21.92 ms** | 3.4 KB |
| 8 | 98.03 +- 0.20% | 97.87 +- 0.42% | 25.63 ms | 3.4 KB |

4 bits reach the accuracy of 8 bits on validation: Digits has only 17 grey levels. With 1 bit, distillation hurts
(88.5% here vs 94.5% with the labels only, see below). With 4 bits both recipes tie (labels only: 98.26 +- 0.69%
validation, 97.87 +- 0.58% test).

Every Uno latency here is measured with the current weights, 3 seeds, 100% equal to the simulation.

### Against TensorFlow Lite Micro (INT8), 4-bit input

Same 360 test images, weights in flash, 3 seeds for SYMBEX (x2, 4-bit input, distillation); TFLite Micro is the
official INT8 conversion of the FP32 original (98.33%), 100% equal to the PC interpreter on both boards.

| Board | INT8, TFLite Micro | SYMBEX | Speed | Flash: INT8 / SYMBEX |
|---|---|---|---|---|
| Arduino Uno | not supported (AVR); the FP32 model does not fit | **21.92 ms**, 97.41% | only SYMBEX runs | -- / 3.4 KB |
| ESP32 | 0.763 ms, 98.33% | **0.382 ms**, 97.41% | **2.0x faster** | 14.7 / 5.5 KB |
| ESP32-S3 | 0.110 ms, 98.33% | **0.109 ms**, 97.41% | a tie (1.01x) | 14.7 / 5.5 KB |

SYMBEX runs AND on both ESP32 chips for this model (and on both cores of the Pico 2: 0.396 ms on the M33, 0.473 ms on
the Hazard3; TFLite Micro on the Pico 2 is future work). The fully binary network of the main result is faster still:
0.232 ms on the ESP32 (3.3x faster than INT8) at 94.5%. On the S3, where ESP-NN runs INT8 on the PIE SIMD unit,
this small network leaves SYMBEX no room: both take about 0.11 ms, and INT8 is 0.9 points more accurate.

### Where to cut the pixel (binary input)

The 1-bit input keeps a pixel when it is at least a threshold (on the original 0-16 scale). Network x4, labels only,
3 seeds:

| Threshold | Validation | Test |
|---|---|---|
| any ink (> 0) | 91.32 +- 0.69% | 90.74 +- 0.58% |
| >= 6 | **93.98 +- 0.40%** | 94.63 +- 0.85% |
| >= 8 (the most significant bit) | 92.13 +- 0.53% | 94.44 +- 0.73% |

On validation >= 6 is better (+1.9 points); on test both tie. The main result keeps the most significant bit because
it needs no step on the chip (a threshold of 6 costs one comparison per pixel before the network).

### Distillation: only with a teacher better than the student

Training the binary network to imitate the FP32 original (distillation) helps only when the original, seeing the
same input as the binary network, is clearly better than what the binary network reaches alone. Here it is not: the
original was trained on grey levels, and on binary input it drops to 71-91%. Test accuracy, 3 seeds:

| Configuration | Distillation | Labels only | The original on that input |
|---|---|---|---|
| 1-bit input (> 0), x4 | 74.44 +- 1.55% | **90.74 +- 0.58%** | 70.83% |
| 1-bit input (>= 6), x4 | 90.74 +- 1.05% | **94.63 +- 0.85%** | 91.39% |
| 1-bit input (>= 8), x4 | 88.52 +- 0.42% | **94.44 +- 0.73%** | 88.33% |
| 1-bit input (>= 8), x2 | 88.52 +- 0.16% | **94.54 +- 1.31%** | 88.33% |
| 4-bit input, x2 | 97.41 +- 0.42% | 97.87 +- 0.58% | 98.33% |

With a strong teacher the picture flips: see `cnn_cifar_resnet20`, where distillation from a published ResNet-20 adds
4.8 points. The same rule explains both.

## Reproduce

From `benchmarks/mlp/mlp_digits/` (environment `env_bench`).

**Option A -- the same FP32 model (recommended).** The FP32 original is published in `models/digits_mlp.pth`
(43 KB, sha256 `9dd155035e5f9a5881b2475bd1b85aadf8eb9b3a990087ab0872deedda5bfa2b`); `model.get_fp32` uses it when
present, and every summary file records the sha256 of the model it started from (`fp32_sha256`). The data split
needs no file: `model.py` rebuilds it from scikit-learn's Digits with a fixed seed.

**Option B -- everything from scratch.** Delete `models/digits_mlp.pth`: the first run retrains it (seconds). Training
is not bit-identical across machines, so the sha256 will differ and the results are expected to move within the seed
noise.

```
sha256sum models/digits_mlp.pth                                                                           # option A
python digits_symbex.py --seeds 3 --input-bits 1 --widths 2 --no-distill                                  # main, PC
python digits_symbex.py --seeds 3 --board uno --port /dev/ttyUSB1 --input-bits 1 --widths 2 --no-distill   # Uno
python digits_symbex.py --seeds 3 --board esp32 --port /dev/ttyUSB0 --input-bits 1 --widths 2 --no-distill --compare-arithmetic
python digits_symbex.py --seeds 3 --board pico2_riscv --port /dev/ttyACM0 --input-bits 1 --widths 2 --no-distill --compare-arithmetic
python digits_symbex.py --seeds 3 --input-bits 1 --widths 4 --no-distill                                  # widths (also 8)
python digits_symbex.py --seeds 3 --input-bits 4                                                          # 4-bit input
python digits_symbex.py --seeds 3 --input-bits 1 --widths 4 --no-distill --input-threshold 6              # thresholds
python digits_tflite.py --board esp32 --port /dev/ttyUSB0                                                 # TFLite
python digits_tflite.py --board s3 --port /dev/ttyACM0 --usb
```

The first run prepares the data (`../../data/digits/`) and, if missing, trains the FP32 original. Trained models stay in `runs/` (the chip stage re-exports in seconds, without retraining). The board measurements
of October 28 were made with `tools/chip_test.py` on the exported folders (both arithmetics, `--popcount swar`;
the ESP32-S3 through its native USB port with `--usb`).

## Files

| File | What it is |
|---|---|
| `models/digits_mlp.pth` | the FP32 original every result starts from (sha256 above) |
| `model.py` | the FP32 original (`DigitsMLP`) and the data (Digits, scaled to 0-255, the same split everywhere) |
| `symbex.toml` | the CDT configuration (Arduino Uno: 2 KB of RAM, 32 KB of flash, 50% budget) |
| `digits_symbex.py` | runs the CDT per seed and per board; summary tables |
| `digits_tflite.py` | the INT8 reference with TensorFlow Lite Micro (ESP32, ESP32-S3) |
| `results/symbex_seeds_<board>_b1_w2_nd.json` | main result (fully binary, x2, labels only), per board |
| `results/symbex_seeds_<board>_b4.json` | 4-bit input (x2), per board |
| `results/symbex_seeds_pc_*.json` | every variant on the PC: validation and test accuracy |
| `results/tflite_<board>.json` | TensorFlow Lite Micro |
| `results/symbex_s<seed>_*` | every seed: training, estimate, export and chip measurements |
| `results/earlier_test_validated/` | the earlier results, when the test images were also the validation set |

## Earlier version

Until October 28 the test images were also the validation set, so the width (x4) and the threshold were chosen on
test. With a validation split carved from the training data, the widths tie and the smallest (x2) is chosen: the
main result is now half the flash and twice as fast, at the same accuracy. The earlier results stay in
`results/earlier_test_validated/`.

Measured in October 2026 on an Arduino Uno (ATmega328P), an ESP32 dev board (PlatformIO `esp32dev`), an
ESP32-S3-DevKitC-1 and a Raspberry Pi Pico 2 (RP2350), all at their stock clocks; weights in flash, the minimum-RAM
counting mode (`popcount = "swar"`, no lookup table in RAM).
