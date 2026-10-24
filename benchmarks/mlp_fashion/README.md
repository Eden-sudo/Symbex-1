# mlp_fashion -- binary MLP vs. official INT8 (TensorFlow Lite Micro) on the ESP32 and the ESP32-S3

**A fully binary network -- input, weights and activations, all 1 bit -- classifies Fashion-MNIST at 83.3% and runs
4.7x faster than the official INT8 model (TensorFlow Lite Micro) on the ESP32 and 4.2x faster on the ESP32-S3, with
5x less flash and a third of its RAM.** With 8-bit pixels in the first layer, the binary network reaches 88.5%, above
the FP32 original. Every result on the chip is identical, bit for bit, to the integer simulation.

## Model

`model.py` -- FashionMLP, FP32: 784 -> 256 -> 128 -> 10, hidden layers Linear -> BatchNorm1d -> ReLU. Input: 28x28
grayscale, normalized as (x / 255 - 0.2860) / 0.3530. Trained 5 epochs with Adam (seed 42). FP32 accuracy: **88.17%**
(10,000 test images). Both sides start from this same `models/fashion_mlp.pth`.

## The two sides

| | SYMBEX (`fashion_symbex.py`) | Official INT8 (`fashion_tflite.py`) |
|---|---|---|
| Quantization | the CDT: every layer 1 bit, hidden layers x1.5 (784 -> 384 -> 192 -> 10), 15 epochs, retrained on the same data | `tf.lite.TFLiteConverter`, full-integer INT8, 1000 training images for calibration, no retraining |
| On the chip | `lib_symbex`, exported as a C header; two arithmetics with identical results (XNOR + popcount, AND + count), chosen per chip | TensorFlow Lite Micro (esp-tflite-micro, Espressif's ESP-NN kernels) |
| Checked against | its integer simulation: **100%** equal on every run | the TFLite interpreter on the PC: **100%** equal on every run |
| Environment | main (`.venv`) | `env_bench` (imports nothing from SYMBEX) |

The same 1000 validation images (the first Fashion-MNIST test images) go to both models on the chip. Timings are
medians of 1000 inferences measured on the chip (`esp_timer`), one core, 240 MHz. SYMBEX: 3 training seeds (42, 43,
44), all from the same FP32 model.

## Main result: fully binary, no distillation

Every pixel becomes one bit (its most significant bit), every weight is one bit, every activation is one bit;
trained on the labels only, 3 seeds.

| Network | Accuracy (10,000 images) | Flash |
|---|---|---|
| **x1.5 (784 -> 384 -> 192 -> 10)** | **83.28 +- 0.26%** | **49.1 KB** |
| x4 (784 -> 1024 -> 512 -> 10) | 84.12 +- 0.15% | 185 KB |

On the chip (x1.5, weights in flash, 1000 images, 100% equal to the simulation):

| Board | Model | Latency | vs. TFLite | Flash | RAM |
|---|---|---|---|---|---|
| ESP32 | INT8, TFLite Micro (87.20%) | 41.97 ms | 1.00x | 242.2 KB | 4.8 KB |
| ESP32 | **SYMBEX, fully binary, XNOR** | **8.97 ms** | **4.68x** | **49.1 KB** | **1.5 KB** |
| ESP32-S3 | INT8, TFLite Micro (87.20%) | 15.89 ms | 1.00x | 242.2 KB | 4.8 KB |
| ESP32-S3 | **SYMBEX, fully binary, XNOR** | **3.75 ms** | **4.24x** | **49.1 KB** | **1.7 KB** |

**The binary network gets far more out of a binary input than the original does:** fed the same 1-bit pixels, the
FP32 original reaches 77.88%; the binary network, 83.28% (+5.4 points). The distance to the 8-bit results (88.5%)
comes from where the pixel is cut: one fixed threshold (128) for every pixel. Learning that threshold per pixel during
training is the next step -- on the chip it is the same comparison, so it costs nothing; in the `experiments` folder,
a fully binary network with learned input thresholds reaches 86.3% on this task.

## Additional results: 8-bit pixels in the first layer

The first layer takes each pixel as 8 bits (uint8 x 1-bit weights, integer sums); every other layer is 1 bit x 1 bit.

### Accuracy

| Model | 10,000 test images | the 1000 images sent to the chip |
|---|---|---|
| FP32 original | 88.17% | -- |
| INT8, official TFLite | -- | 87.20% |
| **SYMBEX, x1.5, labels only (3 seeds)** | **88.50 +- 0.09%** | *(pending)* |
| SYMBEX, x1.5, distilled from the FP32 model (3 seeds) | 86.94 +- 0.13% | 87.03 +- 0.32% |

Trained on the labels only, the binary model is **above the FP32 original** (88.50 vs. 88.17%). Distillation from
this FP32 model costs 1.6 points: the teacher is not better than what the student reaches alone (see `mlp_digits`
and `cnn_cifar_resnet20` for the rule: distillation helps only with a stronger teacher).

### Weights in flash (the usual deployment)

Latency of the distilled models (the same network, so the same speed as the label-only ones).

| Board | Model | Latency | vs. TFLite | Flash | RAM |
|---|---|---|---|---|---|
| ESP32 (no SIMD) | INT8, TFLite Micro | 41.97 ms | 1.00x | 242.2 KB | 4.8 KB |
| ESP32 | **SYMBEX, XNOR** | **12.14 ms** | **3.46x** | **49.1 KB** | 1.4 KB |
| ESP32 | SYMBEX, AND | 12.95 ms | 3.24x | 54.5 KB | 3.8 KB |
| ESP32-S3 (PIE SIMD) | INT8, TFLite Micro | 15.89 ms | 1.00x | 242.2 KB | 4.8 KB |
| ESP32-S3 | **SYMBEX, AND** | **3.87 ms** | **4.11x** | **54.5 KB** | 4.1 KB |
| ESP32-S3 | SYMBEX, XNOR | 6.22 ms | 2.55x | 49.1 KB | 1.6 KB |

### Weights in RAM (best case of each one, when the model fits)

| Board | Model | Latency | vs. TFLite | RAM |
|---|---|---|---|---|
| ESP32 | INT8, TFLite Micro | does not fit (242 KB of weights) | -- | -- |
| ESP32 | **SYMBEX, AND** | **5.81 ms** | -- | 58.3 KB |
| ESP32-S3 | INT8, TFLite Micro (model copied to RAM) | 0.876 ms | 1.00x | 247 KB |
| ESP32-S3 | SYMBEX, AND | *(pending; 0.870 ms with the previous library)* | | 58.5 KB |

## Interpretation

- **In the usual deployment (weights in flash), SYMBEX is faster than TensorFlow Lite Micro on both chips**: 3.5x on
  the ESP32 and 4.1x on the ESP32-S3, with 4.4-4.9x less flash and less RAM.
- **With everything in RAM**, the classic ESP32 runs the SYMBEX model in 5.8 ms; the INT8 model does not fit there at
  all. On the ESP32-S3, TFLite Micro needs 247 KB of RAM for its best case; SYMBEX needs 58.5 KB.
- **Each chip prefers a different arithmetic, and the CDT measures it instead of assuming it**
  (`tools/chip_test.py --kernels` builds a profile per chip; `[output] arithmetic = "auto"` predicts the cost of every
  layer from it). The ESP32-S3 has no popcount instruction but an 8-bit SIMD multiplier: the AND arithmetic lets it
  do the counting (1.6x faster than XNOR on this model). The classic ESP32 has neither: there XNOR wins by 7%, with
  rows that need no padding (49.1 vs. 54.5 KB).
- **The first layer dominates** (86-89% of the time): it receives 8-bit pixels, every other layer is 1 bit x 1 bit.
  On the ESP32 half of that time is reading 37.6 KB of first-layer weights from flash (they do not fit its cache):
  with the weights in RAM the same model runs in 5.8 ms instead of 12.1 ms.

## Limitations

- One task, one FP32 model; SYMBEX trained with 3 seeds (the INT8 model has no training, so no seeds).
- Energy has not been measured; latency and memory have.

## Low-bit baselines: INT4 and INT2 (PC only)

`fashion_int4.py` quantizes the same FP32 architecture (784-256-128-10) with per-channel symmetric INT4 / INT2
weights and int8 activations, by PTQ (max-abs, MSE clipping) and QAT (STE, learnable scales, the CDT's KD recipe),
and evaluates it in integer arithmetic. `pareto.py` collects every `results/*.json` into accuracy vs. packed
weight bytes (`results/pareto.json`).

| Model | Accuracy (10k) | Weights |
|---|---|---|
| INT2 QAT, 15 epochs, 3 seeds | 88.15% | 58.7 KB |
| SYMBEX 1 bit x2, 3 seeds | 87.02% | 66.9 KB |
| INT4 QAT, 15 epochs, 3 seeds | 88.26% | 117.4 KB |
| INT8 QAT, 5 epochs, 3 seeds | 88.27% | 234.8 KB |

On weight bytes alone, **INT2 QAT dominates SYMBEX 1 bit x2** on this MLP: smaller and about 1.1 points more
accurate. SYMBEX's remaining advantage is compute (1-bit activations: XNOR/popcount instead of int8 MACs), which
these PC numbers do not measure. Caveats: the INT runs use a locally retrained FP32 model (87.82% vs. 88.17%);
INT2 here is ternary (3 of 4 levels); requantization is simulated with a float multiplier.

```
python fashion_int4.py --bits 4 2 --semillas 3 --epocas 15
python pareto.py
```

## PopLUT probe: codebook-compressed binary weights (PC only)

`poplut_probe.py` tests whether the 1-bit weights of a trained SYMBEX x2 model can be stored as indices into a
small codebook of 32-bit words. Every weight row of `red.1` and `red.4` is cut into 32-bit words; each word is
stored as a codebook index plus a complement bit (the word is `c_k` or `~c_k`). `red.7` stays plain 1 bit. The
codebook is learned by binary k-means in Hamming distance, then everything except the compressed signs (BatchNorm,
input thresholds, last layer) is fine-tuned for 3 epochs.

Base model: SYMBEX x2, seed 42, 15 epochs, 86.34% on the 10k test images, 66,880 B of 1-bit weights.

| Codebook K | Post hoc | After fine-tune | Weights |
|---|---|---|---|
| 16 | 34.23% | 77.30% | 11,008 B |
| 64 | 51.83% | 81.02% | 15,616 B |
| 256 | 73.37% | 84.04% | 21,376 B |

At K=256 the compressed layers cost about 0.31-0.34 bits per weight, with 19-20% of the weight bits changed
(Hamming). The kill criterion was fixed before running: K=256 after fine-tuning losing more than 1 point against
the base model discards PopLUT as a direct replacement. It loses 2.30 points, so **the verdict is discarded**.

As a size point it is still of some interest. The binary x1 model (784-256-128-10), 15 epochs with the current CDT
(validation split held out of train), reaches 82.85% on test with about 29.3 KB of 1-bit weights
(`results/symbex_x1_e15_train.json`, field `test.acc_symbex`). PopLUT K=256 on x2 is 1.19 points more accurate and
about 27% smaller than that x1. INT2 QAT (88.15%, 58.7 KB) still dominates in accuracy.

Caveats: the x2 model was trained before the val/test leak fix (last checkpoint, no selection); a single k-means
seed; not tested on the chip; the C lookup kernel does not exist yet.

```
python poplut_probe.py --modelo runs/symbex_local_s42/model_binarized.pth
```

## How to reproduce

```
# main environment (.venv), from benchmarks/mlp_fashion/
python ../../tools/chip_test.py --kernels --board s3 --popcount swar          # chip profile (once per chip)

python fashion_symbex.py --seeds 3 --input-bits 1 --no-distill --widths 1.5   # main result (PC)
python fashion_symbex.py --seeds 3 --input-bits 1 --no-distill --widths 1.5 --board esp32 --weights flash --compare-arithmetic
python fashion_symbex.py --seeds 3 --no-distill                               # 8-bit input, labels only (PC)
python fashion_symbex.py --seeds 3 --board s3 --weights flash --compare-arithmetic
python fashion_symbex.py --seeds 3 --board s3 --weights ram --compare-arithmetic

# env_bench, from benchmarks/mlp_fashion/
python fashion_tflite.py --board s3                                             # model in flash
python fashion_tflite.py --board s3 --model-ram                                 # model in RAM (S3 only)
python fashion_tflite.py --board s3 --table-only                                # both tables
```
The same with `--board esp32` (TFLite: flash only). Results land in `results/`; everything generated in `runs/`
(`../clean.py` frees the build folders without touching any result). Measured in October 2026: ESP32-D0WD-V3 and
ESP32-S3, 240 MHz, weights in flash unless stated, the minimum-RAM counting mode (`popcount = "swar"`, no lookup
table in RAM).
