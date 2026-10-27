# mlp_fashion -- binary MLP vs. INT8, INT4 and INT2 on the ESP32-S3 (Fashion-MNIST)

**Sized for binary, a SYMBEX MLP (784-128-512-10) runs in 0.42 ms on the ESP32-S3 with its weights in flash: 28x
faster than the official INT8 model of the same network (TensorFlow Lite Micro, 11.7 ms), with 27 KB of flash
instead of 187 KB and 4.6 KB of RAM instead of 6.7 KB.** With the weights in RAM it is 2.1x faster (0.41 vs. 0.89 ms).
It classifies Fashion-MNIST at 87.5 %; INT8 of the same network, at 89.8 %. Every result on the chip is identical,
image by image, to the integer simulation of its method.

What this benchmark shows, in short:

- **Speed and memory: SYMBEX wins.** With the weights in flash (how a model is usually deployed), every binary model is
  faster than INT8, INT4 and INT2, and needs less flash.
- **Accuracy per byte: low-bit integers win.** INT2 QAT is 1.0-1.6 points more accurate than a binary model of about
  the same size. INT8 QAT is 1.7-2.4 points above the binary ones (labels only).
- **Where the width goes matters more than how much there is.** A narrow first layer and wide binary layers give almost
  the accuracy of the benchmark's x2 model, 13x faster.

## Setup

**Data.** Fashion-MNIST. The 60,000 training images are split once into 54,000 for training and 6,000 for validation
(`[data] validation_split = 0.1`, `split_seed = 42` in `symbex.toml`). The split is the same for every seed and every
method. Every choice is made on validation; the 10,000 test images are measured once.

**FP32 model** (`model.py`): 784 -> 256 -> 128 -> 10, hidden layers Linear -> BatchNorm1d -> ReLU, 30 epochs of Adam
(lr 1e-3, seed 42), trained on the 54,000 training images only. Validation 89.00 %, **test 88.51 %**. Every method
starts from this same file (`models/fashion_mlp.pth`, sha256 `d74f32d30cbf661e...`, see *How to reproduce*).

**Methods** (all 15 epochs, seeds 42, 43 and 44, each with two recipes: distillation from the FP32 model, and labels
only):

| Method | Weights | Activations | Tool |
|---|---|---|---|
| SYMBEX 1 bit | 1 bit | 1 bit (the first layer takes the 8-bit pixels) | the CDT, method `plain` |
| ternary | -1 / 0 / +1 | 1 bit | the CDT, method `ternary` |
| INT8, INT4, INT2 QAT | 8 / 4 / 2 bits, per channel, symmetric (INT2: levels -1, 0, +1) | 8 bits | `earlier/fashion_int4.py`; `fashion_lowbit.py` (Brevitas, `tools/lowbit`) |
| INT8 TFLite | 8 bits, full-integer post-training quantization | 8 bits | `fashion_tflite.py` (official converter, no retraining) |

**On the chip** (ESP32-S3, 240 MHz, one core): every SYMBEX, ternary, INT4 and INT2 model runs through the same tool,
`tools/chip_test.py`: the same firmware, the same 1,000 test images, the median latency, and the result checked
against the method's integer simulation (**100 % on every run**). The integer models of `fashion_lowbit.py` (INT8, INT4,
INT2) run with ordinary multiply-accumulate on Espressif's ESP-NN kernels (`tools/lowbit`); the earlier INT4 / INT2
rows ran as bit planes summed by the library's HAL (`earlier/int_export.py`), SYMBEX arithmetic, kept for the record. INT8 TFLite runs on TensorFlow Lite Micro with Espressif's ESP-NN kernels, checked against
the TFLite interpreter (100 %). Latency depends on the shape, not on the weight values: it is measured on the seed-42
export of each model.

## 1. Accuracy per byte (PC, 3 seeds)

`lowbit_study.py`. Network 784-256-128-10 (x1; SYMBEX also widened x1.5 and x2). Test: mean +- sd over 3 seeds on the
10,000 test images. *int 2000*: the integer model on the first 2,000 test images, the one measure every row has
(TFLite has no seeds).

| Model | Weights KB | Test, distillation | Test, labels only | int 2000 (labels only) |
|---|---|---|---|---|
| SYMBEX 1 bit x1 (784-256-128-10) | 28.7 | 85.69 +- 0.38 | 87.64 +- 0.36 | 88.00 |
| SYMBEX 1 bit x1.5 (784-384-192-10) | 46.0 | 86.77 +- 0.45 | 88.26 +- 0.13 | 88.88 |
| SYMBEX 1 bit x2 (784-512-256-10) | 65.3 | 87.02 +- 0.28 | 88.31 +- 0.20 | 89.10 |
| ternary x1 | 57.3 | 86.13 +- 1.46 | 87.93 +- 0.27 | 88.72 |
| ternary x1.5 | 92.0 | 86.80 +- 0.28 | 88.37 +- 0.12 | 88.93 |
| INT2 QAT x1 | 57.3 | 88.59 +- 0.04 | 89.26 +- 0.10 | 90.25 |
| INT4 QAT x1 | 114.6 | 88.70 +- 0.11 | 89.95 +- 0.08 | 91.07 |
| INT8 QAT x1 | 229.2 | 88.58 +- 0.02 | 90.04 +- 0.02 | 91.18 |
| INT8 TFLite x1 | 229.2 | -- | -- | 89.70 |

- **Per byte, the low-bit integers are more accurate.** INT2 QAT (57.3 KB) is 1 point above SYMBEX x2 (65.3 KB) with
  either recipe. With 8-bit activations, the integer models keep more information per layer than 1-bit activations.
- **Labels only beats distillation for every method here** (1.3-2.0 points for SYMBEX). The teacher (88.51 %) is not
  stronger than what the students reach alone, so it pulls them down; the rule is the same in `mlp_digits`.
- The binary x1.5 and x2 models trained on the labels are **within 0.3 points of the FP32 original** (88.26-88.31 %
  vs. 88.51 % on the same test images), with 1 bit everywhere after the first layer.

### Binary weights or binary activations? (ablation)

`fashion_lowbit.py --ablation`: the same network, recipe and seeds at every pair of weight bits (1, 2, 4, 8) and
activation bits (int8, binary), QAT with Brevitas from the FP32 model. Fake-quant accuracy on the PC: the mixed cells
(W4A1, W1A8...) have no kernel in either library, so this is an ablation, not a measurement on the chip. Test, mean +-
sd over 3 seeds:

| Weights | Labels only, int8 act. | Labels only, binary act. | Distillation, int8 act. | Distillation, binary act. |
|---|---|---|---|---|
| 8 bits | 89.86 +- 0.04 | 88.35 +- 0.15 | 88.61 +- 0.06 | 87.78 +- 0.25 |
| 4 bits | 89.81 +- 0.12 | 88.48 +- 0.15 | 88.64 +- 0.19 | 87.79 +- 0.16 |
| 2 bits | 89.22 +- 0.13 | 87.20 +- 0.25 | 88.19 +- 0.29 | 86.72 +- 0.09 |
| 1 bit | 89.30 +- 0.12 | **87.77 +- 0.41** (SYMBEX) | 88.45 +- 0.10 | 86.93 +- 0.39 |

- **The binary activations cost the accuracy, not the binary weights:** with int8 activations, 1-bit weights lose
  0.56 points against 8-bit weights (labels only); with 8-bit weights, binary activations lose 1.51.
- W1A1 trained with Brevitas (87.77 %) matches SYMBEX trained with the CDT (87.64 %): the ablation measures the same
  thing.
- Binary weights with int8 activations (W1A8, 89.30 %) are as accurate as INT2 (89.22 %) with half the weight bits.
  SYMBEX's first layer already computes uint8 inputs x 1-bit weights; a whole network of that kind is future work.

## 2. Speed on the ESP32-S3

`earlier/lowbit_chip.py`. Accuracy: the distillation recipe of the table above (the network and its speed are the same with
either recipe). RAM: model buffers, input, stack (and the weights, when they are in RAM), measured on the chip.
SYMBEX runs in the counting mode `swar` (`--popcount swar`): the AND layers count with PIE and need no lookup table;
the 64 KB table of the default `lut16` mode gives the same speed here (measured, within 1 %) and is not counted.

### Weights in flash (the usual deployment)

| Model | Test acc | Latency | vs. INT8 TFLite | Flash | RAM |
|---|---|---|---|---|---|
| **SYMBEX 1 bit x1** | 85.69 | **1.237 ms** | **12.8x** | **34.3 KB** | **3.7 KB** |
| SYMBEX 1 bit x1.5 | 86.77 | 3.854 ms | 4.12x | 54.5 KB | 4.2 KB |
| SYMBEX 1 bit x2 | 87.02 | 5.389 ms | 2.95x | 76.5 KB | 4.7 KB |
| ternary x1 | 86.13 | 14.589 ms | 1.09x | 59.9 KB | 66.5 KB (with a 64 KB count table) |
| INT8 QAT x1, ESP-NN | 88.61 | 15.601 ms | 1.02x | 233.9 KB | 3.1 KB |
| INT4 QAT x1, ESP-NN | 88.65 | 10.306 ms | 1.54x | 119.2 KB | 11.1 KB |
| INT2 QAT x1, ESP-NN | 88.20 | 13.376 ms | 1.19x | 61.9 KB | 11.1 KB |
| *INT2 QAT x1, bit planes (earlier)* | 88.59 | 5.576 ms | 2.85x | 68.9 KB | 10.1 KB |
| *INT4 QAT x1, bit planes (earlier)* | 88.70 | 10.053 ms | 1.58x | 133.2 KB | 10.1 KB |
| INT8 TFLite x1 | 89.70 (2,000) | 15.885 ms | 1.00x | 242.2 KB | 4.8 KB (arena) |

### Weights in RAM (the fastest case, when the model fits)

| Model | Latency | vs. INT8 TFLite | RAM |
|---|---|---|---|
| **SYMBEX 1 bit x1** | **0.525 ms** | **1.67x** | **38.0 KB** |
| SYMBEX 1 bit x1.5 | 0.773 ms | 1.13x | 58.6 KB |
| SYMBEX 1 bit x2 | 1.034 ms | 0.85x | 81.1 KB |
| ternary x1 | 11.936 ms | 0.07x | 126.4 KB (with a 64 KB count table) |
| INT8 QAT x1, ESP-NN | 0.556 ms | 1.58x | 242.6 KB |
| INT4 QAT x1, ESP-NN (unpacked to int8 at the first call) | 0.556 ms | 1.58x | 240.3 KB |
| INT2 QAT x1, ESP-NN (unpacked to int8 at the first call) | 0.556 ms | 1.58x | 240.3 KB |
| *INT2 QAT x1, bit planes (earlier)* | 1.832 ms | 0.48x | 79.1 KB |
| *INT4 QAT x1, bit planes (earlier)* | 2.755 ms | 0.32x | 143.4 KB |
| INT8 TFLite x1 (model copied to RAM) | 0.876 ms | 1.00x | about 247 KB |

- **In flash, SYMBEX is the fastest at every size**, and the gap is large: the binary weights are 8x fewer bytes to
  read from flash than INT8.
- **In RAM, the INT8 kernels of ESP-NN are very fast**: 0.556 ms called directly (0.876 ms through TensorFlow Lite
  Micro, its interpreter included). SYMBEX x1 is still the fastest (0.525 ms, 1.06x the direct ESP-NN call) with 6x
  less RAM; x1.5 and x2 are slower. INT4 and INT2 unpacked to int8 run exactly as INT8: the bits only save flash.
- **In flash, fewer weight bits help INT less than expected:** INT4 / INT2 codes are unpacked to int8 before each
  kernel call (10.3 / 13.4 ms against 15.6 ms for INT8); SYMBEX x1 stays 8.3x faster than the best of them.
- The earlier INT4 / INT2 rows (bit planes, `earlier/int_export.py`) used the library's own counting, not integer
  arithmetic: they are kept for the record, the ESP-NN rows are the integer baseline.
- **The ternary kernel is slow** (12-15 ms): it is the next kernel to rewrite (see *Future work*).

### The classic ESP32 (no SIMD, no popcount)

The same seed-42 models on the ESP32 (240 MHz, one core), the current library and `tools/lowbit` (board = simulation,
100 % on every run):

| Model | Weights | Flash | RAM |
|---|---|---|---|
| **SYMBEX 1 bit x1** (`--popcount swar`) | 28.7 KB | **6.754 ms** | **3.840 ms** (38.6 KB of RAM) |
| INT2 QAT x1, ESP-NN | 61.9 KB | 30.352 ms | does not start (likely: no 235 KB block for the unpacked weights) |
| INT4 QAT x1, ESP-NN | 119.2 KB | 34.610 ms | does not start (same) |
| INT8 QAT x1, ESP-NN | 233.9 KB | 41.377 ms | does not fit (DRAM overflowed by 74 KB) |

- In flash, SYMBEX x1 is 4.5x faster than the fastest integer model (INT2); with the weights in RAM it is the only
  one that runs.

### How the ESP32-S3 computes a binary layer

The S3 has no popcount instruction, but its PIE SIMD unit has an 8-bit multiply-accumulate: `ee.vmulas.u8.accx`,
16 pairs per instruction into a 40-bit accumulator -- the same unit ESP-NN uses for INT8. With weights +-1,
`sum x_i s_i = 2 * sum_{s_i=+1} x_i - sum x_i`, so a layer only needs the sum of the inputs whose weight bit is 1:

- **First layer (uint8 pixels x 1-bit weights):** per 16 inputs, `ee.vldbc.16` broadcasts 16 weight bits,
  `ee.andq` + `ee.vcmp.eq` turn each bit into a 0xFF / 0x00 byte, and `ee.vmulas.u8` multiplies the pixels by it:
  the accumulator holds 255 x the sum (divided back exactly).
- **Binary layers (1 bit x 1 bit):** `ee.andq` / `ee.xorq` on 128 bits, a SWAR bit count per byte, then one
  `ee.vmulas.u8` with a vector of ones adds the 16 byte counts.

Per instruction, the first layer does as many multiply-accumulates as INT8 (16) plus three to expand the bits, which
is why the two are close with everything in RAM. The binary model's advantage is memory: 8x fewer weight bytes.

## 3. Sizing the MLP for binary

`fashion_sizing.py`. On the chip the **first layer is the expensive one**: it sums 8-bit pixels, while the hidden
layers are binary. The benchmark's x2 model (784-512-256-10) doubles that first layer. The same width can go to the
second, binary layer instead. Every shape trains its own FP32 model with the same recipe, then the CDT binarizes it
(every layer 1 bit, 15 epochs). Shapes are ranked on validation; test is measured once. Latency is **measured** on the
S3 (seed-42 export, 1,000 test images, 100 % equal to the simulation).

| Shape | Weights KB | Val, labels only | Test, labels only | Test, distillation | ms flash | ms RAM |
|---|---|---|---|---|---|---|
| **784-128-512-10** | **20.9** | 87.88 +- 0.31 | **87.47 +- 0.09** | 86.83 +- 0.14 | **0.420** | **0.414** |
| 784-256-512-10 | 41.1 | 88.60 +- 0.33 | 88.20 +- 0.29 | 88.15 +- 0.23 | 3.529 | 0.673 |
| 784-384-512-10 | 61.4 | 88.69 +- 0.44 | 88.72 +- 0.09 | 88.20 +- 0.14 | 5.104 | 0.933 |
| 784-512-256-10 (the x2 model) | 65.3 | 88.71 +- 0.17 | 88.67 +- 0.13 | 87.16 +- 0.70 | 5.373 | 1.037 |

- **784-128-512-10 is 13x faster than the x2 model with the weights in flash, for 1.2 points.** It needs 27.3 KB of
  flash and 4.6 KB of RAM (counting mode `swar`, no lookup table).

**Against INT8 of the same network** (784-128-512-10, its own FP32 model: test 88.92 %; the same recipe, 15 epochs,
seeds 42-44; on the chip, the official TFLite Micro model, 100 % equal to the PC interpreter):

| 784-128-512-10 | Test, labels only | Test, distillation | Flash | ms flash | ms RAM | RAM |
|---|---|---|---|---|---|---|
| INT8 QAT | 89.81 +- 0.11 | 89.06 +- 0.02 | -- | -- | -- | -- |
| INT8 TFLite Micro | -- (89.60 on the first 2,000) | -- | 187.0 KB | 11.671 | 0.887 | 6.7 KB (arena) |
| **SYMBEX 1 bit** | 87.47 +- 0.09 | 86.83 +- 0.14 | **27.3 KB** | **0.420 (27.8x)** | **0.413 (2.15x)** | **4.6 KB** |

The binary network costs 2.3 points against INT8 of the same shape, and is 28x faster with the weights in flash,
2.1x in RAM, with 6.8x less flash.
- **784-384-512-10 matches the x2 model's accuracy** (88.72 vs. 88.67 %) and is a little faster and smaller.
- **The flash cost is not linear in the number of weights.** The first layer of 784-128-512 (12 KB of weights) runs
  from flash at the same speed as from RAM (0.273 vs. 0.270 ms); the one of 784-256-512 (25 KB) is 4.4x slower from
  flash (2.070 vs. 0.471 ms). The most likely cause is the flash cache (not verified). That is why the sizing table
  measures every shape instead of predicting it from a per-weight cost: the prediction gave 1.71 ms for 784-128-512.
- Both recipes agree on the ranking except at x2, where distillation is unstable (sd 0.70).

## Earlier results (previous FP32 model and library)

These were measured before the fixed validation split, with the earlier FP32 model (5 epochs, 88.17 % on test,
trained on all 60,000 training images) and an earlier version of the library. They are kept because they cover what
the sections above do not (a 1-bit input, the classic ESP32); to be measured again (*Future work*).

**Fully binary, 1-bit input.** Every pixel becomes one bit (its most significant bit); labels only, 3 seeds.
x1.5 (784-384-192-10): **83.28 +- 0.26 %**, 49.1 KB of flash; x4 (784-1024-512-10): 84.12 +- 0.15 %, 185 KB. Fed the
same 1-bit pixels, the FP32 original reaches 77.88 %. The cut is one fixed threshold (128) per pixel; in the
`experiments` folder a fully binary network with learned input thresholds reaches 86.3 % on this task.

**8-bit pixels in the first layer, x1.5, 3 seeds:** labels only **88.50 +- 0.09 %** (above that FP32 model, 88.17 %);
distilled from the FP32 model 86.94 +- 0.13 % (87.03 +- 0.32 % on the 1,000 images sent to the chip). INT8 TFLite on
the same 1,000 images: 87.20 %.

On the chip (x1.5, 1,000 images, 100 % equal to the simulation; counting mode `swar`):

| Board | Model | Weights | Latency | vs. TFLite | Flash | RAM |
|---|---|---|---|---|---|---|
| ESP32 | INT8, TFLite Micro | flash | 41.97 ms | 1.00x | 242.2 KB | 4.8 KB |
| ESP32 | SYMBEX, fully binary, XNOR | flash | 8.97 ms | 4.68x | 49.1 KB | 1.5 KB |
| ESP32 | SYMBEX, 8-bit input, XNOR | flash | 12.14 ms | 3.46x | 49.1 KB | 1.4 KB |
| ESP32 | SYMBEX, 8-bit input, AND | flash | 12.95 ms | 3.24x | 54.5 KB | 3.8 KB |
| ESP32 | INT8, TFLite Micro | RAM | does not fit (242 KB of weights) | -- | -- | -- |
| ESP32 | SYMBEX, 8-bit input, AND | RAM | 5.81 ms | -- | -- | 58.3 KB |
| ESP32-S3 | INT8, TFLite Micro | flash | 15.89 ms | 1.00x | 242.2 KB | 4.8 KB |
| ESP32-S3 | SYMBEX, fully binary, XNOR | flash | 3.75 ms | 4.24x | 49.1 KB | 1.7 KB |
| ESP32-S3 | SYMBEX, 8-bit input, AND | flash | 3.87 ms | 4.11x | 54.5 KB | 4.1 KB |
| ESP32-S3 | SYMBEX, 8-bit input, XNOR | flash | 6.22 ms | 2.55x | 49.1 KB | 1.6 KB |

- **Each chip prefers a different arithmetic, and the CDT measures it instead of assuming it**
  (`tools/chip_test.py --kernels` builds a profile per chip; `[output] arithmetic = "auto"` predicts the cost of every
  layer from it). On the S3 the AND arithmetic lets the multiplier do the counting (1.6x faster than XNOR on this
  model). The classic ESP32 has neither popcount nor SIMD: there XNOR wins by 7 %, with rows that need no padding
  (49.1 vs. 54.5 KB).
- **The first layer dominates** (86-89 % of the time). On the ESP32 half of that time is reading 37.6 KB of
  first-layer weights from flash (they do not fit its cache): with the weights in RAM the same model runs in 5.8 ms
  instead of 12.1 ms.
- With everything in RAM, the classic ESP32 runs the SYMBEX model in 5.8 ms; the INT8 model does not fit there at all.

### PopLUT probe: codebook-compressed binary weights (PC only)

`../../experiments/poplut_probe.py` tests whether the 1-bit weights of a trained SYMBEX x2 model can be stored as indices into a small
codebook of 32-bit words (each word: a codebook index plus a complement bit). The codebook is learned by binary
k-means in Hamming distance, then the rest of the model is fine-tuned for 3 epochs. Base: SYMBEX x2, seed 42, 86.34 %,
66,880 B of 1-bit weights.

| Codebook K | Post hoc | After fine-tune | Weights |
|---|---|---|---|
| 16 | 34.23 % | 77.30 % | 11,008 B |
| 64 | 51.83 % | 81.02 % | 15,616 B |
| 256 | 73.37 % | 84.04 % | 21,376 B |

Every weight row of `red.1` and `red.4` is cut into 32-bit words; `red.7` stays plain 1 bit. At K = 256 the
compressed layers cost about 0.31-0.34 bits per weight, with 19-20 % of the weight bits changed (Hamming). The
criterion fixed before running (K = 256 losing more than 1 point discards it) was met: it loses 2.30 points, so
PopLUT is **discarded** as a direct replacement.

As a size point it is still of some interest: the binary x1 model (784-256-128-10, 15 epochs, validation held out of
training) reached 82.85 % on test with about 29.3 KB of 1-bit weights (`results/earlier/symbex_x1_e15_train.json`); PopLUT
K = 256 on x2 is 1.19 points more accurate and about 27 % smaller. INT2 QAT still dominates in accuracy. Caveats: the
x2 model was trained before the validation fix (last checkpoint, no selection); a single k-means seed; not tested on
the chip; the C lookup kernel does not exist yet.

```
python ../../experiments/poplut_probe.py --modelo runs/symbex_local_s42/model_binarized.pth
```

## How to reproduce

| Script | What it does |
|---|---|
| `model.py` | the FP32 model and the data both sides share |
| `fashion_symbex.py` / `fashion_tflite.py` | the SYMBEX side (CDT) / the official INT8 side (TFLite) |
| `fashion_sizing.py` | the sizing study: which shape for a binary network |
| `fashion_lowbit.py` | INT8 / INT4 / INT2 with `tools/lowbit` (Brevitas, ESP-NN), the weights-vs-activations ablation |
| `lowbit_study.py` | the SYMBEX x1 / x1.5 / x2 and ternary runs of the low-bit study (and its earlier INT rows) |
| `earlier/` | the earlier INT baselines (`fashion_int4.py`, `int_export.py`: bit planes), their chip table (`lowbit_chip.py`) and the Pareto plot (`pareto.py`): kept so the earlier rows can be reproduced |

From `benchmarks/mlp/mlp_fashion/`, environment `env_bench`.

**Option A -- the same FP32 models (recommended).** The FP32 starting models are published in `models/` (the only
models in the repository). `model.get_fp32` uses them when present; every result file records the sha256 of the
model it started from (`fp32_sha256`).

| File | Shape | sha256 |
|---|---|---|
| `fashion_mlp.pth` | 784-256-128-10 | `d74f32d30cbf661e2937e46acbe15cd2a1ecbb2a2f57117b8c8bff53abb560d1` |
| `fashion_mlp_128_512.pth` | 784-128-512-10 | `86a2c40513ed4f7f530b72aac9ac8dff12acc9ca9f7142615a26ba98e04ad41a` |
| `fashion_mlp_256_512.pth` | 784-256-512-10 | `e7504e4423d7049ba649f43164bed67cd00fd4d4e59a4db0cd2425df201ca4d2` |
| `fashion_mlp_384_512.pth` | 784-384-512-10 | `fee62a7ab3c33c2720d9123df9e075de17d2d8b080f7d580b9200fcb55dfe56f` |
| `fashion_mlp_512_256.pth` | 784-512-256-10 | `2d87542149f544846e17f4ce64c79f05c2abf48d4f829c1f3ae28d5ae5a626bc` |

**Option B -- everything from scratch.** Delete `models/`: the scripts retrain each FP32 model (30 epochs, about a
minute each). GPU training is not bit-identical across machines, so the FP32 sha256 will differ and the results are
expected to move within the seed noise.

```
sha256sum models/fashion_mlp*.pth                    # option A: compare with the table

# PC: the FP32 model, TFLite INT8, the low-bit study (both recipes), the sizing study (both recipes)
python fashion_tflite.py
python lowbit_study.py --seeds 3
python lowbit_study.py --seeds 3 --no-distill
python fashion_lowbit.py                             # INT8 / INT4 / INT2 with Brevitas (tools/lowbit), 3 seeds
python fashion_lowbit.py --no-distill
python fashion_sizing.py --shapes 128-512 256-512 384-512 512-256 --seeds 3
python fashion_sizing.py --shapes 128-512 256-512 384-512 512-256 --seeds 3 --no-distill

# ESP32-S3: on its native USB port (/dev/ttyACM0, --usb) or on its UART (/dev/ttyUSB0, without --usb)
python earlier/lowbit_chip.py --board s3 --port /dev/ttyACM0 --usb                     # weights in flash
python earlier/lowbit_chip.py --board s3 --port /dev/ttyACM0 --usb --weights ram
python fashion_lowbit.py --chip-only --board s3 --port /dev/ttyACM0 --usb    # INT8 / INT4 / INT2 on ESP-NN, flash
python fashion_lowbit.py --chip-only --board s3 --port /dev/ttyACM0 --usb --weights ram
python fashion_sizing.py --shapes 128-512 256-512 384-512 512-256 --seeds 3 --measure --board s3 --port /dev/ttyACM0 --usb
python fashion_tflite.py --board s3                                            # INT8 in flash
python fashion_tflite.py --board s3 --model-ram                                # INT8 in RAM
python earlier/fashion_int4.py --bits 8 --epocas 15 --semillas 3 --shape 128-512       # INT8 QAT of the binary-sized MLP
python earlier/fashion_int4.py --bits 8 --epocas 15 --semillas 3 --shape 128-512 --no-distill
python fashion_tflite.py --shape 128-512 --board s3 --port /dev/ttyACM0 --usb  # its TFLite INT8, flash
python fashion_tflite.py --shape 128-512 --board s3 --port /dev/ttyACM0 --usb --model-ram
python ../../../tools/chip_test.py runs/sizing/h128_512_s42 --board s3 --port /dev/ttyACM0 --usb --popcount swar  # minimum RAM
```

The earlier results (1-bit input, the classic ESP32, the INT baselines of `earlier/fashion_int4.py`, the Pareto plot):

```
python ../../../tools/chip_test.py --kernels --board s3 --popcount swar          # chip profile (once per chip)
python fashion_symbex.py --seeds 3 --input-bits 1 --no-distill --widths 1.5   # fully binary (PC)
python fashion_symbex.py --seeds 3 --input-bits 1 --no-distill --widths 1.5 --board esp32 --weights flash --compare-arithmetic
python fashion_symbex.py --seeds 3 --no-distill                               # 8-bit input, labels only (PC)
python fashion_symbex.py --seeds 3 --board s3 --weights flash --compare-arithmetic
python fashion_symbex.py --seeds 3 --board s3 --weights ram --compare-arithmetic
python fashion_tflite.py --board esp32                                        # TFLite: flash only on the ESP32
python fashion_tflite.py --board s3 --table-only                              # both TFLite tables
python earlier/fashion_int4.py --bits 4 2 --semillas 3 --epocas 15                    # INT PTQ + QAT (PC)
python earlier/pareto.py                                                              # accuracy vs. weight bytes
```

Results land in `results/`, one folder per script: `symbex/` (fashion_symbex.py), `tflite/` (fashion_tflite.py),
`lowbit/` (fashion_lowbit.py, lowbit_study.py and the earlier INT scripts), `sizing/` (fashion_sizing.py), and
`earlier/`: the results from before the validation fix or the current library (the *Earlier results* above), kept
for the record. Everything generated goes to `runs/` (`../clean.py` frees the build folders without
touching any result). Measured in October 2026 on an ESP32-S3 (QFN56, rev v0.2), 240 MHz.

## Limitations

- One task; one FP32 model per shape. SYMBEX and the QAT methods with 3 seeds; INT8 TFLite has no training, so no
  seeds.
- The integer models run their requantization with a float multiplier in the simulation (`earlier/fashion_int4.py`); on the
  chip, INT4 / INT2 are checked against their own integer simulation (100 %).
- One board for the current results (ESP32-S3). Energy has not been measured; latency and memory have.

## Future work

- **INT4 / INT2 at the binary-sized shape** (`fashion_lowbit.py --shape 128-512 --bits 4 2`), and on the chip.
- **The classic ESP32 and the 1-bit input** again, with the current library and the fixed validation split.
- **The ternary kernel**: one count per word with HAL row functions (today 12-15 ms).
- **Learned input thresholds** for the fully binary network (86.3 % in `experiments`).
- **Energy**, with a current meter: the metric no benchmark has yet.
