# har_ultracompact — binary 1D convolutions for human activity recognition

Human activity recognition from a phone's accelerometer (UniMiB-SHAR, 17 activities) with **binary 1D convolutional
networks**. It compares two binary libraries first: SYMBEX and the closest prior work, the binary inference library of
**Daghero et al., "Ultra-compact Binary Neural Networks for Human Activity Recognition on RISC-V Processors"** (ACM
Computing Frontiers 2021, `github.com/francescodaghero/ultracompactBNN`). The comparison uses their four networks,
their processor and their own code. INT8 is the reference: TensorFlow Lite Micro and the INT8 / INT4 / INT2 baselines
of `tools/lowbit`, with the same architectures, measured on real boards.

The README follows the order of the work:

1. **Binary vs binary**: SYMBEX and ultracompactBNN on their processor.
2. **Binary vs INT8**: SYMBEX and TFLite Micro on real boards.
3. **Where fully binary stops paying (a working theory)**: the first layer with int8 weights, an unexpected gain
   found while running their network in SYMBEX, and what it suggests about when binary weights pay off (still to be
   tested beyond this benchmark).

## Results in short

- **Binary vs binary, in their own environment** (RI5CY core in GVSoC, their code unchanged): SYMBEX needs **1.07-1.37×
  fewer instructions** than ultracompactBNN on all four networks of their Table 1 (3 seeds). This holds with **their
  own trained network** too (same int8 first layer, same weights, same accuracy): **1.06-1.37×**. Every output is exact:
  equal to the integer simulation, window by window.
- **Against INT8 on real boards** (TFLite Micro, the same architectures, weights in flash), all binary: SYMBEX is
  **5.3-10.4× faster** on the ESP32 and **1.44-1.86× faster** on the ESP32-S3. It takes **2-10× less RAM** and
  **6.5-10× less flash**. INT8 is more accurate on these small networks: about 9 points on the large ones, 23 on the
  smallest.
- **Where fully binary stops paying: a working theory.** Binary network work usually keeps the first layer in higher
  precision because it is a small share of the work. SYMBEX's measurements suggest the reason and the boundary: the
  gain of binary weights depends on the **data**. A binary input makes the first layer one more binary layer (32
  products per bit count), where fully binary should win. An 8-bit input (here the accelerometer) costs about one
  operation per input whether the weights have 1 bit or 8. There the int8 first layer is **faster than the fully
  binary one** on every board measured (equal on the M33's largest network), because the chips multiply bytes faster
  than they count bit planes (on the ESP32-S3, a new PIE kernel: **1.8-2.4×** for that layer, **1.25-1.66×** for the
  whole network). It is also **more accurate**: **+4.5-4.8 points** on the two small networks, equal on the large
  ones. Measured on one task (activity recognition) and checked quickly on two others: a theory to test further, not a
  general rule.
- **A third factor: the size of the layer.** In a quick test on the ESP32-S3, mlp_fashion's 784 → 128 first layer
  stays faster with 1-bit weights, because its int8 weights (100 KB) do not fit the flash cache (section 3).
- **Exact on every board:** ESP32, ESP32-S3 and both cores of the Raspberry Pi Pico 2 (Hazard3 RISC-V, Cortex-M33),
  1,000 windows per run, 100 % equal to the simulation in every run.

## 1. Binary vs binary: SYMBEX and ultracompactBNN on their processor

### The setup

| | ultracompactBNN | SYMBEX |
|---|---|---|
| Processor | RI5CY (PULP), simulated with GVSoC (pulp-open) | the same |
| Compiler | PULP GCC, `-O3 -flto` | the same |
| Bit count | `p.cnt` (hardware) | the same instruction |
| First layer | int8 input × int8 weights (`sdotsp4`) | uint8 input × 1-bit weights (`p.sdotusp4`), or their int8 weights |
| Other layers | their 1-bit kernels, fused max pool | `symbex_bool1d`: XOR + bit count, fused OR pool |
| Networks | Table 1 of their paper, UniMiB-SHAR: min, max-10%, max-5%, max | the same |

The main measure is the **instruction count**. Their chip, PULPissimo, is no longer a buildable target in GVSoC, and
the memory of the pulp-open platform is slower for the core (about 2.2 cycles per instruction for both libraries).
Instructions do not depend on the simulated memory; cycles are reported too.

### Whole networks (instructions, mean of 3 seeds; > 1: SYMBEX does less work)

| Network (Table 1) | ultracompactBNN | SYMBEX, all binary | ratio | SYMBEX, their network (int8 first layer) | ratio |
|---|---|---|---|---|---|
| min | 19 559 | 18 201 | **1.07×** | 18 530 | **1.06×** |
| max-10% | 183 372 | 133 498 | **1.37×** | 133 398 | **1.37×** |
| max-5% | 473 697 | 385 094 | **1.23×** | 385 262 | **1.23×** |
| max | 499 645 | 405 104 | **1.23×** | 405 215 | **1.23×** |

In cycles: 1.26-1.38× on the three large networks; on min, 0.94-0.95× (their code takes fewer cycles there).

Per layer (max-10%, seed 42): first layer 1.07×, binary conv 8→32 k7 + pool 1.59×, binary conv 32→32 k7 + pool 1.14×,
classifier 0.64× (about 1,000 instructions, under 1 % of the network). The binary layers are where SYMBEX gains; the
first layer costs about the same in both libraries, with 1-bit or int8 weights (section 3 explains why).

### Accuracy (test subjects, integer simulation = what the chip runs; mean ± std, 3 seeds)

| Network | SYMBEX, all binary | ultracompactBNN (int8 first layer) | Daghero et al. published* |
|---|---|---|---|
| min | 32.7 ± 2.7 % | 37.5 ± 1.6 % | 33.0 % |
| max-10% | 53.4 ± 1.9 % | 57.9 ± 0.3 % | 58.6 % |
| max-5% | 56.8 ± 0.7 % | 57.0 ± 1.2 % | 65.1 % |
| max | 58.8 ± 0.9 % | 58.5 ± 2.6 % | 68.0 % |

\*Their split is not published, so their numbers are not directly comparable. SYMBEX with their network (int8 first
layer) has their accuracy by construction: it is the same network, run exactly.

## 2. Binary vs INT8 on real boards

### Accuracy of the integer baselines (test, mean ± std)

| Network | SYMBEX (1 bit) | TFLite INT8 | INT8 QAT | INT4 QAT | INT2 QAT | FP32 |
|---|---|---|---|---|---|---|
| min | 32.7 ± 2.7 % | 56.1 ± 4.2 % | 51.3 ± 0.1 % | 51.0 ± 1.2 % | 35.7 ± 0.3 % | 50.6 % |
| max-10% | 53.4 ± 1.9 % | 62.4 ± 2.0 % | 59.9 ± 0.2 % | 59.5 ± 1.0 % | 51.8 ± 3.2 % | 59.8 % |
| max-5% | 56.8 ± 0.7 % | 66.2 ± 2.6 % | 69.1 ± 0.3 % | 69.6 ± 0.1 % | 65.4 ± 0.9 % | 68.6 % |
| max | 58.8 ± 0.9 % | 67.9 ± 0.9 % | 68.0 ± 0.1 % | 67.6 ± 0.3 % | 66.5 ± 1.2 % | 67.6 % |

TFLite INT8: three float networks (seeds 42-44), each quantized after training. QAT (`tools/lowbit`, Brevitas): the
seed-42 float network, quantization-aware training with 3 seeds. As in mlp_fashion, **INT2 QAT is more accurate per byte
than SYMBEX** on the large networks (+8 points with about 1.5× the weight bytes); SYMBEX's case is speed and memory.

### Latency, weights in flash (median, ms; 1,000 validation windows, mean of 3 seeds)

The all-binary network runs its first layer in one of two forms, with the same weights, thresholds and outputs:
"bits" (the 8 bit planes of the input, built once per inference, counted against the 1-bit rows) or "±1 bytes" (the
first-layer weights stored as +1 / −1 bytes and run by the int8 kernel, `har_export.py --first-bytes`).

| Network | ESP32: first layer in bits | ±1 bytes | TFLite INT8 | SYMBEX vs TFLite |
|---|---|---|---|---|
| min | 1.53 | **0.78** | 4.17 | **5.3×** |
| max-10% | 4.60 | **4.01** | 31.23 | **7.8×** |
| max-5% | 14.71 | **10.99** | 114.22 | **10.4×** |
| max | 14.77 | **14.08** | 115.62 | **8.2×** |

| Network | ESP32-S3: first layer in bits | ±1 bytes | TFLite INT8 | SYMBEX vs TFLite |
|---|---|---|---|---|
| min | 0.83 | **0.50** | 0.72 | **1.44×** |
| max-10% | 2.36 | **1.84** | 2.93 | **1.59×** |
| max-5% | 4.87 | **3.85** | 7.14 | **1.86×** |
| max | 4.83 | **3.87** | 7.11 | **1.84×** |

| Network | Pico 2 Hazard3: bits | ±1 bytes | Pico 2 M33: bits | ±1 bytes |
|---|---|---|---|---|
| min | 1.31 | **0.57** | 1.22 | **0.82** |
| max-10% | 3.59 | **2.56** | 4.09 | **3.64** |
| max-5% | 8.26 | **5.99** | 10.48 | **9.89** |
| max | 10.67 | **8.59** | **11.72** | 11.79 |

Every run in these tables is 100 % equal to the simulation; the spread between the 3 seeds is at most 0.03 ms (on the
S3 at most 0.004 ms). The S3's ±1 bytes column uses the PIE kernel of section 3; before it, the bits were faster there
(max: 4.83 vs 8.70 ms).

SYMBEX counting modes: `swar` on the ESP32 and the ESP32-S3 (no lookup table: 1.3-2.8 KB of RAM; `lut16` is up to 8 %
faster on the ESP32, for 64 KB more, and gives the same speed on the S3), native `cpop` on the Hazard3, `lut16` on the M33 (`swar`:
1.25 / 4.92 / 14.49 / 15.49 ms all binary with bits, 2.2-3.4 KB of RAM instead of 67 KB). TFLite Micro on the Pico 2 is future work.

### Latency, weights in RAM (median, ms)

| Network | ESP32: all binary (lut16) | int8 first layer (lut16) | TFLite INT8 | S3: all binary, bits | ±1 bytes | int8 first layer | TFLite INT8 |
|---|---|---|---|---|---|---|---|
| min | 1.58 | 0.79 | 3.74 | 0.83 | **0.50** | **0.50** | 0.71 |
| max-10% | 4.43 | 3.74 | 28.72 | 2.36 | **1.84** | **1.84** | 2.93 |
| max-5% | 13.28 | 9.33 | 100.97 | 4.87 | **3.85** | **3.85** | 6.40 |
| max | 13.40 | 12.30 | 102.70 | 4.83 | **3.87** | **3.87** | 6.16 |

TFLite with the weights in RAM copies the whole model at boot. On the S3, SYMBEX takes the same time with the weights in
flash or in RAM (its weights fit the cache: 0.7-6.5 KB). On the Pico 2, the weights in RAM change SYMBEX's latency by
under 10 %.

### Memory (max network)

| | SYMBEX all binary, bits | first layer as ±1 bytes | SYMBEX int8 first layer | TFLite INT8 |
|---|---|---|---|---|
| Flash (weights + thresholds) | **5.6 KB** | 6.5 KB | 6.5 KB | 42.5 KB |
| RAM (no lookup table) | **2.4-2.7 KB** | 2.4-2.7 KB | 2.4-2.7 KB | 10.3 KB (ESP32) / 25.5 KB (S3) of arena |

The ±1 bytes cost little flash: the 1-bit rows are padded to 4 words (128 bits) for the 45 weights of a row, so bytes
take 2.8× that layer's rows, not 8×. Smallest network (min): SYMBEX 0.7 KB of flash (bits or bytes), 1.3-1.6 KB of
RAM; TFLite 7.3 KB of flash, 2.7-2.9 KB of arena.

### Notes

- **The ESP32-S3 has SIMD on both sides:** TFLite runs on ESP-NN, which uses the S3's PIE vector unit for its int8
  products; SYMBEX uses it to count bits and, since section 3, for the first layer too. The classic ESP32 has no SIMD
  for either: there SYMBEX is 5.3-10.4× faster.
- **The first layer decides the whole network:** the binary layers after it take the same time in every variant
  (Hazard3, max: 2.81 vs 2.86 ms; S3, max: 2.65 ms with bits, bytes or int8), so the differences between the columns
  are the first layer alone. All binary with bits, it is 32-92 % of the network.
- **The first layer was 1.1-4.4× faster in this version** (bit planes built once for the whole input instead of per
  step, no padding of the window to 128 inputs except for the S3's PIE, a running sum): Hazard3 min 5.45 → 1.31 ms,
  M33 2.13 → 1.22 ms, ESP32 2.37 → 1.53 ms; the S3 already had its PIE path (0.84 → 0.83 ms).

## 3. Where fully binary stops paying: the first layer (a working theory)

### How it came up

To compare with ultracompactBNN on equal terms, SYMBEX also had to run **their** network, which keeps the first layer
in int8 (section 1). On the boards that layer first ran on a generic C kernel. Writing a PIE kernel for it on the
ESP32-S3 (`symbex_s8_dot_rows`: 16 products per `ee.vmulas.s8.accx`, rows at any alignment so the weights stay in
flash) made the first layer **1.8-2.4× faster than SYMBEX's own bit planes**, which had been the fastest form on the
S3. The same kernel runs the all-binary network's first layer as ±1 bytes, at the same speed.

### What the earlier work says, and what this suggests (a working theory)

Most binary networks keep the first layer, and often the last, in higher precision. The BNN paper of Hubara et al.
(2016) points out that the first layer of an image network is small in parameters and in work. XNOR-Net (Rastegari et
al., 2016) keeps its first and last layers in full precision. Daghero et al. keep the first layer in int8. The usual
reading is that binarizing those layers **does not give a significant gain**.

SYMBEX started from the opposite bet: **everything binary**, the first layer included. These measurements suggest
where that bet holds and where it reaches its limit. Our working theory: the answer is neither "always" nor "never";
**it depends on the data, the hardware and the size of the layer.** It rests on one task measured in full (this one)
and a quick check of two others (below); more tasks and chips are needed to confirm it.

What the theory expects (the evidence so far in the last row):

| | Binary input (bits, or data that a few bits describe well) | 8-bit input (sensor values, pixels) |
|---|---|---|
| First layer | one more binary layer: 32 products per bit count | about one operation per input, 1-bit or 8-bit weights |
| Expected best form | **fully binary** | **int8 first layer**, where the chip has a fast multiply-accumulate and the weights fit its memory |
| Evidence so far | indirect: no benchmark here has a binary input yet; the binary layers, 1.14-1.59× fewer instructions than ultracompactBNN, 0.19 cycles per product (Hazard3) | the int8 first layer is faster than the bits on all four boards (M33, max: within 1 %) and more accurate (tables below) |

If the theory holds, the earlier practice is right for 8-bit data, for a reason it did not state: with an 8-bit
input, the first layer does not gain from 1-bit weights at all. And on chips that multiply bytes fast (here all four
boards, most of all the ESP32-S3's PIE), the int8 first layer **beats the fully binary network** in speed, not only in
accuracy. The last layer (the classifier) weighs little here: 0.009 ms of 3.87 ms (0.2 %) on the S3.

### The theory: binary weights pay off only where the input is binary too

In a binary layer both operands are bits: one XOR + bit count covers 32 products (0.19 cycles per product on the
Hazard3, max). The first layer is different: its input is 8-bit (the accelerometer). With 1-bit weights,

raw = Σ x·(±1) = 2·Σ_q 2^q·count(P_q AND w) − Σ x, over the 8 bit planes P_q of the input,

so each 32 inputs cost 8 × (AND + count + shift + add) per output channel: about **one operation per input**. A
multiply-accumulate also costs one operation per input (or 1/16 of one with SIMD), **whatever the width of the weight**.
So in the first layer, 1-bit weights save only memory, never time. Where a chip multiplies faster than it counts bits,
the bytes win; and once the layer runs on bytes, the weights can just as well be int8: same kernel, same time, more
accuracy. Memory is the one place 1-bit weights still pay: when the layer is large, its int8 weights can outgrow the
cache (mlp_fashion, below).

### Measured: int8 first layer vs all binary (3 seeds)

| Network | Accuracy, all binary | int8 first layer | gain | Flash, all binary (bits) | int8 first layer |
|---|---|---|---|---|---|
| min | 32.7 ± 2.7 % | 37.5 ± 1.6 % | **+4.8** | 0.7 KB | 0.8 KB |
| max-10% | 53.4 ± 1.9 % | 57.9 ± 0.3 % | **+4.5** | 2.9 KB | 3.1 KB |
| max-5% | 56.8 ± 0.7 % | 57.0 ± 1.2 % | +0.2 | 5.6 KB | 5.7 KB |
| max | 58.8 ± 0.9 % | 58.5 ± 2.6 % | −0.3 | 5.6 KB | 6.5 KB |

Latency of the int8 first layer (weights in flash, ms, mean of 3 seeds; in parentheses the all-binary network as ±1
bytes, the same kernel):

| Network | ESP32 | ESP32-S3 | Pico 2 Hazard3 | Pico 2 M33 | TFLite INT8, ESP32 / S3 |
|---|---|---|---|---|---|
| min | 0.77 (0.78) | 0.50 (0.50) | 0.56 (0.57) | 0.84 (0.82) | 4.17 / 0.72 |
| max-10% | 4.00 (4.01) | 1.84 (1.84) | 2.59 (2.56) | 3.66 (3.64) | 31.23 / 2.93 |
| max-5% | 10.98 (10.99) | 3.85 (3.85) | 6.00 (5.99) | 9.90 (9.89) | 114.22 / 7.14 |
| max | 14.07 (14.08) | 3.87 (3.87) | 8.59 (8.59) | 11.81 (11.79) | 115.62 / 7.11 |

The first layer alone on the ESP32-S3 (ms, from the per-step times of `chip_test.py`):

| Network | bits | int8 (PIE) | faster | whole network, bits → int8 |
|---|---|---|---|---|
| min | 0.57 | 0.24 | 2.4× | 0.83 → 0.50 ms (1.66×) |
| max-10% | 0.97 | 0.46 | 2.1× | 2.36 → 1.84 ms (1.28×) |
| max-5% | 2.06 | 1.04 | 2.0× | 4.87 → 3.85 ms (1.26×) |
| max | 2.18 | 1.22 | 1.8× | 4.83 → 3.87 ms (1.25×) |

Every run is 100 % equal to the simulation (48 runs on the S3: 4 networks × 3 forms × 3 seeds, flash and RAM). With the
int8 first layer, SYMBEX is still **5.4-10.4× faster than TFLite on the ESP32 and 1.44-1.84× on the S3**, with
6.5-10× less flash. TFLite stays more accurate: 4.5 points on max-10%, about 9 on the large networks, 19 on min.

### Does it carry over to the other benchmarks? A quick check

The same question for mlp_fashion and ResNet-8 on the ESP32-S3, with the real library kernels and random weights at the
real shapes (`benchmarks/experiments/int8_first_bench.py`, results in `experiments/results/int8_first_bench_s3.json`,
every method equal to a C reference on the chip):

| First layer | today (1-bit weights) | int8 weights | |
|---|---|---|---|
| mlp_fashion 784 → 128, weights in flash | **0.261 ms** | 6.637 ms | int8 25× slower |
| mlp_fashion 784 → 128, weights in RAM | 0.261 ms | **0.225 ms** | 1.16× (the kernel alone: 0.88×) |
| ResNet-8 stem, 32×32 × 16 channels | 7.84 ms (whole stem, `fine_profile`); counts alone 4.98 ms | **4.68 ms** (whole stem) | estimate: ~3 ms of 39.2 ms |

- **mlp_fashion: no.** Its first layer has 100 KB of int8 weights, which do not fit the S3's flash cache, while the
  1-bit weights (14 KB) do. In RAM, the two forms are within 16 %, as the theory predicts. The MLP keeps its 1-bit
  first layer.
- **ResNet-8: a small gain, an estimate.** An int8 stem would save about 3 ms of the 39.2 ms (~8 %, comparing the
  microbenchmark with the profile of the real stem). It needs an int8 stem in the CDT and retraining; it is future work.
- **Why conv1d gains and the others do not:** its first layer is small (32 rows × 45 weights: 1.4 KB in int8), so the
  int8 weights cost almost no flash and stay in the cache; the gain is the speed of the kernel and the accuracy of the
  weights, at no cost. The three factors together decide: 8-bit data, a chip that multiplies fast, a layer that fits.
  When the layer is large (the MLP), the 1-bit weights win through memory, and the fully binary network is still the
  right choice.

## Decisions (and why)

- **Binary vs binary is the main comparison:** the question this benchmark answers is whether SYMBEX's binary layers
  beat the binary library built for this task, on its own processor, with its own code and networks. INT8 is the
  reference that anyone deploying a network would use.
- **Their code is unchanged, with one exception:** the thresholds of their int8 first layer are `int16_t`, but int8 ×
  int8 products over 15 × 3 taps reach ±58,000 in the trained networks (max). In the copied sources they become
  `int32_t` (on RV32 an int32 load is the same instruction). Their pooled kernels get `dim_out` = the steps the pool
  uses.
- **Each library runs its first layer as designed:** ultracompactBNN with int8 weights, SYMBEX with 1-bit weights.
  Both networks are trained with the same recipe. SYMBEX also runs their network (int8 first layer), so the strict
  comparison (same network, same precision per layer) is there too.
- **Bits or bytes for the first layer is a choice per chip, not per network:** the same 1-bit weights and thresholds,
  exported both ways, give the same outputs. With the S3's PIE kernel, the bytes are now the fastest form on every
  chip measured (on the M33, max, the two are within 1 %); an automatic choice in the CDT should still measure both.
- **The S3's int8 kernel lives in the library** (`hal/esp32s3`, `symbex_s8_dot_rows`), as it is general: any first
  layer with int8 or ±1-byte weights uses it. The threshold shift for the uint8 input (T − 128·Σw) is computed once per
  layer, in 64 bits and clamped, so any threshold stays exact.
- **SYMBEX's PULP first layer lives in this folder** (`pulp/symbex_u8_pulp.c`, `p.sdotusp4`), as their whole library is
  PULP-specific. The library itself only gained general pieces: `symbex_bool1d`'s faster binary layers, the optional
  int8 first-layer weights (`SymbexBool1D.weights_i8`) and the `-DSYMBEX_BOOL1D_U8_EXTERNAL` hook.
- **Their library is not ported to the boards:** it exists only for PULP, and any port would be ours. On the boards
  the reference is TFLite Micro.
- **Split by subjects:** the 30 people are drawn once (numpy seed 0): 6 for test (3, 5, 11, 12, 22, 27), 4 for
  validation, 20 for training, the same for every seed. A window of one person never appears in two splits.
- **One recipe for every network:** Adam, learning rate 5e-3 with cosine decay, 100 epochs, batches of 128, no
  augmentation, seeds 42 / 43 / 44. The checkpoint is the best validation epoch; the test subjects are scored once.
- **Input:** the accelerometer divided by 19.58 m/s² (±2 g, the phone's range) × 127 → int8. SYMBEX takes uint8 =
  int8 + 128; the offset folds into the first thresholds.
- **INT8 from the same float network:** the Keras network of `har_tflite.py` is moved to PyTorch for `tools/lowbit`
  (checked: the same class on every validation window), so TFLite INT8 and the QAT baselines start from the same
  model.

## Earlier: single layers (October 2026, before the whole networks)

The first measurements ran one layer at a time, on six shapes. They used SYMBEX's earlier kernels (one or two windows
per pass): the whole-network tables above replace them. With those kernels, layers whose window fits one word and has
few output channels were slower than ultracompactBNN (0.47× on min's 4 → 4 k7 layers); the current kernel for those
layers (`conv_word`) makes them 1.21-1.50× faster than theirs.

| Layer (L_in × C_in × C_out × K) | ultracompactBNN instr | SYMBEX v2 instr | ratio |
|---|---|---|---|
| 151 × 3 × 8 × 15 | 43 616 | 22 101 | 1.97× |
| 137 × 8 × 8 × 15 | 53 514 | 28 450 | 1.88× |
| 64 × 16 × 16 × 7 | 32 003 | 23 420 | 1.37× |
| 64 × 32 × 32 × 7 | 76 895 | 71 032 | 1.08× |
| 128 × 32 × 32 × 15 | 317 789 | 267 729 | 1.19× |
| 32 × 64 × 64 × 3 | 69 015 | 61 040 | 1.13× |

Their library also needs output channel counts that are powers of 2 (5 or 7 give wrong outputs, checked on the PC);
SYMBEX takes any count.

## Files

| File | What it does |
|---|---|
| `har_data.py` | UniMiB-SHAR (AF-17) → int8 windows, the subject split → `data/har_unimib.npz` |
| `model.py` | the four networks of Table 1, binary; first layer `binary` (1-bit weights) or `int8` |
| `har_train.py` | trains one network, or `--all`: 4 networks × 2 first layers × 3 seeds |
| `har_export.py` | integers, simulation, `model_symbex.h`, the C check on the PC (EXACTNESS 100 %); `--first-bytes`: the 1-bit first layer as ±1 bytes, in `runs/<run>_bytes/` |
| `har_gvsoc.py` | the whole networks through both libraries: `--pc` (gcc, exactness) or GVSoC (instructions, cycles per layer) |
| `pulp/symbex_u8_pulp.c` | SYMBEX's first layer for PULP (`p.sdotusp4`), 1-bit or int8 weights |
| `har_tflite.py` | the same architectures in Keras → INT8 → TFLite Micro on the ESP32 / ESP32-S3 (`--weights ram`) |
| `har_lowbit.py` | INT8 / INT4 / INT2 baselines with `tools/lowbit`, from the same float networks |
| `ucbnn_check.py`, `ucbnn_gvsoc.py` | the single-layer checks and measurements (section "Earlier") |
| `pulp/Dockerfile` | PULP GCC, PULP SDK and GVSoC (pulp-open) on Ubuntu 22.04 |
| `../../experiments/int8_first_bench.py` | the quick check of section 3: an int8 first layer for mlp_fashion and ResNet-8 on the S3 |

## Reproduce

```
cd benchmarks/conv1d/har_ultracompact                       # environment env_bench
python har_data.py                                          # needs UniMiB-SHAR in ../../data/unimib_shar/
python har_train.py --all
for r in runs/*_e100_s4?; do python har_export.py $r; done
for r in runs/*_binary_e100_s4?; do python har_export.py $r --first-bytes; done   # the first layer as +-1 bytes
python har_gvsoc.py --repo ~/Proyectos/ultracompactBNN --pc runs/max_int8_e100_s42 runs/max_binary_e100_s42

# GVSoC (Docker; --user so that runs/ is not owned by root)
cd pulp && docker build -t symbex-pulp . && cd ..
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v ~/Proyectos:/work symbex-pulp bash -c \
  "source /opt/pulp-sdk/configs/pulp-open.sh; export PATH=/opt/gvsoc/install/bin:\$PATH; pip3 install -q --user numpy; \
   cd /work/Symbex1/benchmarks/conv1d/har_ultracompact && \
   python3 har_gvsoc.py --repo /work/ultracompactBNN runs/max_int8_e100_s42 runs/max_binary_e100_s42"

# INT8 references
python har_tflite.py --all
python har_tflite.py --arch max --seed 42 --board esp32 --port /dev/ttyUSB0
python har_tflite.py --arch max --seed 42 --board s3 --port /dev/ttyACM0 --usb --weights ram
python har_lowbit.py --arch max

# SYMBEX on the boards (from tools/)
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_binary_e100_s42 --board esp32 --port /dev/ttyUSB0 --popcount swar
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_int8_e100_s42 --board s3 --port /dev/ttyACM0 --usb --popcount swar
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_int8_e100_s42 --board s3 --port /dev/ttyACM0 --usb --popcount swar --weights ram
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_binary_e100_s42 --board pico2_riscv --port /dev/ttyACM0
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_binary_e100_s42 --board pico2_arm --port /dev/ttyACM0 --weights ram
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_binary_e100_s42_bytes --board pico2_riscv --port /dev/ttyACM0

# The quick check of section 3 (ESP32-S3)
python ../../experiments/int8_first_bench.py --port /dev/ttyACM0
```

Results: `results/` (GVSoC, accuracy, TFLite, lowbit), `runs/<run>/chip_<board>_<popcount>_<weights>_mask.json` (SYMBEX
on the boards).

## Future work

1. **Testing the theory of section 3:** a task with a binary input (fully binary should win there), the int8 first
   layer on other 8-bit tasks and chips, and the layer-size limit measured on whole networks, not only kernels.
2. **Networks sized for binary, with the int8 first layer** (wider binary layers), compared with INT8 sized the same
   way, as in ResNet-8 and mlp_fashion: how speed, accuracy and size change, and the way to close the accuracy gap.
3. **Bits or bytes chosen automatically:** a kernel search (`tools/dev/kernel_search.py`) that measures both forms per
   chip and layer shape and lets the CDT pick, like `arithmetic = "auto"`.
4. **First-layer kernels per chip:** SMLAD on the M33 (the S3's PIE kernel is done), with a running sum of the input
   where the instruction has no sign.
5. **An int8 stem for ResNet-8:** about 8 % of the network on the S3 (estimate from `int8_first_bench.py`), to be
   trained and measured.
6. **Early exit in the OR pool, per channel:** measured on the PC, half of the steps of the first binary layer can be
   skipped (a channel that already gave 1 in its pool window); it trades against the 4-channel blocks, to be measured.
7. **The binary layers on chips without a native bit count** (ESP32, S3, M33): the blocked kernels of the RI5CY.
8. **TFLite Micro on the Pico 2** (CMSIS-NN on the M33), and the rest of the ESP32 grid (`swar` with the weights in
   RAM), energy per inference (INA226).
