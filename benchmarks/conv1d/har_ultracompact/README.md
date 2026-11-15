# har_ultracompact — binary 1D convolutions for human activity recognition

Human activity recognition from a phone's accelerometer (UniMiB-SHAR, 17 activities) with **binary 1D convolutional
networks**. It compares two binary libraries first: SYMBEX and the closest prior work, the binary inference library of
**Daghero et al., "Ultra-compact Binary Neural Networks for Human Activity Recognition on RISC-V Processors"** (ACM
Computing Frontiers 2021, `github.com/francescodaghero/ultracompactBNN`). The comparison uses their four networks,
their processor and their own code. INT8 is the reference: TensorFlow Lite Micro and the INT8 / INT4 / INT2 baselines
of `tools/lowbit`, with the same architectures, measured on real boards.

**The four networks: xs, s, m, l (by size, as the other benchmarks name theirs).** They are the four UniMiB-SHAR
networks of Daghero et al., Table 1, which come from the Pareto points of the authors' architecture search (their
section 4.5). Their paper names them by accuracy: min and max are the networks with the minimum and the maximum
accuracy of the search, max-10% and max-5% the smallest networks whose accuracy is less than 10 % / 5 % below max. The
code and the file names (runs, results) keep those names (`min`, `max10`, `max5`, `max`); every `--arch` takes both.
Every network: three convolutions with no padding, a max pool of 4 after the second and the third, one linear layer
for the 17 classes; input 151 steps × 3 axes.

| Name | Daghero et al. (code, files) | conv 1 (channels, kernel) | conv 2 | conv 3 | their accuracy | their total memory |
|---|---|---|---|---|---|---|
| xs | min (`min`) | 4, 7 | 4, 7 | 4, 7 | 33.0 % | 13.3 kB |
| s | max-10% (`max10`) | 8, 15 | 32, 7 | 32, 7 | 58.6 % | 19.7 kB |
| m | max-5% (`max5`) | 32, 7 | 32, 15 | 32, 15 | 65.1 % | 24.0 kB |
| l | max (`max`) | 32, 15 | 32, 15 | 32, 15 | 68.0 % | 26.1 kB |

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
  **5.3-10.4× faster** on the ESP32, **1.47-1.90× faster** on the ESP32-S3 and **3.3-7.9× faster** on the Pico 2's
  Cortex-M33 (TFLite with CMSIS-NN). It takes **2-10× less RAM** and
  **6.5-10× less flash**. INT8 is more accurate on these small networks: about 9 points on the large ones, 23 on the
  smallest.
- **Where fully binary stops paying: a working theory.** Binary network work usually keeps the first layer in higher
  precision because it is a small share of the work. SYMBEX's measurements suggest the reason and the boundary: the
  gain of binary weights depends on the **data**. A binary input makes the first layer one more binary layer (32
  products per bit count), where fully binary should win. An 8-bit input (here the accelerometer) costs about one
  operation per input whether the weights have 1 bit or 8. There the int8 first layer is **faster than the fully
  binary one** on every board measured (equal on the M33's largest network; on the classic ESP32's largest
  network the bits are faster since the Nov 7 short-layer fix: 12.78 vs 14.07 ms), because the chips multiply bytes faster
  than they count bit planes (on the ESP32-S3, a new PIE kernel: **1.8-2.5×** for that layer, **1.26-1.70×** for the
  whole network). It is also **more accurate**: **+4.5-4.8 points** on the two small networks, equal on the large
  ones. Measured on one task (activity recognition) and checked quickly on two others: a theory to test further, not a
  general rule.
- **A third factor: the size of the layer.** In a quick test on the ESP32-S3, mlp_fashion's 784 → 128 first layer
  stays faster with 1-bit weights, because its int8 weights (100 KB) do not fit the flash cache (section 3).
- **Exact on every board:** ESP32, ESP32-S3 and both cores of the Raspberry Pi Pico 2 (Hazard3 RISC-V, Cortex-M33),
  1,000 windows per run, 100 % equal to the simulation in every run.
- **Accuracy: an open case.** Bop, wider networks (INT8 widened alike) and two training fixes (clipped latent weights,
  pooling before sign), a cross-validated checkpoint choice and distillation from the float network do not close the
  gap to INT8 (sections 4, 5, 7, 8). The cause is not the fixed thresholds (AdaBN does not help) but the training
  fit; two-bit or ternary activations after the first layer recover 1-7 points on the test subjects, and the gap stays
  4.6-14.6 points (section 9).

## 1. Binary vs binary: SYMBEX and ultracompactBNN on their processor

### The setup

| | ultracompactBNN | SYMBEX |
|---|---|---|
| Processor | RI5CY (PULP), simulated with GVSoC (pulp-open) | the same |
| Compiler | PULP GCC, `-O3 -flto` | the same |
| Bit count | `p.cnt` (hardware) | the same instruction |
| First layer | int8 input × int8 weights (`sdotsp4`) | uint8 input × 1-bit weights (`p.sdotusp4`), or their int8 weights |
| Other layers | their 1-bit kernels, fused max pool | `symbex_bool1d`: XOR + bit count, fused OR pool |
| Networks | Table 1 of their paper, UniMiB-SHAR: xs, s, m, l | the same |

The main measure is the **instruction count**. Their chip, PULPissimo, is no longer a buildable target in GVSoC, and
the memory of the pulp-open platform is slower for the core (about 2.2 cycles per instruction for both libraries).
Instructions do not depend on the simulated memory; cycles are reported too.

### Whole networks (instructions, mean of 3 seeds; > 1: SYMBEX does less work)

| Network (Table 1) | ultracompactBNN | SYMBEX, all binary | ratio | SYMBEX, their network (int8 first layer) | ratio |
|---|---|---|---|---|---|
| xs | 19 559 | 18 201 | **1.07×** | 18 530 | **1.06×** |
| s | 183 372 | 133 498 | **1.37×** | 133 398 | **1.37×** |
| m | 473 697 | 385 094 | **1.23×** | 385 262 | **1.23×** |
| l | 499 645 | 405 104 | **1.23×** | 405 215 | **1.23×** |

In cycles: 1.26-1.38× on the three large networks; on `xs`, 0.94-0.95× (their code takes fewer cycles there).

Per layer (`s`, seed 42): first layer 1.07×, binary conv 8→32 k7 + pool 1.59×, binary conv 32→32 k7 + pool 1.14×,
classifier 0.64× (about 1,000 instructions, under 1 % of the network). The binary layers are where SYMBEX gains; the
first layer costs about the same in both libraries, with 1-bit or int8 weights (section 3 explains why).

### Accuracy (test subjects, integer simulation = what the chip runs; mean ± sample std, 3 seeds)

| Network | SYMBEX, all binary | ultracompactBNN (int8 first layer) | Daghero et al. published* |
|---|---|---|---|
| xs | 32.7 ± 3.3 % | 37.5 ± 1.9 % | 33.0 % |
| s | 53.4 ± 2.4 % | 57.9 ± 0.4 % | 58.6 % |
| m | 56.8 ± 0.9 % | 57.0 ± 1.5 % | 65.1 % |
| l | 58.8 ± 1.1 % | 58.5 ± 3.2 % | 68.0 % |

\*Their split is not published, so their numbers are not directly comparable. SYMBEX with their network (int8 first
layer) has their accuracy by construction: it is the same network, run exactly.

## 2. Binary vs INT8 on real boards

### Accuracy of the integer baselines (test, mean ± sample std, 3 seeds)

| Network | SYMBEX (1 bit) | TFLite INT8 | INT8 QAT | INT4 QAT | INT2 QAT | FP32 |
|---|---|---|---|---|---|---|
| xs | 32.7 ± 3.3 % | 56.1 ± 5.2 % | 51.3 ± 0.1 % | 51.0 ± 1.2 % | 35.7 ± 0.3 % | 50.6 % |
| s | 53.4 ± 2.4 % | 62.4 ± 2.5 % | 59.9 ± 0.2 % | 59.5 ± 1.0 % | 51.8 ± 3.2 % | 59.8 % |
| m | 56.8 ± 0.9 % | 66.2 ± 3.2 % | 69.1 ± 0.3 % | 69.6 ± 0.1 % | 65.4 ± 0.9 % | 68.6 % |
| l | 58.8 ± 1.1 % | 67.9 ± 1.1 % | 68.0 ± 0.1 % | 67.6 ± 0.3 % | 66.5 ± 1.2 % | 67.6 % |

TFLite INT8: three float networks (seeds 42-44), each quantized after training. QAT (`tools/lowbit`, Brevitas): the
seed-42 float network, quantization-aware training with 3 seeds (15 epochs, distillation). The FP32 column is that
seed-42 TFLite float network, copied into PyTorch and scored by `tools/lowbit`: one seed, not a three-seed mean. On
`xs`, TFLite INT8 above FP32 (56.1 vs 50.6 %) compares a three-seed mean with one seed, not a quantization gain (the
seed-42 INT8 network scores 50.6 % too).
The ± is the sample standard deviation (n − 1) over 3 seeds; earlier versions of this README used the population one
(n), about 1.22× smaller.

The test set has few subjects and they differ a lot: TFLite INT8 on `xs` goes from 50.6 to 60.8 % across seeds, and
validation (4 other subjects, none in training) is 3-12 points above test, mean over seeds, depending on method. Read
differences of a few points as within noise. As in mlp_fashion, **INT2 QAT is more accurate per byte
than SYMBEX** on the large networks (+8 points with about 1.5× the weight bytes); SYMBEX's case is speed and memory.

### Latency, weights in flash (median, ms; 1,000 validation windows, mean of 3 seeds)

The all-binary network runs its first layer in one of two forms, with the same weights, thresholds and outputs:
"bits" (the 8 bit planes of the input, built once per inference, counted against the 1-bit rows) or "±1 bytes" (the
first-layer weights stored as +1 / −1 bytes and run by the int8 kernel, `har_export.py --first-bytes`).

| Network | ESP32: first layer in bits | ±1 bytes | TFLite INT8 | SYMBEX vs TFLite |
|---|---|---|---|---|
| xs | 1.11 | **0.78** | 4.17 | **5.3×** |
| s | 4.10 | **4.01** | 31.23 | **7.8×** |
| m | 11.21 | **10.99** | 114.22 | **10.4×** |
| l | **12.78** | 14.08 | 115.62 | **9.0×** |

| Network | ESP32-S3: first layer in bits | ±1 bytes | TFLite INT8 | SYMBEX vs TFLite |
|---|---|---|---|---|
| xs | 0.83 | **0.49** | 0.72 | **1.47×** |
| s | 2.36 | **1.83** | 2.93 | **1.60×** |
| m | 4.87 | **3.76** | 7.14 | **1.90×** |
| l | 4.83 | **3.83** | 7.11 | **1.85×** |

| Network | Pico 2 M33: bits | ±1 bytes | TFLite INT8 (CMSIS-NN) | SYMBEX vs TFLite |
|---|---|---|---|---|
| xs | 1.22 | **0.82** | 2.71 | **3.3×** |
| s | 4.09 | **3.64** | 12.00 | **3.3×** |
| m | 10.48 | **9.89** | 78.04 | **7.9×** |
| l | **11.72** | 11.79 | 77.70 | **6.6×** |

| Network | Pico 2 Hazard3: bits | ±1 bytes | TFLite INT8 (reference kernels) |
|---|---|---|---|
| xs | 1.31 | **0.57** | 17.29 |
| s | 3.59 | **2.56** | 225.72 |
| m | 8.26 | **5.99** | 1159.86 |
| l | 10.67 | **8.59** | 1133.74 |

Every run in these tables is 100 % equal to the simulation; the spread between the 3 seeds is at most 0.03 ms (on the
S3 at most 0.004 ms). The ESP32's bits columns were measured again on Nov 7, after a faster method for short
first layers (`symbex_swar_u8_masked_sum` counts the last partial word 4 inputs at a time; before: 1.53 / 4.60 /
14.71 / 14.77 ms in flash, 1.58 / 4.43 / 13.28 / 13.40 ms in RAM); the other boards do not use that code. The S3's ±1 bytes column uses the PIE kernel of section 3; before it, the bits were faster there
(`l`: 4.83 vs 8.70 ms).

SYMBEX counting modes: `swar` on the ESP32 and the ESP32-S3 (no lookup table: 1.3-2.8 KB of RAM; `lut16` is up to 8 %
faster on the ESP32, for 64 KB more, and gives the same speed on the S3), native `cpop` on the Hazard3, `lut16` on the M33 (`swar`:
1.25 / 4.92 / 14.49 / 15.49 ms all binary with bits, 2.2-3.4 KB of RAM instead of 67 KB).

TFLite Micro on the Pico 2 (`har_tflite.py --board pico2_arm | pico2_riscv`, TFLite Micro at a pinned commit, 1,000
validation windows, 3 seeds, the spread at most 0.01 ms): every run gives 100 % the same class as the PC interpreter.
On the M33 TFLite runs CMSIS-NN, its optimized kernels for that core: the fair comparison, SYMBEX **3.3-7.9×
faster**. On the Hazard3 TFLite has only its plain reference kernels (no optimized kernels exist for that core), so
its 30-194× gap is a reference, not a like-for-like comparison. TFLite's flash / arena on the Pico 2: 7.3 / 2.8 KB
(`xs`), 20.5 / 7.4 KB (`s`), 41.8 / 12.7 KB (`m`), 42.5 / 12.2 KB (`l`), M33.

### Latency, weights in RAM (median, ms)

| Network | ESP32: all binary (lut16) | int8 first layer (lut16) | TFLite INT8 | S3: all binary, bits | ±1 bytes | int8 first layer | TFLite INT8 |
|---|---|---|---|---|---|---|---|
| xs | 1.15 | 0.79 | 3.74 | 0.83 | **0.49** | **0.49** | 0.71 |
| s | 3.93 | 3.74 | 28.72 | 2.36 | **1.83** | **1.83** | 2.93 |
| m | 9.77 | 9.33 | 100.97 | 4.87 | **3.76** | **3.76** | 6.40 |
| l | **11.39** | 12.30 | 102.70 | 4.83 | **3.83** | **3.84** | 6.16 |

TFLite with the weights in RAM copies the whole model at boot. On the S3, SYMBEX takes the same time with the weights in
flash or in RAM (its weights fit the cache: 0.7-6.5 KB). On the Pico 2, the weights in RAM change SYMBEX's latency by
under 10 %.

### Memory (`l` network)

| | SYMBEX all binary, bits | first layer as ±1 bytes | SYMBEX int8 first layer | TFLite INT8 |
|---|---|---|---|---|
| Flash (weights + thresholds) | **5.6 KB** | 6.5 KB | 6.5 KB | 42.5 KB |
| RAM (no lookup table) | **2.4-2.7 KB** | 2.4-2.7 KB | 2.4-2.7 KB | 10.3 KB (ESP32) / 25.5 KB (S3) of arena |

The ±1 bytes cost little flash: the 1-bit rows are padded to 4 words (128 bits) for the 45 weights of a row, so bytes
take 2.8× that layer's rows, not 8×. Smallest network (`xs`): SYMBEX 0.7 KB of flash (bits or bytes), 1.3-1.6 KB of
RAM; TFLite 7.3 KB of flash, 2.7-2.9 KB of arena.

### Notes

- **The ESP32-S3 has SIMD on both sides:** TFLite runs on ESP-NN, which uses the S3's PIE vector unit for its int8
  products; SYMBEX uses it to count bits and, since section 3, for the first layer too. The classic ESP32 has no SIMD
  for either: there SYMBEX is 5.3-10.4× faster.
- **The first layer decides the whole network:** the binary layers after it take the same time in every variant
  (Hazard3, `l`: 2.81 vs 2.86 ms; S3, `l`: 2.65 ms with bits, bytes or int8), so the differences between the columns
  are the first layer alone. All binary with bits, it is 32-92 % of the network.
- **The first layer was 1.1-4.4× faster in this version** (bit planes built once for the whole input instead of per
  step, no padding of the window to 128 inputs except for the S3's PIE, a running sum): Hazard3 `xs` 5.45 → 1.31 ms,
  M33 2.13 → 1.22 ms, ESP32 2.37 → 1.53 ms (1.11 ms after the Nov 7 short-layer fix); the S3 already had its PIE path (0.84 → 0.83 ms).

## 3. Where fully binary stops paying: the first layer (a working theory)

### How it came up

To compare with ultracompactBNN on equal terms, SYMBEX also had to run **their** network, which keeps the first layer
in int8 (section 1). On the boards that layer first ran on a generic C kernel. Writing a PIE kernel for it on the
ESP32-S3 (`symbex_s8_dot_rows`: 16 products per `ee.vmulas.s8.accx`, rows at any alignment so the weights stay in
flash) made the first layer **1.8-2.5× faster than SYMBEX's own bit planes**, which had been the fastest form on the
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
| Evidence so far | indirect: no benchmark here has a binary input yet; the binary layers, 1.14-1.59× fewer instructions than ultracompactBNN, 0.19 cycles per product (Hazard3) | the int8 first layer is faster than the bits on all four boards (M33, l: within 1 %; ESP32, l: the bits are faster, 12.78 vs 14.07 ms) and more accurate (tables below) |

If the theory holds, the earlier practice is right for 8-bit data, for a reason it did not state: with an 8-bit
input, the first layer does not gain from 1-bit weights at all. And on chips that multiply bytes fast (here all four
boards, most of all the ESP32-S3's PIE), the int8 first layer **beats the fully binary network** in speed, not only in
accuracy. The last layer (the classifier) weighs little here: 0.009 ms of 3.84 ms (0.2 %) on the S3.

### The theory: binary weights pay off only where the input is binary too

In a binary layer both operands are bits: one XOR + bit count covers 32 products (0.19 cycles per product on the
Hazard3, `l`). The first layer is different: its input is 8-bit (the accelerometer). With 1-bit weights,

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
| xs | 32.7 ± 3.3 % | 37.5 ± 1.9 % | **+4.8** | 0.7 KB | 0.8 KB |
| s | 53.4 ± 2.4 % | 57.9 ± 0.4 % | **+4.5** | 2.9 KB | 3.1 KB |
| m | 56.8 ± 0.9 % | 57.0 ± 1.5 % | +0.2 | 5.6 KB | 5.7 KB |
| l | 58.8 ± 1.1 % | 58.5 ± 3.2 % | −0.3 | 5.6 KB | 6.5 KB |

Latency of the int8 first layer (weights in flash, ms, mean of 3 seeds; in parentheses the all-binary network as ±1
bytes, the same kernel):

| Network | ESP32 | ESP32-S3 | Pico 2 Hazard3 | Pico 2 M33 | TFLite INT8, ESP32 / S3 |
|---|---|---|---|---|---|
| xs | 0.77 (0.78) | 0.49 (0.49) | 0.56 (0.57) | 0.84 (0.82) | 4.17 / 0.72 |
| s | 4.00 (4.01) | 1.83 (1.83) | 2.59 (2.56) | 3.66 (3.64) | 31.23 / 2.93 |
| m | 10.98 (10.99) | 3.76 (3.76) | 6.00 (5.99) | 9.90 (9.89) | 114.22 / 7.14 |
| l | 14.07 (14.08) | 3.84 (3.83) | 8.59 (8.59) | 11.81 (11.79) | 115.62 / 7.11 |

The first layer alone on the ESP32-S3 (ms, from the per-step times of `chip_test.py`):

| Network | bits | int8 (PIE) | faster | whole network, bits → int8 |
|---|---|---|---|---|
| xs | 0.57 | 0.23 | 2.5× | 0.83 → 0.49 ms (1.70×) |
| s | 0.97 | 0.45 | 2.2× | 2.36 → 1.83 ms (1.29×) |
| m | 2.06 | 0.96 | 2.2× | 4.87 → 3.76 ms (1.29×) |
| l | 2.18 | 1.19 | 1.8× | 4.83 → 3.84 ms (1.26×) |

Every run is 100 % equal to the simulation (48 runs on the S3: 4 networks × 3 forms × 3 seeds, flash and RAM). With the
int8 first layer, SYMBEX is still **5.4-10.4× faster than TFLite on the ESP32 and 1.47-1.90× on the S3**, with
6.5-10× less flash. TFLite stays more accurate: 4.5 points on `s`, about 9 on the large networks, 19 on `xs`.

### Does it carry over to the other benchmarks? A quick check

The same question for mlp_fashion and ResNet-8 on the ESP32-S3, with the real library kernels and random weights at the
real shapes (`benchmarks/experiments/chip_kernels/int8_first_bench.py`, results in `experiments/chip_kernels/results/int8_first_bench_s3.json`,
every method equal to a C reference on the chip):

| First layer | today (1-bit weights) | int8 weights | |
|---|---|---|---|
| mlp_fashion 784 → 128, weights in flash | **0.261 ms** | 6.633 ms | int8 25× slower |
| mlp_fashion 784 → 128, weights in RAM | 0.261 ms | **0.221 ms** | 1.18× (the kernel alone: 0.89×) |
| ResNet-8 stem, 32×32 × 16 channels | 7.84 ms (whole stem, `fine_profile`); counts alone 4.98 ms | **4.52 ms** (whole stem) | estimate: ~3 ms of 39.2 ms |

- **mlp_fashion: no.** Its first layer has 100 KB of int8 weights, which do not fit the S3's flash cache, while the
  1-bit weights (14 KB) do. In RAM, the two forms are within 18 %, as the theory predicts. The MLP keeps its 1-bit
  first layer.
- **ResNet-8: a small gain, an estimate.** An int8 stem would save about 3 ms of the 39.2 ms (~8 %, comparing the
  microbenchmark with the profile of the real stem). It needs an int8 stem in the CDT and retraining; it is future work.
- **Why conv1d gains and the others do not:** its first layer is small (32 rows × 45 weights: 1.4 KB in int8), so the
  int8 weights cost almost no flash and stay in the cache; the gain is the speed of the kernel and the accuracy of the
  weights, at no cost. The three factors together decide: 8-bit data, a chip that multiplies fast, a layer that fits.
  When the layer is large (the MLP), the 1-bit weights win through memory, and the fully binary network is still the
  right choice.

## 4. Bop for the binary weights (November 13, validation only)

**Question.** Bop gave the binary ResNet-8 +2.5 points (benchmarks/experiments/training_gap). Does it close part of the
gap to INT8 here?

**What.** `har_train.py --optimizer bop --bop-gamma g` (the CDT's Bop for the 1-bit weights, the rest with Adam),
the same recipe as Adam otherwise: 100 epochs, seeds 42, 43, 44, all-binary and int8 first layer, the 4 networks.
Every choice is made on the validation subjects (`--no-test`); the test subjects are scored once, at the end, with the
chosen recipe.

**Result.** Validation accuracy, mean ± sample std over 3 seeds:

| Network | first layer | Adam | Bop γ 1e-3 | Bop γ 1e-2 | Bop γ 1e-4 |
|---|---|---|---|---|---|
| xs | binary | **44.9 ± 2.2** | 40.6 ± 3.1 | 41.9 ± 2.4 | 40.1 ± 5.2 |
| xs | int8 | **46.1 ± 2.2** | 45.1 ± 1.0 | 44.8 ± 1.4 | 46.0 ± 1.6 |
| s | binary | 62.8 ± 1.3 | **62.9 ± 1.1** | 60.8 ± 0.6 | 63.5 ± 1.4 |
| s | int8 | 63.6 ± 1.7 | 63.4 ± 1.0 | 62.7 ± 1.7 | **63.8 ± 1.1** |
| m | binary | 69.2 ± 0.6 | **69.7 ± 0.5** | 68.6 ± 1.8 | 69.6 ± 0.7 |
| m | int8 | 67.9 ± 1.4 | **69.0 ± 2.1** | 68.9 ± 1.1 | 69.0 ± 1.4 |
| l | binary | 68.6 ± 0.9 | **71.3 ± 0.6** | 69.2 ± 1.2 | 69.1 ± 0.1 |
| l | int8 | 69.3 ± 0.9 | 68.7 ± 1.2 | **71.1 ± 1.9** | 69.4 ± 1.0 |

**Answer.** Bop helps clearly in one case: the all-binary `l` network, +2.7 points with γ 1e-3 (more than the spread
of either method). Elsewhere the differences are within one standard deviation, and on the smallest network Adam is
ahead: there the network lacks capacity, not training. γ 1e-4 (the CDT's default) is no better than Adam anywhere.

**Which recipe goes to the test subjects, and why.** The plan fixed beforehand said "choose on validation" but not how
large a gain must be; the rule below was written after seeing the table above and before any Bop run was scored on
the test subjects:

> For each network and first layer, Bop replaces Adam only if its best γ beats Adam's validation mean by more than
> the larger of the two standard deviations. Otherwise Adam stays.

- **Seed noise.** With 3 seeds, a difference smaller than one standard deviation can come from the seeds alone.
- **Picking the best of several options flatters it.** Each row compares 4 candidates (Adam and 3 values of γ). If
  all four were equally good, the best mean of 3 seeds would still look about 0.6 standard deviations better than it
  is (the expected maximum of 4 normal values is 1.03 σ, and a mean of 3 seeds has σ / √3; a textbook estimate, not
  measured here). A margin of one full standard deviation is larger than this bias.
- **The test subjects are used once.** Adam is already scored on them (sections above); a change goes to the test
  only when validation shows it clearly, so the test is not used to decide between near-ties.
- **The first layer is not a choice here:** both forms (1-bit and int8) are reported, as in the rest of this README.

Applied to the table: only **`l`, all-binary, γ 1e-3** passes (71.3 vs 68.6, margin 2.7 > 0.9). The nearest miss is
`l` with an int8 first layer, γ 1e-2 (71.1 ± 1.9 vs 69.3 ± 0.9, margin 1.8 < 1.9): it stays with Adam. The chosen
runs' saved checkpoints (the best validation epoch, as always) are scored on the test subjects without training again
(`har_train.py --arch max --optimizer bop --bop-gamma 1e-3 --seeds 42,43,44 --score-test`).

**Test subjects (scored once).** The gain does not carry over:

| l, all-binary | validation | test | test per seed (42 / 43 / 44) |
|---|---|---|---|
| Adam | 68.6 ± 0.9 | 58.8 ± 1.1 | 59.5 / 57.5 / 59.4 |
| Bop γ 1e-3 | 71.3 ± 0.6 | 59.3 ± 3.1 | 59.4 / 56.1 / 62.3 |

Paired by seed, Bop minus Adam is -0.1 / -1.4 / +2.9 (+0.5 on average), within the spread. On the test subjects Bop
gives no clear gain, and the gap to TFLite INT8 (67.9 on `l`) stays. The +2.7 on validation was partly the validation
subjects themselves: they are few, and the earlier sections already show validation running 5-10 points above test.
The network, its layers and its speed are the same as with Adam, so nothing new is exported or measured on the chips;
the Adam results above stay the reference. What may still close the gap is capacity (wider layers, sized for binary,
with INT8 resized the same way), not the optimizer.

## 5. Capacity: wider networks (November 13, test scored once)

**Question.** Section 4 says the gap to INT8 is not about the optimizer. Is it capacity? Do wider layers close it?

**What.** `xs` and `l` with every convolution's channels × 1.5 and × 2; kernels and pools stay the same (`model.py`,
`--arch xs_w15 / xs_w2 / l_w15 / l_w2`, files `min_w15` ... `max_w2`): `xs` 4 → 6 / 8 channels, `l` 32 → 48 / 64. TFLite INT8 is resized the
same way (`har_tflite.py --arch ...`). Same recipe as the rest of this README: Adam, 100 epochs, seeds 42, 43, 44, the
best validation epoch. All of it was screened on the validation subjects only (`--no-test`).

**Result.** Validation accuracy, mean ± sample std over 3 seeds (× 1 = the networks of the sections above):

| Network | width | SYMBEX all-binary | SYMBEX int8 first layer | TFLite INT8 | INT8 model |
|---|---|---|---|---|---|
| xs | × 1 | 44.9 ± 2.2 | 46.1 ± 2.2 | 61.1 ± 3.4 | 7.4 KB |
| xs | × 1.5 | 49.0 ± 1.6 | 50.5 ± 0.6 | 63.2 ± 0.7 | 8.0 KB |
| xs | × 2 | 51.0 ± 0.9 | 50.9 ± 1.5 | 65.8 ± 1.4 | 8.8 KB |
| l | × 1 | 68.6 ± 0.9 | 69.3 ± 0.9 | 73.6 ± 1.1 | 42.5 KB |
| l | × 1.5 | 71.7 ± 0.4 | 72.9 ± 1.1 | 74.4 ± 0.5 | 82.9 KB |
| l | × 2 | 72.7 ± 0.2 | 72.8 ± 0.6 | 76.6 ± 1.0 | 138.5 KB |

Width helps both methods. On validation the gap to INT8 at the same width narrows on `l` (4.3-5.0 points at × 1,
1.5-2.7 at × 1.5, 3.8-3.9 at × 2) and stays large on `xs` (13-16 points at every width). Binary `l` × 1.5 / × 2 is
close to INT8 `l` × 1 (71.7-72.9 vs 73.6), the "sized for binary" comparison; whether it is also faster is not measured yet.

**What goes to the test subjects, and why.** This rule was written after the table above and before any wider network
was scored on the test subjects:

> Nothing is chosen here: this is a curve, not a search. Every wider network (2 widths × 2 networks × SYMBEX
> all-binary, SYMBEX int8 first layer, TFLite INT8 × 3 seeds) is scored on the test subjects once, with its best
> validation checkpoint and without training again. All of them are reported.

- **No option is picked from the test subjects,** so the best-of-several bias of section 4 does not apply.
- **Same architecture, same recipe:** each SYMBEX row is compared with TFLite INT8 at the same width.
- **The × 1 rows are already scored** (sections 1-2); they are not scored again.

**Test subjects (scored once).** Mean ± sample std over 3 seeds; per seed (42 / 43 / 44) in the JSONs
(`results/symbex/har_capacity.json`, `results/tflite/tflite_pc_<arch>_s*.json`):

| Network | width | SYMBEX all-binary | SYMBEX int8 first layer | TFLite INT8 | gap (best SYMBEX) |
|---|---|---|---|---|---|
| xs | × 1 | 32.7 ± 3.3 | 37.5 ± 1.9 | 56.1 ± 5.2 | 18.6 |
| xs | × 1.5 | 39.6 ± 1.7 | 44.7 ± 2.5 | 58.8 ± 5.8 | 14.1 |
| xs | × 2 | 45.6 ± 2.1 | 45.7 ± 2.5 | 62.7 ± 3.3 | 17.0 |
| l | × 1 | 58.8 ± 1.1 | 58.5 ± 3.2 | 67.9 ± 1.1 | 9.1 |
| l | × 1.5 | 62.9 ± 2.5 | 58.3 ± 3.6 | 69.2 ± 2.2 | 6.3 |
| l | × 2 | 60.0 ± 1.9 | 60.6 ± 0.4 | 69.8 ± 1.8 | 9.2 |

**Answer.** Not up to × 2. Width raises SYMBEX's accuracy (`xs` +13 points, `l` +2-4), but INT8 rises with it, and the
gap at the same width stays about 14-19 points on `xs` and 6-9 on `l`. Sized for binary does not reach INT8 either: the
best wide binary network (`l` × 1.5, 62.9 %) is still 5 points below INT8 `l` × 1 (67.9 %), so there is no speed vs
accuracy trade to measure on the chips here, and nothing is exported. As in the earlier sections, validation runs
3-13 points above test, and `l` × 1.5 above `l` × 2 for all-binary is within that noise.

On this dataset, then, the gap is neither the optimizer (section 4) nor capacity up to × 2. What is left: the
input (3 accelerometer axes at 8 bits, the first layer's information that a 1-bit network loses early, section 3) and
the small, varied test set. HAR is reported as an open case where binary costs accuracy.

## 6. How hard is this split? A Random Forest as a yardstick (November 13)

**Question.** Daghero et al. publish 68.0 % for their max network (`l`); the same network trained here scores 58.5 % on our
test subjects. Their split is not published. Is ours harder, or is our training of the binary networks weaker?

**What.** Daghero et al. describe neither their split nor their training recipe (epochs, optimizer, subjects); they only
say the networks come from a grid search. But they also report a Random Forest on the raw values ("RF raw", up to
59 %), a model that needs no training recipe. Micucci et al. 2017, the data set paper, report the same model (AF-17,
raw data, macro average accuracy, MAA): 81.48 % with subjects mixed (5-fold) and 56.53 % leaving one subject out.
`har_rf.py` trains it on our split: trees and depth chosen on the validation subjects, then the test subjects once,
3 seeds.

**Result.** Random Forest on the raw windows (300 trees, depth 20), mean ± sample std over 3 seeds:

| | validation | test (accuracy) | test (MAA) |
|---|---|---|---|
| this split, `har_rf.py` | 65.9 ± 0.4 | **62.2 ± 0.4** | 52.0 ± 0.5 |
| Daghero et al., their split | | up to 59 | |
| Micucci et al., one subject out | | | 56.53 |
| Micucci et al., subjects mixed | | | 81.48 |

**Answer.** Our split is not harder than theirs: the same kind of model scores 62.2 % here against their 59 %, and both
are close to Micucci's subject-wise number, far from the mixed one. So Daghero et al. also held people out.

This points at our training of the binary networks, not at the data:

| | vs the Random Forest on the same split |
|---|---|
| Daghero et al., binary l (int8 first layer) | 68.0 vs 59: **+9** |
| this README, TFLite INT8 l | 67.9 vs 62.2: +5.7 |
| this README, SYMBEX l (all-binary / int8 first layer) | 58.8 / 58.5 vs 62.2: **−3.4 / −3.7** |

A binary network below a Random Forest is a warning sign. Two cautions: their 68.0 % is one network, picked from a
grid, perhaps on the same data it is scored on, so it may be optimistic; and the binary networks drop more from
validation to test (69.3 → 58.5 on `l`) than the Random Forest (65.9 → 62.2) or TFLite (73.6 → 67.9). Picking the
best of 100 noisy epochs on 4 validation subjects may explain part of that drop.

Next (future work): the binary training recipe, compared on validation and then the test once: distillation from the
float network, a lower learning rate and longer training, averaging checkpoints instead of picking one epoch; and
cross-validation over the training subjects so that the choice does not rest on 4 people.

## 7. Two training defects, fixed: the gap stays (November 13, validation only)

**Question.** Section 6 points at the binary training. Two defects were then found in it (PR #47). Were they the cause?

1. **Frozen weights.** The straight-through sign passes no gradient where a latent weight is beyond |w| > 1, and
   nothing kept the weights inside: once Adam pushed a weight past 1, its bit never flipped again. Audit of the Adam
   checkpoints above (`tools/dev/binary_health.py`, all-binary, 3 seeds): **33-38 %** of the 1-bit weights frozen in
   `xs`, 16 % in `s`, 9 % in `m`, 7-9 % in `l`.
2. **Ties in the max pool.** The network pools ±1 bits (max = OR). In a window of tied values PyTorch sends the whole
   gradient to the first element; 82-94 % of the pool windows here are tied, so most of the gradient through a pool
   went to one arbitrary position.

**What.** Two training fixes, neither changes the exported network:

- `--clip`: the latent 1-bit weights are clamped to [-1, 1] after every step (BinaryConnect). 0 % frozen.
- `--pool-first`: the BatchNorm output is pooled before the sign. sign is monotone, so sign(max(v)) = max(sign(v)):
  the same outputs as the OR the chip runs (checked: identical outputs on the same weights, `xs` and `l`, both first
  layers), but the gradient reaches the true maximum.

Both alone and together, the same recipe as Adam otherwise (100 epochs, seeds 42, 43, 44), the 4 networks, both first
layers, validation only. The rule, the same as Bop's in section 4 and fixed before looking: a fix is kept only if it
beats Adam by more than the larger of the two standard deviations; then, and only then, the test subjects are scored
once with it.

**Result.** Validation accuracy, mean ± sample std over 3 seeds (Bop γ 1e-3 from section 4 for reference; per-seed
values and the frozen-weight audit in `results/symbex/har_training_fixes.json`):

| Network | first layer | Adam | clip | pool first | clip + pool first | Bop γ 1e-3 |
|---|---|---|---|---|---|---|
| xs | binary | 44.9 ± 2.2 | 45.6 ± 3.3 | 45.9 ± 1.4 | 45.8 ± 1.4 | 40.6 ± 3.1 |
| xs | int8 | 46.1 ± 2.2 | 44.9 ± 1.3 | 47.4 ± 0.5 | 47.3 ± 0.3 | 45.1 ± 1.0 |
| s | binary | 62.8 ± 1.3 | 62.5 ± 1.0 | 63.9 ± 0.5 | 64.1 ± 0.4 | 62.9 ± 1.1 |
| s | int8 | 63.6 ± 1.7 | 64.0 ± 0.4 | 64.2 ± 1.2 | 64.3 ± 0.6 | 63.4 ± 1.0 |
| m | binary | 69.2 ± 0.6 | 68.2 ± 0.8 | 66.8 ± 1.0 | 66.5 ± 1.0 | 69.7 ± 0.5 |
| m | int8 | 67.9 ± 1.4 | 68.7 ± 1.1 | 67.6 ± 1.6 | 66.4 ± 1.1 | 69.0 ± 2.1 |
| l | binary | 68.6 ± 0.9 | 69.0 ± 0.9 | 69.0 ± 1.1 | 67.9 ± 0.7 | 71.3 ± 0.6 |
| l | int8 | 69.3 ± 0.9 | 68.9 ± 1.1 | 69.4 ± 1.0 | 69.1 ± 0.1 | 68.7 ± 1.2 |

**Answer.** No fix passes the rule, so the test subjects were not scored and nothing is exported. The closest is clip +
pool first on `s` all-binary (+1.3 against a spread of 1.3); pool first gives about +1 on the two small networks
but −2.4 on `m` all-binary. The fixes are correct (no frozen bits, gradient to the true maximum) and pooling first
makes the seeds steadier on `xs` and `s` (smaller spreads), but they do not move the accuracy. The same holds on
ResNet-8 (PR #47, clip: 70.93 ± 1.02 vs Adam 71.28 ± 0.81 on validation; int8 stem 75.33 ± 0.72 vs 75.63 ± 0.50). The binary networks of
this benchmark were not held back by these two defects.

### Where HAR stands: an open case

Measured so far on this task, with the same architectures as INT8:

| Tried | Effect on the binary networks |
|---|---|
| int8 first layer (section 3) | +4.5-4.8 on the two small networks, equal on the large ones |
| Bop (section 4) | +2.7 on l all-binary (validation); on the test subjects 59.3 vs 58.8: nothing |
| wider networks, INT8 widened the same way (section 5) | the gap stays: 14-19 points on xs, 6-9 on l |
| clipping and pooling before sign (this section) | nothing beyond the spread |

The gap to INT8 is real and none of the training levers tried closes it. On the test subjects the binary `l` network
(58.5-58.8 %) is still below a Random Forest on the raw windows (62.2 %, section 6). SYMBEX keeps its speed here
(5.3-10.4× on the ESP32, 1.47-1.90× on the S3, 3.3-7.9× on the M33, exact on every board), but on this task the
accuracy cost of going binary is high, and we report it as such: in ECG (`../ecg_mitbih`), another 1D task with the
same engine, the binary networks tie INT8.

What could be tried next, ideas not measured (each chosen on validation, the test scored once at the end):

1. **A steadier choice of checkpoint** (measured in section 8: about one point). The binary networks drop more from validation to test (69.3 → 58.5 on `l`)
   than TFLite or the Random Forest. Cross-validation over the 20 training subjects instead of 4 validation people,
   and averaging the last epochs instead of picking the best of 100, would show how much of the gap is the choice.
2. **More than one bit after the first layer** (measured in section 9: +1 to +7 on the test subjects, the gap stays). The working hypothesis (not measured): the accelerometer's amplitude
   matters for activity recognition and a 1-bit map loses it right after the first layer; ECG's shape and RR intervals
   survive it. Two-bit activations in the first binary layer (`../../experiments/pair_bits`) would test that at a
   known cost.
3. **Distillation from the float or INT8 network** (measured in section 8: within the spread on the test subjects). It gave nothing on ResNet-8 (`../../experiments/training_gap`),
   but it was never tried here.
4. **Augmentation chosen for binary.** In ECG it helped INT8 and hurt the binary networks; a light version (time shift,
   small scaling) chosen on validation for both methods alike.
5. **A network designed for binary at the same cost**, compared with INT8 resized the same way (the methodology of
   ResNet-8 and mlp_fashion), only if 1-4 leave the gap open.

## 8. Checkpoint choice and distillation (November 14, validation; test scored once)

**Question.** Ideas 1 and 3 of section 7: does a steadier choice of checkpoint, or distillation from the float network,
close the gap?

**What.** Two screenings, the same recipe as Adam otherwise (100 epochs, seeds 42, 43, 44), the 4 networks, both first
layers, the test subjects not scored:

- **Cross-validation inside the training subjects** (`har_train.py --cv-fold k`, `har_data.load_cv`): the 20 training
  subjects in 5 groups of 4; each run trains on 16 and picks its checkpoint on the other 4. The 4 validation subjects
  are scored every epoch as people outside every choice. Four ways to pick the network are compared on them: the best
  epoch on the fold (what the earlier sections do), the epoch that maximizes the mean of the 5 fold curves, the last
  epoch, and the weights averaged over the last 10 epochs. 120 runs.
- **Distillation** (`--distill T --distill-alpha a`): loss (1 − a) · cross-entropy + a · T² · KL to the FP32 network
  of the same architecture and seed (the float network `har_tflite.py` quantizes to INT8), T ∈ {2, 4},
  a ∈ {0.5, 0.9}. 96 runs.

**Result: checkpoint choice.** Accuracy on the 4 validation subjects, per seed the mean of the 5 folds, then mean ±
sample std over 3 seeds. These networks see 16 subjects, not 20, so compare the columns with each other, not with
the other sections:

| Network | first layer | best epoch on the fold | epoch of the mean CV curve | last epoch | average of the last 10 |
|---|---|---|---|---|---|
| xs | binary | 38.5 ± 1.1 | 37.3 ± 1.2 | 37.8 ± 1.3 | 35.5 ± 1.9 |
| xs | int8 | 41.1 ± 1.7 | 40.6 ± 2.1 | 37.7 ± 3.8 | 39.3 ± 2.5 |
| s | binary | 58.4 ± 1.2 | 59.7 ± 0.8 | 59.6 ± 0.9 | 59.4 ± 1.1 |
| s | int8 | 59.7 ± 0.8 | 59.7 ± 0.5 | 60.3 ± 0.4 | 59.5 ± 0.4 |
| m | binary | 62.7 ± 0.5 | 63.8 ± 0.6 | 64.0 ± 0.2 | 64.1 ± 0.8 |
| m | int8 | 62.6 ± 1.0 | 63.6 ± 0.8 | 63.4 ± 0.4 | 63.7 ± 1.0 |
| l | binary | 64.7 ± 0.6 | 65.0 ± 1.3 | 65.7 ± 0.4 | 65.5 ± 0.9 |
| l | int8 | 64.3 ± 1.2 | 65.1 ± 1.1 | 65.1 ± 1.0 | 65.3 ± 0.8 |

On the three larger networks, picking the best of 100 epochs on 4 people costs about 1 point against simply taking the
last epoch (the choice fits those 4 people); on `xs` it is the other way round. One point is real but small: it
explains little of the 9-point drop from validation to test of section 6.

**Result: distillation.** Validation, mean ± sample std over 3 seeds; ✓ = beats Adam by more than the larger standard
deviation (the rule of sections 4 and 7):

| Network | first layer | Adam | T 2, a 0.9 | T 2, a 0.5 | T 4, a 0.9 | T 4, a 0.5 |
|---|---|---|---|---|---|---|
| xs | binary | 44.9 ± 2.2 | 43.0 ± 1.4 | 44.4 ± 1.3 | 43.3 ± 0.8 | 41.8 ± 1.3 |
| xs | int8 | 46.1 ± 2.2 | 46.5 ± 1.5 | 46.1 ± 2.5 | 44.4 ± 2.4 | 44.5 ± 0.8 |
| s | binary | 62.8 ± 1.3 | 61.7 ± 1.7 | 62.5 ± 0.8 | 63.0 ± 0.5 | 61.9 ± 0.5 |
| s | int8 | 63.6 ± 1.7 | 64.7 ± 1.3 | 64.6 ± 2.2 | 64.4 ± 0.6 | 63.8 ± 0.8 |
| m | binary | 69.2 ± 0.6 | 69.2 ± 0.8 | 68.0 ± 0.3 | 68.9 ± 0.9 | 69.8 ± 1.3 |
| m | int8 | 67.9 ± 1.4 | 68.7 ± 0.3 | 68.4 ± 0.6 | **70.0 ± 0.4** ✓ | 69.5 ± 0.7 ✓ |
| l | binary | 68.6 ± 0.9 | 69.5 ± 0.6 | 68.9 ± 0.6 | 69.0 ± 0.9 | 69.2 ± 0.8 |
| l | int8 | 69.3 ± 0.9 | **70.8 ± 0.3** ✓ | 69.9 ± 0.4 | 70.6 ± 0.4 ✓ | 70.8 ± 1.7 |

Distillation never helps the all-binary networks; with the int8 first layer it gains 1.5-2.1 points on the two large
networks. **Which goes to the test subjects.** Fixed after this table and before any test score: for each network
where some configuration passes, the passing one with the best validation mean, scored once on the test subjects
(saved checkpoints, no training again). Each row picks the best of 4 configurations, so its validation gain is
flattered (section 4).

| Network (int8 first layer) | Adam, test | distilled, test | test per seed (42 / 43 / 44) | TFLite INT8, test |
|---|---|---|---|---|
| m, T 4, a 0.9 | 57.0 ± 1.5 | 58.1 ± 0.4 | 57.6 / 58.2 / 58.4 | 66.2 ± 3.2 |
| l, T 2, a 0.9 | 58.5 ± 3.2 | 58.9 ± 2.0 | 57.1 / 58.6 / 61.1 | 67.9 ± 1.1 |

**Answer.** Neither closes the gap. On the test subjects distillation gives +1.1 and +0.4 points (paired by seed:
−1.0 / +1.4 / +2.7 and +2.1 / −2.4 / +1.3), within the spread; nothing is exported. The checkpoint choice is worth
about one point on the larger networks. Per-seed values: `results/symbex/har_cv_distill.json`.

What these screenings add to the picture: on `l` the binary networks are about 4-5 points below INT8 on the
validation subjects (section 5) but 9 below on the test subjects, and they drop more from validation to test than INT8 or a Random Forest
(section 6). The noise of the checkpoint choice explains about one point of that. The rest looks like the binary
networks carrying over worse to new people on this task (not measured directly). The ideas left from section 7
(two-bit activations after the first layer, augmentation for amplitude and orientation, a network designed for binary)
are the ones that address that.

## 9. Why the binary networks carry over worse, and two-bit activations (November 14)

**Question.** Section 8 left one observation open: the binary networks drop more from validation to test than INT8
or a Random Forest. A common explanation for binary networks (ReActNet, and three literature summaries we read) is
that the fixed BatchNorm + sign thresholds break when the input statistics move, here with a new person (amplitude,
phone orientation). Is that what happens? And does more than one bit after the first layer help?

**What (a): a diagnostic, no training** (`har_shift.py`, the saved Adam networks of `har_train.py` and the float
networks TFLite quantizes, 4 networks × 3 seeds):

- the share of +1 per channel after every sign, on train, validation and test; drift = mean over channels of
  |share(split) − share(train)|;
- AdaBN: the BatchNorm statistics recomputed on the target windows, labels unused, once per split and once per
  person. Measured on the validation subjects; on the test subjects it is a diagnostic only (recalibrating on them is
  not a deployable result, and nothing here is chosen from it);
- the phone's orientation: by person, the y axis reads about +0.9 g or −1 g while walking (the phone upside down in
  the pocket; 12 people one way and 8 the other in training), and the accuracy of each group.

**Result (a).** Validation subjects, mean ± sample std, 3 seeds:

| Network | stored BN | AdaBN, all windows | AdaBN, per person | train (stored) |
|---|---|---|---|---|
| xs, binary / int8 first / float | 44.9 / 46.1 / 61.4 | 38.0 / 40.7 / 56.8 | 28.2 / 28.2 / 35.2 | 37.3 / 41.1 / 71.8 |
| s | 62.8 / 63.6 / 70.3 | 54.9 / 58.3 / 65.4 | 40.5 / 48.0 / 49.7 | 70.9 / 75.6 / 99.9 |
| m | 69.2 / 67.9 / 73.9 | 57.0 / 59.2 / 69.1 | 32.3 / 35.7 / 52.3 | 82.7 / 88.2 / 99.9 |
| l | 68.6 / 69.3 / 73.7 | 59.8 / 62.5 / 70.1 | 44.9 / 49.3 / 57.3 | 83.8 / 88.6 / 99.9 |

- **The bits barely move with new people:** per layer, the drift is 2-11 % on the validation subjects and 2-6 % on
  the test subjects, smaller on test than on validation, although the test accuracy is lower.
- **AdaBN does not help:** it lowers validation accuracy for every network, the float one too (binary −7 to −12
  points, float −2 to −5). Per person it is much worse, since each person's windows have a different mix of
  activities. On the test subjects (diagnostic only) it changes the binary networks by −0.6 to +3.4 points.
- **The orientation is not the cause:** `l`, binary, test: 60.6 % for the people with the phone one way, 57.1 % the
  other way (int8 first layer 59.8 / 57.4).
- **The OR pool does not flatten the maps:** after the OR 42-74 % of the bits are +1. The ties of section 7 come as
  much from windows of four −1 as from windows with several +1.
- **What stands out is the training fit:** the binary networks reach only 37-84 % on their own training windows
  (int8 first layer 41-89 %); the float networks 72-100 %.

**What (b): two-bit and ternary activations** (`har_train.py --acts 2|ternary`, `model.py`): after every conv, 4
levels {−1, −1/3, 1/3, 1} or 3 levels {−1, 0, 1} instead of sign (three or two thresholds per channel), weights still
1-bit, the same Adam recipe, 4 networks × 2 first layers × 3 seeds, validation only. **Rule, written before the
results:** a variant passes if its validation mean beats the 1-bit network with the same first layer by more than the
larger sample std; every variant that passes is scored once on the test subjects; the headline per network is the
passing variant with the best validation mean.

**Result (b).** Validation (✓ = passes the rule) and the fit on the training windows:

| Network | first layer | 1 bit | 2 bits | ternary | train: 1 bit / 2 bits / ternary |
|---|---|---|---|---|---|
| xs | binary | 44.9 ± 2.2 | 47.9 ± 1.9 ✓ | 46.4 ± 1.2 | 37.3 / 41.9 / 40.2 |
| xs | int8 | 46.1 ± 2.2 | 49.1 ± 1.2 ✓ | **49.6 ± 0.6** ✓ | 41.1 / 45.9 / 46.2 |
| s | binary | 62.8 ± 1.3 | 68.5 ± 1.6 ✓ | 66.5 ± 0.9 ✓ | 70.9 / 85.1 / 80.7 |
| s | int8 | 63.6 ± 1.7 | 68.2 ± 0.8 ✓ | **69.2 ± 0.7** ✓ | 75.6 / 88.8 / 85.2 |
| m | binary | 69.2 ± 0.6 | 71.2 ± 0.3 ✓ | 70.1 ± 0.3 ✓ | 82.7 / 91.8 / 88.8 |
| m | int8 | 67.9 ± 1.4 | 69.7 ± 0.1 ✓ | **71.3 ± 1.3** ✓ | 88.2 / 96.9 / 94.1 |
| l | binary | 68.6 ± 0.9 | 70.7 ± 0.6 ✓ | 71.1 ± 1.0 ✓ | 83.8 / 93.9 / 93.3 |
| l | int8 | 69.3 ± 0.9 | **71.8 ± 0.4** ✓ | 71.7 ± 0.1 ✓ | 88.6 / 96.6 / 95.8 |

Test subjects, scored once (all 15 passing variants; per-seed values in `results/symbex/har_acts.json`):

| Network | first layer | 1 bit | 2 bits | ternary | TFLite INT8 |
|---|---|---|---|---|---|
| xs | binary | 32.7 ± 3.3 | 38.8 ± 1.6 | (did not pass) | 56.1 ± 5.2 |
| xs | int8 | 37.5 ± 1.9 | 44.6 ± 1.1 | **41.5 ± 1.9** | |
| s | binary | 53.4 ± 2.4 | 57.6 ± 3.1 | 58.0 ± 0.3 | 62.4 ± 2.5 |
| s | int8 | 57.9 ± 0.4 | 59.4 ± 2.1 | **57.8 ± 3.0** | |
| m | binary | 56.8 ± 0.9 | 59.9 ± 2.3 | 59.8 ± 2.6 | 66.2 ± 3.2 |
| m | int8 | 57.0 ± 1.5 | 60.8 ± 1.2 | **58.7 ± 0.7** | |
| l | binary | 58.8 ± 1.1 | 61.8 ± 1.2 | 60.6 ± 1.5 | 67.9 ± 1.1 |
| l | int8 | 58.5 ± 3.2 | **60.8 ± 4.8** | 59.5 ± 1.8 | |

**Answer.** The fixed thresholds are not what breaks: the bits look the same on new people and recalibrating
BatchNorm does not help. The binary networks first of all do not fit their own training data. More than one bit
after the first layer fixes much of that (training fit +3 to +14 points, validation +1 to +6, test +1 to +7 on 14 of
the 15 variants; `s` int8 ternary −0.1), but the gap to INT8 stays: the variants chosen on validation (bold) are 4.6-14.6 points below
TFLite INT8 on the test subjects (1 bit, int8 first layer: 4.5-18.6). The best test value, `l` with 2 bits and a binary first layer
(61.8 %), is still below the Random Forest (62.2 %). And the drop from validation to test is unchanged (`l`, 2 bits:
70.7 → 61.8): with the fit fixed, what remains is how the networks carry over to new people. Nothing is exported:
these activations had no C kernel yet; two bits would roughly double the work of the binary layers (an estimate,
not measured; section 10 adds the kernel and measures it: 1.3-2.2×). Next, if pursued: augmentation for new people (moderate rotation, ±20°,
and amplitude scaling 0.8-1.2, the same for INT8) on top of the two-bit networks, chosen on validation.

## 10. Augmentation for new people, and the two-bit networks in C on the chips (November 14)

**Question.** Section 9 fixed most of the training fit with two-bit / ternary activations, but the drop from
validation to test stayed. (a) Does augmentation that imitates new people (another phone angle, another amplitude)
help, for the binary networks and for INT8 alike? (b) How fast are the two-bit / ternary networks on the real chips,
whole network, not an estimate?

**What (a).** `har_data.augment`: every training window rotated about a random axis by a uniform angle in ±20° and
scaled by one factor in 0.8-1.2, new draws every epoch (`har_train.py --augment`, `har_tflite.py --augment`), the
same Adam / Keras recipe otherwise. Every variant of section 9 (4 networks × 2 first layers × 1 bit / 2 bits /
ternary) and TFLite INT8, 3 seeds, validation only. **Rule, written before the results** (in the screening script):
augmentation passes if its validation mean beats the same variant without it by more than the larger sample std;
only passing variants are scored on the test subjects, once. Headline per network: the best validation mean among the
variants that have a test score (without augmentation, or passing with it), against the INT8 one chosen the same way.

**Result (a).** Validation, without → with augmentation (mean ± sample std, 3 seeds; ✓ = passes):

| Network | first layer | 1 bit | 2 bits | ternary |
|---|---|---|---|---|
| xs | binary | 44.9 → 43.9 | 47.9 → 50.2 ✓ | 46.4 → 45.5 |
| xs | int8 | 46.1 → 47.4 | 49.1 → **52.1 ± 0.2** ✓ | 49.6 → 49.4 |
| s | binary | 62.8 → 62.3 | 68.5 → 68.2 | 66.5 → 67.1 |
| s | int8 | 63.6 → 65.9 ✓ | 68.2 → 68.8 | **69.2 ± 0.7** → 67.6 |
| m | binary | 69.2 → 68.4 | 71.2 → 70.5 | 70.1 → 70.2 |
| m | int8 | 67.9 → 68.8 | 69.7 → 69.2 | **71.3 ± 1.3** → 69.4 |
| l | binary | 68.6 → 69.4 | 70.7 → 71.6 ✓ | 71.1 → 71.7 |
| l | int8 | 69.3 → 70.9 ✓ | **71.8 ± 0.4** → 72.1 | 71.7 → 72.1 |
| TFLite INT8 | | xs 61.1 → 60.9, s 70.2 → 71.9 ✓, m 73.7 → 74.2, l 73.6 → 73.4 | | |

Test subjects of the passing variants, scored once (without → with augmentation; per seed in
`results/symbex/har_augment.json`): `xs` binary 2 bits 38.8 → 39.3, `xs` int8 2 bits 44.6 → 45.7 ± 3.4, `s` int8
1 bit 57.9 → 58.2, `l` binary 2 bits 61.8 → 63.0 ± 3.8, `l` int8 1 bit 58.5 → 64.8 ± 2.4, TFLite `s` 62.4 → 64.6 ± 2.9.

Headline (test, the variants chosen on validation, bold above):

| Network | binary (chosen) | TFLite INT8 (chosen) | gap | gap in section 9 |
|---|---|---|---|---|
| xs | int8 first, 2 bits, augmented: 45.7 ± 3.4 | 56.1 ± 5.2 | 10.4 | 14.6 |
| s | int8 first, ternary: 57.8 ± 3.0 | augmented: 64.6 ± 2.9 | 6.8 | 4.6 |
| m | int8 first, ternary: 58.7 ± 0.7 | 66.2 ± 3.2 | 7.5 | 7.5 |
| l | int8 first, 2 bits: 60.8 ± 4.8 | 67.9 ± 1.1 | 7.1 | 7.1 |

**What (b).** A C kernel for these activations, as an experiment of this folder (`planes1d.h`, embedded by
`acts1d.py` into `model_symbex.h`; `har_export.py` uses it for `--acts` runs). No new HAL function: a map holds two
bit planes, and a binary layer counts each plane with lib_symbex's row functions, the same ones as the 1-bit layers
(2 bits: plane = high / low bit, raw = 2 (n − 2 d_high) + (n − 2 d_low) from two XOR counts; ternary: plane = "+1" /
"−1", raw from two AND counts). Each output takes the number of its ascending thresholds raw reaches (3 or 2, folded
from BatchNorm like the 1-bit threshold); pooling = max of raw before the thresholds. First layer: the int8 one of
section 3 (PIE on the S3). Integer simulation = PyTorch on every validation and test window (fidelity 100 %), C on the
PC = simulation (100 %). On the chips: the four networks chosen in (a), seed 42, weights in flash, the flags of the
1-bit rows of section 2 (`chip_test.py`), 1,000 validation windows.

**Result (b).** Median ms, every run equal to the integer simulation on 100 % of the windows (in brackets: × the 1-bit
network with the same int8 first layer; TFLite INT8 of the same architecture, flash):

| Network | ESP32 | ESP32-S3 | Pico 2 M33 | Pico 2 Hazard3 |
|---|---|---|---|---|
| xs (2 bits) | 1.06 (1.27×) · TFLite 4.17 | 0.74 (1.53×) · 0.72 | 1.19 (1.42×) · 2.71 | 0.95 (1.71×) · 17.3 |
| s (ternary) | 6.51 (1.65×) · 31.2 | 2.96 (1.61×) · 2.91 | 5.43 (1.48×) · 12.0 | 4.75 (1.83×) · 226 |
| m (ternary) | 17.8 (1.80×) · 114 | 7.07 (1.88×) · 7.14 | 17.6 (1.78×) · 78.0 | 12.9 (2.16×) · 1160 |
| l (2 bits) | 21.6 (1.66×) · 116 | 6.68 (1.74×) · 7.11 | 18.3 (1.55×) · 77.7 | 14.7 (1.71×) · 1134 |

The earlier estimate (`har_acts_speed.py`, proxy networks) was close on the M33 and below the real time elsewhere;
both are in `results/symbex/har_acts_speed.json`. On the Hazard3 the kernel uses the HAL row functions, not the inline
two-window path of the 1-bit layers, so its numbers may be a little pessimistic there.

**Answer.** (a) Augmentation helps a few variants on validation (5 binary, 1 INT8); on the test subjects most gains
are within the spread. The chosen binary networks stay 6.8-10.4 points below TFLite INT8; `xs` gains the most (the
gap shrinks from 14.6 to 10.4). One unchosen variant stands out on the test subjects, `l` int8 first layer 1 bit
augmented (64.8 %), but validation did not pick it, so it is not a result. (b) Two bits or ternary cost 1.3-2.2× the
1-bit time, exact on every chip. On the ESP32, the M33 and the Hazard3 the networks stay 2.2-90× faster than TFLite
INT8; on the S3 they tie it (0.94-1.04×), so there the extra accuracy costs the speed advantage. Next, if pursued: a
row function that counts both planes in one pass (one load of the weights instead of two), and the vertical engine in
1D for the S3.

## Decisions (and why)

- **Binary vs binary is the main comparison:** the question this benchmark answers is whether SYMBEX's binary layers
  beat the binary library built for this task, on its own processor, with its own code and networks. INT8 is the
  reference that anyone deploying a network would use.
- **Their code is unchanged, with one exception:** the thresholds of their int8 first layer are `int16_t`, but int8 ×
  int8 products over 15 × 3 taps reach ±58,000 in the trained networks (`l`). In the copied sources they become
  `int32_t` (on RV32 an int32 load is the same instruction). Their pooled kernels get `dim_out` = the steps the pool
  uses.
- **Each library runs its first layer as designed:** ultracompactBNN with int8 weights, SYMBEX with 1-bit weights.
  Both networks are trained with the same recipe. SYMBEX also runs their network (int8 first layer), so the strict
  comparison (same network, same precision per layer) is there too.
- **Bits or bytes for the first layer is a choice per chip, not per network:** the same 1-bit weights and thresholds,
  exported both ways, give the same outputs. With the S3's PIE kernel, the bytes are now the fastest form on every
  chip measured (on the M33, `l`, the two are within 1 %); an automatic choice in the CDT should still measure both.
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
few output channels were slower than ultracompactBNN (0.47× on `xs`'s 4 → 4 k7 layers); the current kernel for those
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
| `model.py` | the four networks of Table 1, binary; first layer `binary` (1-bit weights) or `int8`; `acts` 1 / 2 / ternary (section 9, PC only) |
| `har_train.py` | trains one network, or `--all`: 4 networks × 2 first layers × 3 seeds; screening options `--optimizer bop`, `--clip`, `--pool-first`, `--distill`, `--cv-fold`, `--acts`, `--augment` (sections 4, 7, 8, 9, 10), `--no-test`, `--score-test` |
| `har_shift.py` | the diagnostic of section 9: bit shares per channel, AdaBN, phone orientation (no training) |
| `har_export.py` | integers, simulation, `model_symbex.h`, the C check on the PC (EXACTNESS 100 %); `--first-bytes`: the 1-bit first layer as ±1 bytes, in `runs/<run>_bytes/`; `--acts` runs through `acts1d.py` |
| `acts1d.py`, `planes1d.h` | section 10: the two-bit / ternary networks into integers and C (two bit planes per map on lib_symbex's HAL row functions, several thresholds per channel) |
| `har_acts_speed.py` | section 10: the speed estimate from proxy networks, and the table of the measured two-bit / ternary networks |
| `har_gvsoc.py` | the whole networks through both libraries: `--pc` (gcc, exactness) or GVSoC (instructions, cycles per layer) |
| `pulp/symbex_u8_pulp.c` | SYMBEX's first layer for PULP (`p.sdotusp4`), 1-bit or int8 weights |
| `har_tflite.py` | (`--augment`: section 10) the same architectures in Keras → INT8 → TFLite Micro on the ESP32 / ESP32-S3 (`--weights ram`) and the Pico 2 (`--board pico2_arm`, `pico2_riscv`) |
| `har_lowbit.py` | INT8 / INT4 / INT2 baselines with `tools/lowbit`, from the same float networks |
| `ucbnn_check.py`, `ucbnn_gvsoc.py` | the single-layer checks and measurements (section "Earlier") |
| `pulp/Dockerfile` | PULP GCC, PULP SDK and GVSoC (pulp-open) on Ubuntu 22.04 |
| `../../experiments/chip_kernels/int8_first_bench.py` | the quick check of section 3: an int8 first layer for mlp_fashion and ResNet-8 on the S3 |

## Reproduce

**Without training:** [`reference/`](reference/README.md) holds seed 42 of the `l` network, fully binary, ready to compile or
measure, next to its INT8 model.

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

# Section 9 (validation; the passing variants scored once with --score-test)
python har_shift.py
python har_train.py --arch max --first int8 --acts 2 --seeds 42,43,44 --no-test

# Section 10 (augmentation: validation, passing variants scored once; the two-bit networks on the chips)
python har_train.py --arch max --first int8 --acts 2 --augment --seeds 42,43,44 --no-test
python har_tflite.py --arch max10 --seed 42 --augment --no-test
python har_export.py runs/max_int8_e100_a2_s42
python ../../../tools/chip_test.py runs/max_int8_e100_a2_s42 --board s3 --port /dev/ttyACM0 --usb --popcount swar
python har_acts_speed.py report

# The quick check of section 3 (ESP32-S3)
python ../../experiments/chip_kernels/int8_first_bench.py --port /dev/ttyACM0
```

Results: `results/symbex/` (SYMBEX on GVSoC, accuracy), `results/ucbnn/` (ultracompactBNN on GVSoC), `results/tflite/`,
`results/lowbit/`, `runs/<run>/chip_<board>_<popcount>_<weights>_mask.json` (SYMBEX on the boards).

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
8. The rest of the ESP32 grid (`swar` with the weights in RAM), energy on the other boards (classic ESP32 done: [ENERGY.md](ENERGY.md), 8.7× less energy per window).
