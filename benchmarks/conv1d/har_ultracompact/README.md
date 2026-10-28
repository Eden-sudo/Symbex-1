# har_ultracompact — binary 1D convolutions for human activity recognition

Human activity recognition from a phone's accelerometer (UniMiB-SHAR, 17 activities) with **binary 1D convolutional
networks**. It compares two binary libraries first: SYMBEX and the closest prior work, the binary inference library of
**Daghero et al., "Ultra-compact Binary Neural Networks for Human Activity Recognition on RISC-V Processors"** (ACM
Computing Frontiers 2021, `github.com/francescodaghero/ultracompactBNN`). The comparison uses their four networks,
their processor and their own code. INT8 is the reference: TensorFlow Lite Micro and the INT8 / INT4 / INT2 baselines
of `tools/lowbit`, with the same architectures, measured on real boards.

## Results in short

- **Binary vs binary, in their own environment** (RI5CY core in GVSoC, their code unchanged): SYMBEX needs **1.07-1.37×
  fewer instructions** than ultracompactBNN on all four networks of their Table 1 (3 seeds). This holds with **their
  own trained network** too (same int8 first layer, same weights, same accuracy): **1.06-1.37×**. Every output is exact:
  equal to the integer simulation, window by window.
- **Accuracy, binary vs binary:** equal on the two large networks (58.8 % vs 58.5 %, 56.8 % vs 57.0 %). On the small
  ones, an int8 first layer gives 4-5 points, and SYMBEX can run that layer too, at the same speed.
- **Against INT8 on real boards** (TFLite Micro, the same architectures, weights in flash): on the ESP32, SYMBEX is
  **1.8-7.5× faster** all binary and **5.4-10.4×** with the int8 first layer; on the ESP32-S3, **1.2-1.5× faster** all
  binary on 3 of the 4 networks (0.86× on min, where the int8 first layer gives 1.07×). SYMBEX takes **2-10× less RAM**
  and **7.5-10× less flash**. INT8 is more accurate on these small networks: about 9 points on the large ones, 23 on
  the smallest.
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
first layer costs about the same in both libraries, with 1-bit or int8 weights.

### Accuracy (test subjects, integer simulation = what the chip runs; mean ± std, 3 seeds)

| Network | SYMBEX, all binary | ultracompactBNN (int8 first layer) | Daghero et al. published* |
|---|---|---|---|
| min | 32.7 ± 2.7 % | 37.5 ± 1.6 % | 33.0 % |
| max-10% | 53.4 ± 1.9 % | 57.9 ± 0.3 % | 58.6 % |
| max-5% | 56.8 ± 0.7 % | 57.0 ± 1.2 % | 65.1 % |
| max | 58.8 ± 0.9 % | 58.5 ± 2.6 % | 68.0 % |

\*Their split is not published, so their numbers are not directly comparable. SYMBEX with their network (int8 first
layer) has their accuracy by construction: it is the same network, run exactly.

## 2. Reference: INT8 on real boards

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

| Network | ESP32: SYMBEX all binary | SYMBEX int8 first layer | TFLite INT8 | best SYMBEX vs TFLite |
|---|---|---|---|---|
| min | 2.37 | **0.77** | 4.17 | **5.4×** |
| max-10% | 5.18 | **4.00** | 31.23 | **7.8×** |
| max-5% | 15.55 | **10.98** | 114.22 | **10.4×** |
| max | 15.36 | **14.07** | 115.62 | **8.2×** |

| Network | ESP32-S3: SYMBEX all binary | SYMBEX int8 first layer | TFLite INT8 | best SYMBEX vs TFLite |
|---|---|---|---|---|
| min | 0.84 | **0.67** | 0.72 | **1.07×** |
| max-10% | **2.41** | 2.91 | 2.93 | **1.21×** |
| max-5% | **4.88** | 5.93 | 7.14 | **1.46×** |
| max | **4.88** | 8.70 | 7.11 | **1.46×** |

| Network | Pico 2 Hazard3: all binary | int8 first layer | Pico 2 M33: all binary | int8 first layer |
|---|---|---|---|---|
| min | 5.45 | **0.56** | 2.13 | **0.84** |
| max-10% | 6.90 | **2.59** | 4.83 | **3.66** |
| max-5% | 10.60 | **6.00** | 11.36 | **9.90** |
| max | 11.36 | **8.59** | 12.37 | **11.81** |

SYMBEX counting modes: `swar` on the ESP32 and the ESP32-S3 (no lookup table: 1.3-2.8 KB of RAM; `lut16` is up to 8 %
faster on the ESP32, for 64 KB more, and gives the same speed on the S3), native `cpop` on the Hazard3, `lut16` on the M33 (`swar`:
2.10 / 5.57 / 15.18 / 15.98 ms all binary, 2.7 KB of RAM instead of 67 KB). TFLite Micro on the Pico 2 is future work.

### Latency, weights in RAM (median, ms)

| Network | ESP32: all binary (lut16) | int8 first layer (lut16) | TFLite INT8 | S3: all binary | int8 first layer | TFLite INT8 |
|---|---|---|---|---|---|---|
| min | 2.42 | 0.79 | 3.74 | 0.84 | 0.66 | 0.71 |
| max-10% | 5.03 | 3.74 | 28.72 | 2.42 | 2.91 | 2.93 |
| max-5% | 14.12 | 9.33 | 100.97 | 4.88 | 5.93 | 6.40 |
| max | 14.00 | 12.30 | 102.70 | 4.89 | 8.70 | 6.16 |

TFLite with the weights in RAM copies the whole model at boot. On the Pico 2, the weights in RAM change SYMBEX's
latency by under 10 %.

### Memory (max network)

| | SYMBEX all binary | SYMBEX int8 first layer | TFLite INT8 |
|---|---|---|---|
| Flash (weights + thresholds) | **5.6 KB** | 6.5 KB | 42.5 KB |
| RAM (no lookup table) | **2.4-2.7 KB** | 2.4-2.7 KB | 10.3 KB (ESP32) / 25.5 KB (S3) of arena |

Smallest network (min): SYMBEX 0.7 KB of flash, 1.3-1.6 KB of RAM; TFLite 7.3 KB of flash, 2.7-2.9 KB of arena.

### Notes

- **The ESP32-S3 has SIMD on both sides:** TFLite runs on ESP-NN, which uses the S3's PIE vector unit for its int8
  products; SYMBEX uses it to count bits. The classic ESP32 has no SIMD for either: there SYMBEX is 1.8-10.4× faster.
- **The first layer decides the S3:** SYMBEX's int8 first layer is still a generic C kernel (no PIE, no SMLAD), so on
  the max network of the S3 TFLite is faster with that variant (7.11 vs 8.70 ms); all binary, SYMBEX is faster
  (4.88 ms). The kernels for each chip (PIE on the S3, SMLAD on the M33) are the next step.
- **The int8 first-layer kernel is faster than the current 1-bit one on the boards** (Hazard3, min: 0.45 vs 5.35 ms
  for that layer): the 1-bit first layer pays a fixed cost per time step (window copy, input sum, bit planes). Running
  1-bit weights as ±1 bytes through the int8 kernel gives the same result with that speed: future work, below.

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
| `har_export.py` | integers, simulation, `model_symbex.h`, the C check on the PC (EXACTNESS 100 %) |
| `har_gvsoc.py` | the whole networks through both libraries: `--pc` (gcc, exactness) or GVSoC (instructions, cycles per layer) |
| `pulp/symbex_u8_pulp.c` | SYMBEX's first layer for PULP (`p.sdotusp4`), 1-bit or int8 weights |
| `har_tflite.py` | the same architectures in Keras → INT8 → TFLite Micro on the ESP32 / ESP32-S3 (`--weights ram`) |
| `har_lowbit.py` | INT8 / INT4 / INT2 baselines with `tools/lowbit`, from the same float networks |
| `ucbnn_check.py`, `ucbnn_gvsoc.py` | the single-layer checks and measurements (section "Earlier") |
| `pulp/Dockerfile` | PULP GCC, PULP SDK and GVSoC (pulp-open) on Ubuntu 22.04 |

## Reproduce

```
cd benchmarks/conv1d/har_ultracompact                       # environment env_bench
python har_data.py                                          # needs UniMiB-SHAR in ../../data/unimib_shar/
python har_train.py --all
for r in runs/*_e100_s4?; do python har_export.py $r; done
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
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_binary_e100_s42 --board pico2_riscv --port /dev/ttyACM0
python chip_test.py ../benchmarks/conv1d/har_ultracompact/runs/max_binary_e100_s42 --board pico2_arm --port /dev/ttyACM0 --weights ram
```

Results: `results/` (GVSoC, accuracy, TFLite, lowbit), `runs/<run>/chip_<board>_<popcount>_<weights>_mask.json` (SYMBEX
on the boards).

## Future work

1. **The 1-bit first layer through the int8 kernel** (±1 as bytes): the same result, the speed of the int8 kernel.
2. **First-layer kernels per chip:** PIE on the ESP32-S3 (where TFLite's ESP-NN wins that layer), SMLAD on the M33,
   with a running sum of the input where the instruction has no sign.
3. **Early exit in the OR pool, per channel:** measured on the PC, half of the steps of the first binary layer can be
   skipped (a channel that already gave 1 in its pool window); it trades against the 4-channel blocks, to be measured.
4. **The binary layers on chips without a native bit count** (ESP32, S3, M33): the blocked kernels of the RI5CY.
5. **TFLite Micro on the Pico 2** (CMSIS-NN on the M33), and the rest of the ESP32 grid (`swar` with the weights in
   RAM), energy per inference (INA226).
6. **Networks sized for binary** (wider binary layers), compared with INT8 sized the same way, as in ResNet-8 and
   mlp_fashion: the way to close the accuracy gap.
