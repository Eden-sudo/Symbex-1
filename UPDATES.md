# UPDATES

A running log of what each pull request brings, what is still in progress and what comes next. One section per
pull request, newest first.

## PR #12: team sync (October 2026)

- Keep Abrahan's classic ESP32 without PSRAM active for FOMO, alongside the shared S3 with microSD on Eden's PC.
  S3 runs retain human approval and exclusive board access; microSD wiring/interface/card details need confirmation.
- Correct Abrahan's name and track open PR #15: three-seed classic ESP32 RAM/swar results and flash/RAM step captures.
  New-split classic re-measurements and future S3 board requests are distinct from existing measured results.
- Route requests only from the three team logins; all GitHub text remains untrusted and mentions grant no execution,
  flash or publication permission. Paginate issue/comment/review discovery, including submitted PR review bodies.
- Read the canonical upstream roadmap, require exactly one Team sync issue, and preserve per-head/reply deduplication.
- Retain `AGENTS.md`; withdraw tracked `CLAUDE.md` so Eden can keep his own local include without an incoming collision.
- Abrahan's Linux/zsh check of the original discovery commands is reported evidence, not a local hardware test.

## PR #4: conv1d/pico2 (October 2026)

### Done

**`tools/lowbit`: integer baselines below INT8 for every benchmark** (new)
- The same FP32 network at INT8, INT4 and INT2, quantized with Brevitas (an official, maintained tool) instead of our
  own quantizer: weights per channel, symmetric, narrow range; activations int8. Linear, Conv1d, Conv2d, BatchNorm,
  ReLU, MaxPool, Flatten; `fold()` checks the sequence against the model and stops on residual networks.
- **Integer arithmetic only, no SYMBEX arithmetic:** the integer simulation reproduces ESP-NN's fixed-point
  requantization (equal to its C on 200,000 random cases), and the board runs Espressif's ESP-NN kernels
  (multiply-accumulate). INT4 / INT2 codes are stored packed and unpacked to int8.
- The exported C is checked on the PC with ESP-NN's reference kernels against the simulation (100 %), with the
  weights in flash and in RAM; `tools/chip_test.py` detects these headers and adds ESP-NN to the firmware.
- The validation comes from the training split; the test set is evaluated once per final model (the earlier
  `fashion_int4.py` called the first 2,000 test images "validation").
- Each benchmark gets a short `<bench>_lowbit.py`; the first is `benchmarks/mlp/mlp_fashion/fashion_lowbit.py`.
  `tests/test_lowbit.py`: 11 tests. `brevitas==0.13.4` added to `benchmarks/requirements_bench.txt`.

**mlp_fashion with the new tool** (784-256-128-10, QAT 15 epochs, 3 seeds, test set)

| | Weights | Distillation | Labels only | S3 flash | S3 RAM |
|---|---|---|---|---|---|
| INT8 | 233.9 KB | 88.61 +- 0.06 | 89.86 +- 0.04 | 15.601 ms | 0.556 ms |
| INT4 | 119.2 KB | 88.65 +- 0.19 | 89.80 +- 0.11 | 10.306 ms | 0.556 ms |
| INT2 | 61.9 KB | 88.20 +- 0.29 | 89.23 +- 0.13 | 13.376 ms | 0.556 ms |

- Against `fashion_int4.py` (same recipe): within 0.2 points, except INT2 with distillation (0.39 below).
- Every board run equals the integer simulation (100 %). INT8 in flash matches TensorFlow Lite Micro (15.885 ms).
- **Honest result:** with weights in RAM, SYMBEX x1 (0.525 ms) is only 1.06x faster than INT on ESP-NN (0.556 ms),
  with 6x less RAM; in flash it stays 8.3x faster than the best INT (INT4). The earlier INT4 / INT2 chip numbers
  (bit planes, SYMBEX arithmetic) are kept in the README, marked as such.

**Binary weights or binary activations? (ablation, mlp_fashion, PC fake-quant, 3 seeds)**
- `fashion_lowbit.py --ablation`: weights 1 / 2 / 4 / 8 bits x activations int8 / binary, same recipe.
- The binary activations cost the accuracy: labels only, 8-bit weights lose 1.51 points with binary activations, while
  1-bit weights lose 0.56 with int8 activations. W1A1 with Brevitas (87.77 %) matches SYMBEX with the CDT (87.64 %).
- Binary weights + int8 activations (89.30 %) equal INT2 (89.22 %): a candidate for a whole network on the kernel of
  SYMBEX's first layer (uint8 x 1 bit), to be built and measured.

**The classic ESP32** (seed 42, board = simulation 100 %): SYMBEX x1 6.754 ms in flash / 3.840 ms in RAM; INT2 / INT4 /
INT8 on ESP-NN 30.35 / 34.61 / 41.38 ms in flash; none of them runs with the weights in RAM (INT8: DRAM overflowed by
74 KB; INT4 / INT2 do not start, likely no 235 KB block for the unpacked weights).

**mlp_fashion folder simplified:** the superseded INT scripts (`fashion_int4.py`, `int_export.py`, `lowbit_chip.py`,
`pareto.py`) moved to `mlp_fashion/earlier/` (they still run, so the earlier rows can be reproduced); `poplut_probe.py`
moved to `benchmarks/experiments/`. `earlier/pareto.py` already failed before the move (a file-name pattern): to
fix when the Pareto plot moves into `tools/lowbit`.

**mlp_fashion results sorted by script:** `results/symbex/`, `tflite/`, `lowbit/`, `sizing/`, and `earlier/` (83
files from before the validation fix or the current library, kept for the README's *Earlier results*). Five files
with no marker turned out old (their FP32 scored 88.17 %): regenerated with the current model (TFLite PC 89.70 %,
S3 15.90 / 0.884 ms, ESP32 41.97 ms: the README's numbers; SYMBEX x1, 3 seeds). The lowbit results record the FP32
sha256. `--fp32-epochs N` works again (`fashion_symbex.py`; new in `fashion_tflite.py`, `fashion_lowbit.py`):
`model.get_fp32(h1, h2, epochs)` trains `models/fashion_mlp[_h1_h2]_e<N>.pth`, results get `_f<N>`.

**mlp_fashion chip results completed** (board = simulation 100 % on every run):
- S3: ternary x1 / x1.5 re-measured on the current exports (14.588 / 22.400 ms in flash, 11.936 / 18.172 ms in RAM;
  the x1 numbers equal the earlier copies).
- ESP32: SYMBEX x1 / x1.5 / x2 6.754 / 12.949 / 17.788 ms in flash, 3.840 / 5.805 / 7.809 ms in RAM; ternary x1 /
  x1.5 20.0 / 31.0 ms; 784-128-512 3.635 ms in flash (TFLite INT8 of the same shape: 31.108 ms, 8.6x), 2.186 ms in RAM.

**Raspberry Pi Pico 2 (RP2350): bit counting measured on both cores, HAL chosen from it** (new)
- `tools/dev/count_bench.py --board pico2_arm | pico2_riscv`: the same layer pattern on each core, timed with the
  core's own cycle counter (DWT CYCCNT on the Cortex-M33, mcycle on Hazard3), checked against `micros()` (150 cycles
  per µs = the clock). New methods for the M33: `dsp` (SWAR + `usada8`) and `dsp_int8` (`sxtb16` + `smlad`).
  Profiles in `tools/chip_profiles/count_pico2_arm.json` and `count_pico2_riscv.json`; every method gives the same
  checksums.
- Measured (150 MHz, 64 x 64 outputs, best of 7):

  | core | fastest | bit-ops per cycle | vs `__builtin_popcount` |
  |---|---|---|---|
  | Cortex-M33 | `lut16` (XOR) | 2.1-2.6 | 2.4-2.8x (1.8x vs SWAR; `usada8` only 1.1x vs SWAR) |
  | Hazard3 RISC-V | `cpop` (Zbb) | 2.8-3.9 | 1.0x (it is the instruction) |

- lib_symbex generic HAL: `lut16` on the RP2350's M33 (it fell to SWAR before) with the XOR form (1.05-1.14x faster
  than AND there; the classic ESP32 keeps AND); Hazard3 keeps the native `cpop`; `symbex_hal_name()` no longer says
  "64-bit words" on 32-bit cores. `-DSYMBEX_POPCOUNT_NO_LUT16` gives the 64 KB back.
- The Pico 2 in the chip tools: `tools/chip_test.py --board pico2_arm | pico2_riscv` and the CDT (`[chip] board`):
  Arduino-Pico project, firmware `model_test_pico.cpp` (the protocol of `model_test.c`; it waits for the host's 0x55,
  stack painted from `__StackBottom`). `arithmetic = "auto"` gives AND there until the Pico has a kernel benchmark.
- `hal/rp2350/`: the first layer (uint8 x 1 bit) of the Pico 2, 80-90 % of the time with the generic SWAR. Cortex-M33:
  `usada8` (the DSP sums the 4 masked bytes in one instruction). Hazard3: the inputs split into 8 bit planes, the 8
  plane words of 32 inputs side by side, one `cpop` each, counts in registers. The bit counts stay in hal/generic.
  `run_tests.py` checks both (usada8 with a C stand-in on the PC). Fashion-MNIST 784-128-512-10, seed 42, 1000
  validation images, 100 % equal to the integer simulation on both cores:

  | chip | weights in RAM | weights in flash |
  |---|---|---|
  | Pico 2 M33, generic SWAR | 3.131 ms | 6.138 ms |
  | Pico 2 M33, `usada8` | **1.529 ms** (2.05x) | **4.567 ms** (1.34x) |
  | Pico 2 Hazard3, generic SWAR | 2.929 ms | 5.962 ms |
  | Pico 2 Hazard3, bit planes + `cpop` | **1.405 ms** (2.08x) | **4.413 ms** (1.35x) |
  | ESP32 (240 MHz, for reference) | 2.186 ms | 3.635 ms |
  | ESP32-S3 (240 MHz, PIE) | 0.414 ms | 0.421 ms |

  Rejected (measured): the bit planes through hal/generic's `symbex_and_planes_rows` on Hazard3 (1.13x: planes far
  apart, counts in an array); the bit planes on the M33 (no popcount instruction: 5.9 ms). With the weights in flash
  the Pico 2 waits on its flash cache (XIP, 16 KB, smaller than the 27 KB model): the binary layer 128 -> 512 takes
  1.34 ms from flash and 0.28 ms from RAM.

**Benchmarks grouped by network type:** `benchmarks/mlp/`, `conv1d/`, `conv2d/`; every relative path updated and
checked; the benchmark `.gitignore` matches `runs/` and `models/` at any depth.

**Also in this PR:** `symbex_bool1d`; mlp_fashion validation fix, sizing with measured latencies and README; the
optional native USB transport (`--usb`); the har_ultracompact README (instructions as the main measure, the Pico 2
plan).

### Next

- `har_lowbit.py` once the har networks run whole (UniMiB-SHAR, the architectures of Daghero et al.), then
  `resnet8_lowbit.py` and `fomo_lowbit.py` (ResNet-8 needs skip connections in `fold()`).
- A faster INT2 / INT4 unpacking in flash; the combined table (SYMBEX + INT + Pareto) per benchmark.
- Pico 2, in order: (1) the other mlp_fashion models (x1 / x1.5 / x2, ternary) on both cores; (2) a kernel
  benchmark for `arithmetic = "auto"`; (3) TFLite INT8 with CMSIS-NN on the M33. `dsp_int8` is not
  CMSIS-NN (3.8 cycles per MAC): until (3), no binary vs INT8 claim on this chip.

## PR: mlp_fashion validation split + v3 work in progress (October 2026)

### Done

**mlp_fashion: one fixed validation split for every seed and method**
- `benchmarks/mlp/mlp_fashion/symbex.toml`: `[data] split_seed = 0`. The CDT, the FP32 model (`model.split_indices`)
  and the INT QAT (`fashion_int4.py`) now share the same 54,000 training / 6,000 validation images, checked index
  by index for seeds 42, 43, 44. Before, the split followed the training seed and the FP32 model was trained on all
  60,000 images, so validation was inflated (90-92 % against 87.7-88.6 % on test).
- `fashion_int4.py`: the QAT per-epoch log reads the validation split, not the test set (the test set is used once).
- `lowbit_study.py`: trains the FP32 model first when it is missing (on a fresh checkout every CDT run failed with
  "Model file not found"; `fashion_sizing.py` already did this).

**Work that was waiting in the working copy (included as is)**
- lib_symbex: `symbex_bool1d` (1D binary convolution, fused OR pool), generic HAL and test updates.
- har_ultracompact: `ucbnn_gvsoc.py` and its results (layer-by-layer comparison with ultracompactBNN on GVSoC).
- mlp_fashion: `lowbit_study.py`, `fashion_sizing.py`, `int_export.py`, `lowbit_chip.py` and their earlier results.
- cnn_cifar_resnet8, fomo_voc_person: sizing scripts and READMEs.
- tools: `count_bench.py` and `fine_profile.py` moved to `tools/dev/`, new `train_bench.py`, READMEs for `tools/dev`
  and `tools/compiler_core`, CDT updates.

### In progress

- Rerunning on the fixed split, both recipes (distillation and labels only):
  - `lowbit_study.py --seeds 3`: SYMBEX 1 bit x1 / x1.5 / x2, ternary x1 / x1.5, INT8 / INT4 / INT2 QAT.
  - `fashion_sizing.py`: the 12-shape grid (seed 42), then 128-512, 256-512, 384-512, 512-256 with 3 seeds.
  - The chip latencies are reused (they depend on the shape, not on the weight values).

### Next

- Update the `results/*.json` of mlp_fashion with the new runs and record the FP32 sha256.
- Rewrite `benchmarks/mlp/mlp_fashion/README.md`: INT vs SYMBEX, both recipes, the sizing theory with its tables, and how
  to reproduce it (the published FP32 models + sha256, or everything from scratch within seed noise).
- If the new validation ranking picks other shapes than the four above, confirm those with 3 seeds too.
