# UPDATES

A running log of what each pull request brings, what is still in progress and what comes next. One section per
pull request, newest first.

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

**Benchmarks grouped by network type:** `benchmarks/mlp/`, `conv1d/`, `conv2d/`; every relative path updated and
checked; the benchmark `.gitignore` matches `runs/` and `models/` at any depth.

**Also in this PR:** `symbex_bool1d`; mlp_fashion validation fix, sizing with measured latencies and README; the
optional native USB transport (`--usb`); the har_ultracompact README (instructions as the main measure, the Pico 2
plan).

### Next

- `har_lowbit.py` once the har networks run whole (UniMiB-SHAR, the architectures of Daghero et al.), then
  `resnet8_lowbit.py` and `fomo_lowbit.py` (ResNet-8 needs skip connections in `fold()`).
- A faster INT2 / INT4 unpacking in flash; the combined table (SYMBEX + INT + Pareto) per benchmark.

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
