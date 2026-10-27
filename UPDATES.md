# UPDATES

A running log of what each pull request brings, what is still in progress and what comes next. One section per
pull request, newest first. The plan as a whole lives in CLAUDE.md ("Next, in order").

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
- `CLAUDE.md`: how to work on the project (methodology, layout, commands, what is next).

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
