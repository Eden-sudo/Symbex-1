# CLAUDE.md — working on SYMBEX-1

SYMBEX-1 compiles neural networks into **fully binary networks for microcontrollers** (lib_symbex, a C engine; the CDT,
a compiler in Python) and measures them on real chips against the official INT8 toolchain (TensorFlow Lite Micro).
Author: Carlos Duarte (Edén), Panama. Contributor: Obed (GitHub Obed0101).

## Talking with Edén

- **Spanish**, simple and short. He does not read long answers: a table, the result, the next command.
- Direct, honest technical feedback. If a result is bad for SYMBEX, say it plainly.
- **Never present an estimate as a fact**: say "estimate" and verify it with a measurement. He insists on this.
- He copies file contents into his folders (no zips, no downloads): one self-contained file per change.
- Do not recommend buying hardware: use what he has.

## Rules for the code

- **Everything in English**: names, comments, messages, docstrings (the documentation will be bilingual).
- **Never remove functionality** from an existing script: improve, fix, add.
- **The repository is the only source of truth.** Read the current file before changing it. Change existing files with
  edits that check the exact text first and stop if it is not there; never rewrite a file from memory.
- Short scripts with one purpose each (train, measure, a study), one parameterized script instead of near-copies.
- Functions used together live in the same module.
- Benchmark-specific code stays in its benchmark folder; lib_symbex only gets what is general. Example: PULP support
  lives in benchmarks/har_ultracompact, the library only offers the generic `-DSYMBEX_POPCOUNT_FORCE_NATIVE`.
- READMEs: clear, simple, positive tone; what was achieved first, open items as future work.

## Methodology (the rules every benchmark follows)

- **Same architecture** whenever SYMBEX is compared with INT8 (the official model, or the same network trained with the
  same recipe). A network resized for binary is compared with INT8 resized the same way.
- **The test set is used once.** Every choice (checkpoint, threshold, variant, shape) is made on a validation split
  carved out of the TRAINING data. The CDT's `load_data` returns (train, validation, test).
- **Same recipe for every method** in a comparison: same epochs, seeds (42, 43, 44), data split, FP32 starting model.
  Show both recipes when one favors a method (distillation vs labels only).
- **Exact, always**: the compiled C (PC and chip) must equal the integer simulation 100 %, image by image.
- **Weights in flash first** (how anyone deploys), then the RAM variant.
- Results are means ± standard deviation over 3 seeds; single-seed numbers are called rankings, not results.

## Layout

```
Symbex1_PIO/lib/lib_symbex/   the C engine: include/, src/ (layers/conv/symbex_bool*.c, layers/dense/...), hal/
                              (generic, esp32s3), tests/run_tests.py
Symbex1_PIO/src/main.cpp      minimal example: a demo network checked against its PC simulation on the board
tools/                        user tools: cdt.py, chip_test.py, configs/, chip_profiles/ (see tools/README.md)
tools/dev/                    developer tools: count_bench.py, fine_profile.py, train_bench.py
tools/compiler_core/          the CDT inside (see its README: current vs earlier-version code)
benchmarks/                   cnn_cifar_resnet8, fomo_voc_person, mlp_fashion, mlp_digits, har_ultracompact, experiments
```

## Environments and commands

- `benchmarks/env_bench` (Python 3.12): PyTorch with CUDA (cu130) + TensorFlow. Runs both sides of every benchmark.
- Library tests: `python Symbex1_PIO/lib/lib_symbex/tests/run_tests.py` (4 HAL configurations; `--stress` adds
  sanitizers and boundary sizes). Python side: `pytest -q` from the root. Run both before every push.
- Boards: ESP32 (classic, no SIMD) and ESP32-S3 (PIE SIMD) through PlatformIO; Raspberry Pi Pico 2 (RP2350: Hazard3
  RISC-V with Zbb popcount + Cortex-M33 with DSP) just arrived, not brought up yet.
- PULP / GVSoC (the Daghero comparison): Docker image `symbex-pulp` from benchmarks/har_ultracompact/pulp/Dockerfile;
  `docker run --rm -it -v ~/Proyectos:/work symbex-pulp`.

## Things that already caused errors

- CDT flags: yes/no options are written alone (`--data-on-gpu`, `--no-distill`), never `--x true`. Lists (`widths`)
  cannot go through the command line: the studies write a temporary copy of symbex.toml next to the original.
- `tools/chip_test.py` resolves the library relative to tools/: run it from tools/ (to fix: resolve from its own path).
- `fashion_int4.py` (Obed) still has Spanish options (`--epocas`, `--semillas`): to translate.
- The FP32 model of mlp_fashion is retrained per machine and is not bit-identical: the tables record its sha256.
- Old SYMBEX chip results in mlp_fashion/results (symbex_s*_chip_*.json) predate the current library and the
  validation fix: `fashion_tflite.py`'s table mixes them in.

## Where things stand (late October 2026)

- **ResNet-8 (CIFAR-10), fully binary:** 40.7 ms on the ESP32-S3 (1.28× faster than MLPerf INT8), 59.7 ms on the ESP32
  (12×), 24 KB of flash. Sized for binary (16-32-128 / 1-1-2): 73.1 % vs 71.3 % at the same speed.
- **FOMO person detection:** binary and INT8 detect alike (F1 31.2 vs 30.1 %, 3 seeds), binary 2× faster on the S3,
  ~6× less flash, and the only one that fits the classic ESP32.
- **Training speed:** data on the GPU + CUDA graphs (`data_on_gpu = true`, `compile = "reduce-overhead"` in
  [training]): ResNet-8 8.6 → 3.0 min, same accuracy and exactness.
- **mlp_fashion low-bit study:** INT2 QAT is more accurate per byte than SYMBEX; SYMBEX wins speed and the small end.
  Sizing the MLP for binary (narrow first layer, wide binary layers): 784-128-512-10 runs in 0.42 ms on the S3 with the
  weights in flash (INT8 TFLite: 15.9 ms) and 0.41 ms in RAM (INT8: 0.88 ms), 27 KB of flash, 4.6 KB of RAM.
- **1D convolutions:** lib_symbex has `symbex_bool1d` (any channel count, fused OR pool, two windows per pass where the
  compiler counts bits natively). Layer by layer on GVSoC (RI5CY, same p.cnt instruction), SYMBEX needs 1.08-1.97×
  fewer instructions than ultracompactBNN (Daghero et al. 2021) on all 6 shapes, both exact.

## Next, in order

1. **mlp_fashion validation contamination.** The FP32 model (now 30 epochs) is trained on all 60,000 training images,
   including the 6,000 the CDT keeps for validation, so validation is inflated (90-92 % vs 87.7-88.6 % on test) and the
   sizing ranking is biased. Fix: one FIXED validation split for every seed (the CDT's split seed today is the training
   seed), the FP32 trained without it (model.py), fashion_int4.py using the same split. Then rerun lowbit_study.py (both
   recipes, now with INT8 QAT: check `INT_BITS = [8, 4, 2]`) and fashion_sizing.py (both recipes). The chip latencies
   do not need re-measuring (they depend on the shape, not the weight values): copy the chip_*.json files back from
   ~/Proyectos/mlp_fashion_fp32_e5/runs/ for the predictor. Then rewrite mlp_fashion/README.md in full: the INT vs
   SYMBEX comparison, both recipes, the sizing theory with its tables, how to reproduce (option A: the published FP32
   models + sha256, .gitignore exception; option B: everything from scratch, within seed noise).
2. **1D convolutions to the end:** UniMiB-SHAR data, the architectures of Daghero's Table 1, the whole network in both
   libraries on GVSoC (instructions as the main measure: GVSoC's pulp-open memory is slower than their PULPissimo),
   TFLite INT8, then the boards. This closes v3.
3. **Pico 2:** bring-up on both cores, count_bench, a HAL per core; TFLite INT8 with CMSIS-NN on the M33.
4. **Energy** with the current meter Edén ordered: the metric no benchmark has yet.
5. Smaller: ternary kernel (one count per word, HAL row functions), `chip_test.py` library path, CDT labels
   ("Test" that is now validation), translate remaining Spanish modules, FOMO INT8 with the binary recipe.
6. After v3 (Obed's ideas, agreed): a live camera demo (FOMO on ESP32-CAM vs TFLite, FPS side by side), keyword
   spotting on a coin cell (needs 1D + energy), then the bolder ones (model in a QR code, logic gates on an FPGA, web
   playground).
