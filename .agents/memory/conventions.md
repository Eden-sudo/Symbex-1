# Project conventions

Collected from the merged pull requests and their reviews (November 2026). The measurement rules themselves are in
[docs/methodology.md](../../docs/methodology.md), "The standard"; this file does not replace it.

## Git and pull requests

- Branches: `<type>/<topic>` from `master`, e.g. `bench/ecg-mitbih` (#35), `perf/s3-stem-pie` (#29),
  `data/train-gap-results` (#24), `docs/updates` (#37), `fix/tools-review` (#38). Stacked PRs say which PR they sit on.
- Commits: `type(scope): subject`; types seen: `feat`, `fix`, `perf`, `bench`, `docs`, `test`, `refactor`, `chore`.
  The body carries the numbers a change produces.
- Code, comments, commits and PRs in English. User documentation in English and Spanish: `README.md` with
  `README.es.md`, `docs/<name>.md` with `docs/<name>.es.md`; the Spanish file says the same as the English one (#20).
- PR description: what was done, what is in progress, what is next; the test counts and the exactness measured.
- Every PR adds its section to [UPDATES.md](../../UPDATES.md): `## PR #N: title (Month YYYY)`, newest first.
- Large result data goes in its own data PR: a diff above ~4 MiB breaks the automatic review (#23 -> #24).
- Reviews are posted by MendCode; the author asks for a re-review after each push. A review request in a comment
  routes work, it does not authorize running code, flashing a board or pushing (PR #12).

## Measurement rules most often asked for in reviews

- The test set is used once, on the final models; every choice is made on validation (#1, #10, #11, #29, #35).
  Validation comes from the training data with one fixed split for every seed
  (`benchmarks/mlp/mlp_fashion/symbex.toml`: `split_seed = 42`).
- Seeds 42 / 43 / 44, mean +- standard deviation; one seed is a ranking, not a result.
- Every figure in a README has a committed JSON or log behind it (#8, #11, #15). Figures not measured are estimates
  and say where they come from (#2, #33).
- The board gives the same output as the integer simulation on 100 % of the inputs, for SYMBEX and the integer
  baselines; host tests replace the S3 PIE assembly with C, so only the board checks it (#18).
- The documented command must reproduce the measurement, not silently retrain (#31) or reuse a stale cache (#35).

## What is not published

`.gitignore`: `archive/`, `convul_symbex_v3/`, `symbex_x86/`, datasets (`data/`, `*.npy`, `*.npz`), trained models
and binaries (`*.pth`, `*.pt`, `*.onnx`, `*.tflite`, `*.u8`, `**/models/`) and `benchmarks/**/runs/`. Only scripts,
configs, READMEs and `results/*.json` are published, plus the reference models that a benchmark measures as they are
(`benchmarks/**/reference/*.pth`, `*.tflite`, added on purpose despite the ignore rules) and a few `mlp/*/models/*.pth`.

## Environment

Benchmarks run in `benchmarks/env_bench` (Python 3.11 or 3.12, PyTorch and TensorFlow; `benchmarks/README.md`). The
library tests need gcc and numpy; `pytest` reads `pytest.ini` (`testpaths = tests`, `pythonpath = tools`). Board
names: `esp32`, `s3`, `pico2_arm`, `pico2_riscv`.
