---
name: symbex-benchmark-run
description: Adds, repeats or re-measures a SYMBEX-1 benchmark against TFLite Micro INT8 (and INT4 / INT2) on the boards, following docs/methodology.md, and records the results. Use when a benchmark is created or re-measured, or when an engine change needs new figures.
---

# SYMBEX-1 benchmark run

The rules are in [docs/methodology.md](../../../docs/methodology.md) ("The standard" and "Repeat a benchmark"); this
skill is the order of work. Each benchmark folder has `<bench>_symbex.py`, `<bench>_tflite.py` and, where it exists,
`<bench>_lowbit.py`; read its README and `--help` first, the flags differ per benchmark.

## 1. Before measuring

- Environment: `benchmarks/env_bench` (Python 3.11 or 3.12, PyTorch and TensorFlow; `benchmarks/README.md`).
- Same architecture, recipe, seeds (42 / 43 / 44) and data split for every method; validation from the training data.
  Pass the architecture explicitly to every script: their defaults can differ. Where a benchmark's README documents a
  different recipe per method (FOMO), keep it and say so; `docs/methodology.md` asks to show both recipes then.
- Board, owner and commit noted; the shared S3 needs a board request ([symbex-sync](../symbex-sync/SKILL.md)).

## 2. Measure

From the benchmark folder:

1. SYMBEX: train, export and simulate, then on the board with `--board <board> --port <port>`, weights in flash
   first, then `--weights ram`.
2. INT8: `<bench>_tflite.py --board <board> --port <port>`. To measure an existing `.tflite` without retraining, use
   `--model <file>` where the script has it (#31). Watch for cached models that ignore new epochs or learning rate (#35).
3. INT4 / INT2: `<bench>_lowbit.py` (`tools/lowbit`).
4. Every run must be exact on the board ([symbex-exactness-check](../symbex-exactness-check/SKILL.md)); a run that lost
   serial lines is not saved as a result (#23).

## 3. Record

- Results in the benchmark's `results/` as JSON; superseded results move to `results/earlier/`, nothing measured is
  deleted. `runs/` is never published.
- Accuracy on validation, the test set once on the final models; mean +- standard deviation over 3 seeds; differences
  inside the deviations are a tie.
- Update the benchmark README, `benchmarks/README.md`, `README.md` and `README.es.md` with the same figures, each
  pointing to its JSON; then `UPDATES.md` and `docs/roadmap.md`.
- Large result diffs go in a separate data PR (#23 -> #24).

Finish with [symbex-pre-pr-check](../symbex-pre-pr-check/SKILL.md).
