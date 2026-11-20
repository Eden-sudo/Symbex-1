# Sensor-to-text lane

Goal: classifier/sensor readings become one short English sentence locally on the
ESP32-S3. This is not open dialogue, recognition, medical diagnosis, or a trained demo.

## What exists

- `make_dataset.py`: seeded synthetic HAR, ECG and FOMO events; templates a/b for
  train, c for validation, d for test. All repository VOC labels are included,
  not just person. Numeric event overlap across splits is possible; this is a
  template holdout, not an independent real-world sensor benchmark.
- `prompt.py` and `prompt_format.md`: strict versioned integer serialization;
  `template_id` never enters the prompt. No silent rounding.
- `tokenizer.py`: 512-token default BPE, identity normalization, byte fallback,
  atomic framing tokens; trains only on a small generated training corpus.
  Actual vocabulary may be smaller than requested. Use its actual vocabulary size.
- `pretokenize.py`: BOS + prompt + sentence + `</text>` + EOS, concatenated without
  headers as native-endian uint16. Compatible with `tiny_llm/train.py:load_split`;
  IDs must be <=32767 because that loader casts to int16. On this laptop the files
  are little-endian. Split boundaries are preserved; no shuffling across splits.
- `train.py`: imports the existing TinyLM classes; configurable width, layers,
  heads, hidden size, context and weight format. Defaults: width 64, 2 layers,
  4 heads, hidden 128, context 128, binary block weights, fp32 tied output head,
  CPU, five steps. Full causal loss includes prompt tokens (not answer-only SFT).
  Saves the best checkpoint including step zero, chosen on at most four fixed
  validation windows. Never reads test.bin. This is smoke infrastructure, not a
  validated training recipe or full-validation measurement.
- `evaluate.py`: JSONL `event`, `text` (reference), `generated` -> JSON metrics.
  Exact match is byte-sensitive. Field accuracy is micro-averaged over class and
  numeric fields, preserving ECG before/after identity and required units.
  The four declared sentence forms per source are recognized; unsupported prose
  conservatively gets zero field credit and an explicit unsupported flag.
  Hallucinations count numeric occurrences whose values are absent from the record;
  swapping two allowed values is a field error, not a numeric hallucination.
  This grammar-based evaluator is not a general semantic or clinical judge.
- `prompt.c/.h`: allocation-free host-tested C99 serialization; required capacity
  includes NUL; -1 on invalid input or truncation. C uses only the selected source's
  fields. Python rejects unrelated dictionary fields. Class membership is supplied
  by the classifier; both serializers enforce only ASCII identifier syntax.
- `decoding.py`: optional NumberCopy trie over tokenized record numbers with a
  trailing space. Caller explicitly enters/exits a numeric slot; it is not a
  whole-sentence factuality guarantee. Optional field selection prevents swapping
  fields. Signed integer logits keep their dtype. Use the returned legal bitmap
  or `choose()`, not plain argmax over masked scores (minimum-integer ties).

## Small local workflow (not run as a real training here)

Run scripts from the repository root; every destination must be new:

```
python benchmarks/experiments/sensor_text/make_dataset.py --out /tmp/sensor-data --n 82 41 41 --seed 0
python benchmarks/experiments/sensor_text/tokenizer.py --out /tmp/sensor-tok --rows 82
python benchmarks/experiments/sensor_text/pretokenize.py --dataset /tmp/sensor-data --tokenizer /tmp/sensor-tok/tokenizer.model --out /tmp/sensor-bin
```

The training CLI requires `--data`, `--out` and `--vocab` (the tokenizer's actual
size), and supports the model dimensions and `--steps`. Outputs are local experiment
artifacts, not committed models. Failed preparation can leave a partial destination;
retry with a new path. No downloads or implicit GPU use.

## Evidence and remaining work

Each addition has its own `tests/test_sensor_text_<part>.py`, imported by file path.
Tests use temporary directories: exhaustive allowed-number tokenizer round-trips,
actual upstream binary-loader compatibility, exactly five optimizer steps on a
repeated sensor toy example, hand-written evaluation failures, 606 C/Python prompt
comparisons plus invalid/buffer cases, and exhaustive numeric-copy trie traversal.
The toy check establishes finite and decreasing loss, not language quality.

No real model has been trained, no held-out model accuracy has been measured, no
sensor model exported to integer inference, no tokenizer integrated into firmware,
and no ESP32-S3 latency, memory, power or end-to-end generation measured. Existing
TinyStories board numbers do not establish sensor-text results. Future work needs
realistic data, validation-based recipe selection, one final test evaluation, export
and host integer parity, then an explicitly authorized S3 board run.
