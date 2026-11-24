# Sensor-to-text lane

Goal: classifier/sensor readings become an English or Spanish report locally on the
ESP32-S3. This is not open dialogue, recognition, medical diagnosis, or a trained demo.

## Round 1 infrastructure (preserved)

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

## Round 2 dataset card

**Identity:** `sensor_v2`, entirely synthetic environmental sensor-to-text data.
`environment_dataset.py` generates it; the original `make_dataset.py` still generates
the round-1 classifier data. No real device readings, people, medical observations,
downloaded corpora, or measured environmental distributions are included.

**Sizes:** default recipe is **256 train / 64 val / 64 test rows**, configurable with
`--n TRAIN VAL TEST`. These are generator settings, not a committed dataset or model
accuracy measurement. A row contains `event` and `text`; output is UTF-8 JSONL plus
`meta.json` recording sizes, master seed, derived split seeds, mode, languages,
allowed phrasing IDs, and the leakage audit. No generated data/models are committed.

| Field | Integer unit | Inclusive range | Meaning |
|---|---|---|---|
| temperature | C | -40..85 | Temperature |
| humidity | pct | 0..100 | Relative humidity |
| co2 | ppm | 250..5000 | Carbon dioxide concentration |
| battery | pct | 0..100 | Remaining battery charge |
| motion | bool | 0..1 | Absent / detected |
| light | lux | 0..100000 | Illuminance |
| sound | dBA | 0..140 | Sound level |
| door | bool | 0..1 | Closed / open |

These are physically plausible operating envelopes, not calibrated sensor limits.
All six analog fields have `history` arrays of `window` readings (2..16) and a
`trends` entry: `rising`, `falling`, or `steady`, derived by comparing the last
reading with the first. The current value equals the last reading. This is endpoint
change, **not monotonicity**. Histories are independent uniform integer draws:
their rates of change, correlations, and timing are not realistic telemetry.
Motion and door are current states only. There are no timestamps or sampling-rate claims.

**Language and targets:** `language=en|es` is explicit in the prompt. Each split
alternates languages, starting with English (an odd split has one extra English row).
Each language has three phrasings per fact; a row uses one phrasing index for all
facts. Every target mentions all eight current values/states, all six trends and
their window lengths, plus an alert clause. Units remain canonical wire units in
both languages. Targets are controlled multi-clause reports, not necessarily short
enough for the round-1 128-token training context. Size the context for the actual
tokenized records before any meaningful training; arbitrary stream windows can cut
examples. No training recipe or generation quality was established in round 2.

**Splits and leakage:** `--mode seed` uses phrasings 0/1/2 in all splits;
`--mode template_holdout` (default) uses 0/1 in train and val, and 2 only in test.
The latter measures unseen-wording generalization, not an independent sensor domain.
SHA-256 derives separate RNG seeds from the master seed and split name. Generation
rejects duplicate `(canonical record, sentence)` pairs and runs an explicit cross-split
audit. `template_id` is excluded from record identity and never serialized into the
prompt. Histories and language are input facts and remain in identity. This guarantee
does not prohibit shared scalar values or logically equivalent facts across splits;
it is not a claim of semantic deduplication. Bounded retries fail rather than leak.
Tokenizer training uses only generated train rows with phrasings 0/1; held-out text
must not be used for tokenizer fitting or checkpoint selection.

**Alerts:** illustrative, inclusive rules: `co2_high` at co2 >=1000 ppm,
`battery_low` at battery <=20 pct, and `sound_high` at sound >=85 dBA. Labels are
derived from records, never independently sampled. Targets name each active alert
in the selected language, or explicitly say none. These are not certified safety
thresholds. `evaluate.py` aggregates `missed_alerts`, `false_alerts`, and
`unsupported_alert_count`; only the declared terminal alert grammar is recognized.
Unknown prose cannot establish alert correctness and is flagged unsupported.
Environmental fact credit is conservative, clause-based: a complete matching sensor
clause earns its value/unit/trend/window facts together; a changed clause earns none.
This is not a general semantic evaluator. Round-1 class/numeric evaluation is preserved.

**Serialization and C encoding:** `prompt.py` dispatches `<sensor_v1>` and
`<sensor_v2>`; v2 orders source, language, window, then each field in the table above,
followed for analog fields by comma-separated history and trend. Integers are decimal,
units follow one space, fields use `|`, and `<text>` ends the prompt. Parsing requires
exact canonical reserialization. `pretokenize.py` accepts both formats unchanged.
`NumberCopy` supports v2 current numeric fields and window, including negative
temperatures; history-array slots are not automatically constrained.

`export_encoder.py` exports identity-normalized SentencePiece BPE models to a C table
of printable-ASCII pieces, IDs, scores, and kinds. `encoder.c` performs greedy
highest-score adjacent merges with leftmost tie-breaking, preserving atomic framing
symbols and byte fallback. It is **not longest-prefix token matching**. It emits
neither BOS nor EOS, accepts at most 4096 printable ASCII bytes, and rejects insufficient
output capacity without writing IDs. The table must have fewer than 2048 entries.
Non-ASCII target text is deliberately outside this prompt-only encoder's domain.
Workspace is fixed (4096 symbol entries plus a 4096-int lookup table); this has not
been sized or integrated for an ESP32 task stack. The original `prompt.c` builder
still builds **v1 only**; v2 serialization is Python-side in this round.

Host tests compile with `/usr/bin/clang` and compare 352 generated v1/v2 prompts
plus four edge strings against SentencePiece for each of two independently fitted
domain tokenizers (712 ID-sequence comparisons), with capacity and invalid-input
checks. This is host evidence, not a board run or a proof over every possible model.
No protobuf dependency is required by the exporter; it reads the required wire fields.

### Tiny round-2 workflow

Run from the repository root, choosing fresh output directories:

```
python benchmarks/experiments/sensor_text/environment_dataset.py --out /tmp/env-data --n 256 64 64 --seed 0 --mode template_holdout
python benchmarks/experiments/sensor_text/tokenizer.py --out /tmp/env-tok --rows 256 --seed 0 --environment
python benchmarks/experiments/sensor_text/pretokenize.py --dataset /tmp/env-data --tokenizer /tmp/env-tok/tokenizer.model --out /tmp/env-bin
python benchmarks/experiments/sensor_text/export_encoder.py /tmp/env-tok/tokenizer.model /tmp/env-vocab.c
```

The default-size workflow is a reproducible recipe, not a reported training run.
Tiny tests cover ranges, deterministic histories, all bilingual phrasing facts,
split policy and injected leakage, alert boundaries, C IDs, and document defaults.

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
