# Bounded host evaluation

These tools evaluate local TinyLM runs without training, downloads, CUDA or a
board. Run the examples from `benchmarks/experiments/tiny_llm`, using a Python
environment with torch and numpy. The determinism harness also requires
`/usr/bin/clang`. All model files must be trusted local checkpoints.

## Loss and perplexity

```sh
python eval_loss.py runs/example validation.bin --max-tokens 128 --backend float
python eval_loss.py runs/example validation.bin --max-tokens 128 --backend intsim
```

`model.pt` must contain the stock `train.py` `args` and `model` state dictionary.
Vocabulary size comes from the embedding, not the token file. Float evaluation
reconstructs stock training options, including GQA; unsupported or factored state
dictionaries fail strict loading. This is not a universal checkpoint importer.
Integer evaluation requires binary or ternary weights, 64-wide heads, and the
configurations accepted by the existing `intsim.load_model` guards. FP32 is not
silently binarized. Ternary simulation does not imply a ternary C export exists.

The input is a raw little-endian uint16 stream using the model's vocabulary.
`--offset` is a token offset (default zero). At most `--max-tokens` next-token
targets plus one initial token are read. Context resets at each checkpoint-sized
window; adjacent windows share their boundary token. A final partial window is
included. This is teacher forcing, not a rolling-cache perplexity measurement.
Loss is summed cross-entropy in nats divided by the actual target count;
perplexity is `exp(loss)`. Integer logits use `intsim.float_logits` scaling.
Odd byte counts, empty slices, invalid caps and out-of-vocabulary IDs fail.

CLI execution uses one torch CPU thread. Imported API callers choose their own
thread limits. The cap bounds token data, not checkpoint size or model memory.
Choose checkpoints and variants on validation; reserve test for final reporting.

## Float/integer agreement

```sh
python eval_agreement.py runs/example validation.bin --windows 2 --max-tokens 128
```

The report includes top-1 match count, agreement fraction, and maximum absolute
logit difference across positions and vocabulary. Both paths receive identical
teacher-forced inputs. Integer scores are dequantized before logit subtraction.
Ties follow torch argmax (lowest ID). The smaller of the requested window/token
budgets and available data wins; actual counts are recorded. Like loss evaluation,
the reader reserves a following target token even though agreement does not use it.

## Generation determinism

```sh
python eval_determinism.py runs/example determinism.json --prompt 1 2 --steps 8 --seed 42 --temperature 16
```

Two Python sampler runs and one C sampler run start with independent histories
and the same nonzero uint32 seed. Each gets intsim scores for its own evolving
history, cropped to the checkpoint context. JSON records generated IDs, every RNG
state, final state, options and `equal`; a mismatch writes evidence and exits 1.
The output path is overwritten, so choose a new path to retain earlier evidence.
The temporary compiled sampler library is removed after the check.

Temperature is integer score units per log2 weight halving, **not** a conventional
floating softmax temperature; zero is greedy. `--k 0` keeps all tokens and
`--top-p` uses Q16 mass in 1..65536. This is a fixed token budget, without EOS or
byte-stop policies. Both backends use **Python model forward**: parity does not
prove a C transformer, tokenizer, board or firmware is correct. The `check` API
also accepts a pure `next_scores(history)` callback returning signed int64 scores.

## Conversation format and scoring

`eval_conversations.json` contains Spanish and English multi-turn fixtures.
Each case has a unique `id`, `language` (`es` or `en`), and ordered `turns`.
A turn has a user `prompt`, `keywords`, `facts`, `min_words` and `max_words`.
Facts are lists of alternative phrases: `[ ["azul", "blue"], ["dos gatos"] ]`
requires one phrase from each inner list. Prior prompts and generated answers
form the conversation history; the scorer does not generate responses itself.

An answers file maps each case ID to one string per turn, with no extra IDs:

```json
{"memory-es": ["Ana", "Lima"], "memory-en": ["Sam", "blue"]}
```

```sh
python eval_conversation.py eval_conversations.json answers.json
```

Matching uses NFC-normalized, casefolded Unicode words and contiguous phrases;
accents remain meaningful, punctuation is ignored, and `cat` does not match
`concatenate`. Empty requirement lists have recall 1. Length bounds are inclusive.
Repetition is `(word count - distinct word count) / word count`, or zero for an
empty answer. The report contains each turn and unweighted macro means.
**Lexical recall is not semantic truth**: negation and incorrect relations are
not understood, and repetition includes ordinary repeated function words.

## Result JSON and methodology

The canonical methodology on this branch is
[`docs/en/methodology.md`](../../../docs/en/methodology.md).
`eval_results.py` accepts an input JSON with `seeds`, `metrics`, `evidence`,
`source`, `environment`, `split`, `commit` and `engine`:

```json
{"seeds": [42, 43, 44], "metrics": {"loss": {"unit": "nats/token", "values": [1.0, 2.0, 3.0]}}, "evidence": "measured", "source": "synthetic example, not model quality", "environment": "host CPU", "split": "synthetic", "commit": "replace-with-evaluated-commit", "engine": "float"}
```

```sh
python eval_results.py summary_input.json summary.json
```

Raw values align with unique seeds; output adds mean and population standard
deviation (`std_ddof: 0`). One seed is labelled `ranking`, two `preliminary`, and
three or more `multi_seed`. Use the methodology's 42/43/44 for comparable runs.
`evidence` is explicitly `measured` or `estimate`; both require a source and an
environment. Host measurements are not chip measurements. Nonfinite values and
missing provenance fail before writing; existing output files are never replaced.
This host schema does not invent board latency, RAM, energy, or hardware parity.

## Tiny checks and limits

Run only the needed test file from the repository root with
`python -m pytest -q tests/<name>.py`:

- `test_llm_eval_loss.py`: random model and manual cross-entropy, both backends.
- `test_llm_eval_agreement.py`: direct toy-model comparison and bounded counts.
- `test_llm_eval_determinism.py`: compiled C sampler, RNG history and mismatches.
- `test_llm_eval_conversation.py`: hand-written bilingual and rejecting cases.
- `test_llm_eval_results.py`: schema, statistics, provenance and no-overwrite.
- `test_llm_eval_doc.py`: executable documentation examples on synthetic inputs.

These checks establish host tooling behavior only. No trained-model quality,
real-tokenizer compatibility, board execution, latency or energy is measured.
