# Integrated LLM work (host only)

This index covers the ten branches merged into `lane/integration`, in the requested
order: LittleBit, GQA/KV, runtime, sensor-text, import-models, quantizer tests,
integer tests, preparation tests, Hadamard tests, and energy-per-token tooling.
All ten merges were conflict-free. No tests or lane features were discarded.
Every incoming test file passed immediately after its merge; no repair commits
were needed. Nothing was pushed and no board or firmware build was run.

Paths below are relative to this directory; tests are relative to repository root.
Each linked test was run on the host. These are synthetic checks, not evidence
of trained model quality, firmware compatibility, hardware speed or energy use.

## Integrated acceptance paths

- [Checkpoint to C](../../../tests/test_llm_host_pipeline.py): synthetic
  llama2.c checkpoint, binary import saved/reloaded, `w1` and `w2` factored in
  each block, serialized blobs read back, clang C matvec exactly equals Python
  integer output. Single and residual pairs, zero/extreme/random int8 inputs.
  This is projection-level execution, **not a complete factored transformer C ABI**.
- [Chat](../../../tests/test_llm_chat_pipeline.py): stub tokenizer chat framing,
  tiny random binary GQA model (two query heads, one KV head), six autoregressive
  selections with independent histories; Python/C token IDs, persistent RNG states
  and streamed bytes agree for greedy and stochastic nucleus sampling.
  **Both histories use Python GQA forward**; C implements sampling and decoding,
  not a complete GQA model forward. This test uses a fixed generation budget,
  not the runtime's EOS stopping policy (covered separately).
- [Sensor smoke](../../../tests/test_llm_sensor_pipeline.py): 50 records
  (41 train, 9 validation), temporary lossless tokenizer and uint16 streams,
  five CPU training steps, finite train/validation loss, saved-model reload,
  numeric-slot generation and sentence evaluation. Prose is explicitly templated;
  the model scores constrained number tokens. No free-form quality claim.
- Shared implementations: `memory_budget.py` calls `attention_memory.kv_bytes`
  and adds the legacy 16-byte guard; `sample_topk` calls `sample` with full mass.
  Their previous public inputs remain valid. C samplers already share
  `tinylm_rng.h`; the header-only top-k API remains usable without linking chat.

## Core and attention

- `train.py` quantizers — `tests/test_tiny_llm_quant.py`: quantization and gradients.
- `train.py` GQA blocks — `tests/test_tiny_llm_gqa.py`: grouped heads and validation.
- `train.py`/`intsim.py` Hadamard helpers — `tests/test_tiny_llm_hadamard.py`:
  tiny transforms and rotation behavior, not a training campaign.
- `intsim.py` — `tests/test_tiny_llm_intsim.py`: integer helpers and packing.
- `prepare.py` — `tests/test_tiny_llm_prepare.py`: split/token preparation fixtures.
- `gqa_intsim.py` — `tests/test_tiny_llm_gqa_intsim.py`: bounded integer attention,
  causality, float tolerance and exact MHA special case; 64-wide heads required.
- `c/gqa_attention.c`, `c/gqa_attention.h` — `tests/test_tiny_llm_gqa_c.py`:
  C attention equals Python integers on bounded host cases.
- `gqa_rolling.py` — `tests/test_tiny_llm_gqa_rolling.py`: rolling integer cache.
- `c/gqa_rolling.c`, `c/gqa_rolling.h` — `tests/test_tiny_llm_gqa_rolling_c.py`:
  cache wrapping and absolute-position host parity.
- `attention_memory.py` — `tests/test_tiny_llm_attention_memory.py`: per-head
  payload/scales and padding. Sub-8-bit KV sizes are hypothetical, not codecs.
- `memory_budget.py` — `tests/test_tiny_llm_memory_budget.py`: legacy export byte
  sizes, allocation arithmetic and shared KV delegation, not measured free RAM.
- `attention_memory.md` — `tests/test_tiny_llm_attention_doc.py`: documented contract.

## LittleBit

- `littlebit.py` — `tests/test_littlebit_factor.py`, `tests/test_littlebit_residual.py`:
  trainable sign factors, gradients and optional residual pair.
- `littlebit_itq.py` — `tests/test_littlebit_itq.py`: seeded CPU Joint-ITQ rotation.
- `littlebit_model.py` — `tests/test_littlebit_model.py`: opt-in projection replacement.
- `littlebit_int.py` — `tests/test_littlebit_intsim.py`: tri-scale integer reference.
- `littlebit_blob.py` — `tests/test_littlebit_blob.py`: strict versioned serialization.
- `c/tinylm_factor.h` — `tests/test_littlebit_factor_c.py`: integer factor kernel parity.
- `c/littlebit_blob.c`, `c/littlebit_blob.h` — `tests/test_littlebit_blob_c.py`:
  blob-consuming C matvec parity and rejection before output writes.
- `littlebit_budget.py` — `tests/test_littlebit_budget.py`: packed factor accounting.
- `littlebit_analysis.py`, `littlebit.md` — `tests/test_littlebit_analysis.py`:
  storage/operation estimates, not measured throughput or perplexity.

## Runtime

- `sample_int.py`, `c/tinylm_sample.h` — `tests/test_tiny_llm_sampler.py`:
  integer top-k sampling, known RNG vectors, rejection and independent streams.
- `c/tinylm_rng.h` — `tests/test_tiny_runtime_rng.py`: shared xorshift32 contract.
- `c/chat_sample.c`, `c/chat_sample.h` — `tests/test_tiny_runtime_sample.py`:
  temperature/top-p and Python/C parity.
- `chat_format.py`, `c/chat_format.c`, `c/chat_format.h` —
  `tests/test_tiny_runtime_format.py`: existing-vocabulary framing.
- `chat_memory.py`, `c/chat_memory.c`, `c/chat_memory.h` —
  `tests/test_tiny_runtime_memory.py`: retain whole turns, preserve system span.
- `chat_penalty.py`, `c/chat_penalty.c`, `c/chat_penalty.h` —
  `tests/test_tiny_runtime_penalty.py`: saturating presence penalty.
- `chat_stop.py`, `c/chat_stop.c`, `c/chat_stop.h` —
  `tests/test_tiny_runtime_stop.py`: EOS, byte stops and token limit.
- `chat_pieces.py`, `c/chat_pieces.c`, `c/chat_pieces.h` —
  `tests/test_tiny_runtime_pieces.py`: exported piece bytes and streaming decoding.
- `c/chat_loop.c`, `c/chat_loop.h`, `c/chat_host.c` —
  `tests/test_tiny_runtime_loop.py`: bounded callback loop and host driver checks.
- `chat_runtime.md` — `tests/test_tiny_runtime_doc.py`: documented host limitations.

## Importers

- `importers/llama2c.py` — `tests/test_import_llama2c.py`: synthetic v0 layouts.
- `importers/mapping.py` — `tests/test_import_mapping.py`: tensor mapping,
  expanded GQA and classifier semantics.
- `importers/binarize.py` — `tests/test_import_binarize.py`: binary initialization.
- `importers/gguf.py` — `tests/test_import_gguf.py`: bounded GGUF tensor reader;
  not general GGUF architecture import.
- `importers/tokenizer.py` — `tests/test_import_tokenizer.py`: original token bytes/IDs.
- `importers/import_model.py` — `tests/test_import_cli.py`: local import, manifest,
  CPU reload and rejecting unsupported inputs without output overwrite.
- `importers/importers.md`: supported-format and synthetic-only verification limits.
  Real pretrained checkpoints/tokenizer compatibility have not been validated here.

## Sensor-to-text (`../sensor_text`)

- `../sensor_text/make_dataset.py` — `tests/test_sensor_text_dataset.py`: seeded
  synthetic records and template-separated splits using repository class labels.
- `../sensor_text/prompt.py` — `tests/test_sensor_text_prompt.py`: canonical records.
- `../sensor_text/prompt.c`, `../sensor_text/prompt.h` —
  `tests/test_sensor_text_c_prompt.py`: C/Python prompt parity.
- `../sensor_text/tokenizer.py` — `tests/test_sensor_text_tokenizer.py`: lossless
  numeric and prompt round trips, trained on synthetic train examples only.
- `../sensor_text/pretokenize.py` — `tests/test_sensor_text_pretokenize.py`: uint16 layout.
- `../sensor_text/train.py` — `tests/test_sensor_text_train.py`: toy CPU optimization.
- `../sensor_text/decoding.py` — `tests/test_sensor_text_decoding.py`: explicit numeric
  copy slots; this does not automatically detect numbers in arbitrary prose.
- `../sensor_text/evaluate.py` — `tests/test_sensor_text_evaluate.py`: field-aware
  template grammar and hallucination counting, not a general semantic evaluator.
- `../sensor_text/_modules.py`: file-path loader exercised by the sensor tests above.
- `../sensor_text/README.md`, `../sensor_text/sensor_text.md`,
  `../sensor_text/prompt_format.md`: domain contracts and readiness boundaries.

## Energy and existing research

- `tools/energy_per_token.py` (repository root) — `tests/test_energy_per_token.py`:
  synthetic meter-trace analysis. **No real power trace was measured here.**
- Existing `c/tinylm.c`, `c/tinylm.h`, `c/tinylm_symbex.h`, `c/check_pc.c`,
  `chip_run.py`, and `chip/`: legacy model/firmware, not upgraded into a combined
  LittleBit/GQA board runtime by this integration.
- Existing `layer_sensitivity.py`, `damage_map.py`, `w1a1_diag.py` and full
  `hadamard_probe.py` experiments: not rerun; historical results in README are
  distinct from this branch's synthetic host evidence.

## Reproduce only the needed check

From this worktree, run one named file at a time (no broad suite):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /Users/obed/repos/Symbex-1/.venv/bin/python -m pytest -q tests/test_llm_host_pipeline.py
```

The three integration files above need only CPU, temporary synthetic files,
installed torch/numpy/SentencePiece and `/usr/bin/clang`; no downloads. Each was
under one second locally. The sampler regression file was under six seconds.
`tests/test_llm_work_index.py` checks this index's test references and lane coverage.
Board execution, combined firmware/export ABI, real training, model quality,
hardware latency, physical energy, and real-model downloads remain unverified.
