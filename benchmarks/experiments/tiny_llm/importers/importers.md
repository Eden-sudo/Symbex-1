# Local model importers (synthetic verification only)

These tools do not fetch models. All tests create tiny files in temporary
directories; **no real 260K/15M llama2.c or ThoxMicro checkpoint was loaded**.
No training quality, real tokenizer equivalence, C export, board correctness,
memory capacity or performance is established by these tests.

## Import a local llama2.c checkpoint

From the repository root, with NumPy and PyTorch installed:

```sh
python benchmarks/experiments/tiny_llm/importers/import_model.py LOCAL_CHECKPOINT NEW_RUN_DIR
# Optional lossless float initialization instead of the binary default:
python benchmarks/experiments/tiny_llm/importers/import_model.py LOCAL_CHECKPOINT NEW_RUN_DIR --weights fp32
```

The output directory must not exist. Outputs are `model.pt` (a plain state dict,
`args` and `step: 0`) and `import.json` (source dimensions, vocabulary size,
conversion choices and limitations). No `train.json` or measured losses are
fabricated. Binary output is **an initialization for fine-tuning**, not a trained
binary replacement with preserved logits. Float output preserves the supported
source model's forward computation within floating-point tolerance.

### Loading is not the same as running the stock scripts

The checkpoint loads through `train.TinyLM(...).load_state_dict(ck['model'])`
and `intsim.load_model(run, source_vocab_size, 'cpu')`. Training's `--init-from`
uses this same state dictionary; supply matching architecture options and source
token IDs. GQA expands whole K/V heads into ordinary multi-head matrices, with
the original query-head count. This preserves float inference but increases
storage; replicated KV parameters can diverge during fine-tuning.

Stock `train.py` and `intsim.py` command-line entry points use CUDA and the
project's prepared `tok4096.model`/token splits. **Do not use those assets with
an imported vocabulary merely because its size matches.** This importer does
not change those entry points, supply a tokenizer, or prepare data. Integer
attention also assumes **64-wide heads**; `import.json` records whether that
dimension matches. Other widths can load as float models but must not be used
for integer inference. Integer export requires binary weights; loading an fp32
checkpoint does not make `integerize(..., 'binary')` a lossless conversion.

Untied classifiers are supported by `mapping.to_model`'s float adapter, but
**rejected by the run-directory CLI**: stock TinyLM and intsim tie the output to
the embedding. Silently discarding an untied head would change the model.

## Supported file formats

### llama2.c run.c v0 (`llama2c.read_checkpoint`)

Little-endian seven-int32 header: `dim, hidden_dim, n_layers, n_heads,
n_kv_heads, vocab_size, seq_len`. Negative vocabulary means an untied
classifier. Contiguous float32 tensor order:

1. token embeddings, attention RMS weights;
2. all-layer Q, K, V, O matrices;
3. FFN RMS weights, all-layer W1, W2, W3;
4. final RMS weights, legacy RoPE real and imaginary tables;
5. classifier weights only for negative vocabulary.

Matrices are output-row/input-column, layer-major. K/V rows are
`dim / n_heads * n_kv_heads`. Exact file size and attention dimensions are
validated before read-only mmap views are created. Mapping copies weights into
owned CPU parameters and rejects non-finite weights. Like run.c, it computes
adjacent-pair RoPE at base 10000 and ignores the legacy tables. This is not a
reader for versioned/newer llama2.c export formats or arbitrary endian variants.

### llama2.c tokenizer.bin (`tokenizer.read_tokenizer`)

An int32 maximum token byte length, then one `(float32 score, int32 byte length,
raw bytes)` record per vocabulary entry. The caller supplies the vocabulary
count from the checkpoint; the file does not contain it. Index is token ID.
Bytes (including non-UTF8 and NUL) and scores are preserved, not normalized.

This reader is **not** a text encoder or a SentencePiece `.model` converter.
run.c BPE scores, BOS/EOS handling, whitespace and byte fallback must accompany
any future tokenizer implementation. IDs cannot be remapped independently of
embedding/classifier rows and prepared training data.

### GGUF v3 (`gguf.read_gguf`)

Little-endian header, all standard scalar metadata types, strings, and arrays
of scalars/strings; tensor infos and default 32-byte or explicit power-of-two
alignment. GGUF dimensions are fastest-axis-first; NumPy shapes reverse them.
Supports F32, F16, Q8_0 (32 int8 weights per fp16 scale), TQ1_0 and TQ2_0
(256 weights per fp16 scale, 54 and 66 bytes per block). Returned tensors
are float32. Invalid lengths, alignment, duplicate names, overlaps, unsupported
versions and types are rejected.

TQ layouts are GGML's plane-interleaved blocks, not `ternary_pack.py` payloads.
Synthetic tests cover all TQ1 five-trit patterns and multi-block row ordering.
Known unsupported types are reported by name, including IQ1_S, IQ1_M and
Q1_0; unknown/vendor types retain their numeric ID. **ThoxMicro's
actual file encoding has not been inspected.** Its advertised 1-bit/ternary
description does not identify a GGML tensor type. GGUF tensor decoding is not
architecture mapping: the CLI deliberately rejects GGUF rather than assume
that metadata, norms, activations, tokenizer or tensor names match TinyLM.

### Binary initialization (`binarize.binarize_model`)

Copies the float model, replacing the seven block linear matrices with
`where(w >= 0, +1, -1) * max(mean(abs(w), row), 1e-5)`. Enables binary QLinear
weights and int8 inputs; norms, embedding and classifier remain float. The
stored parameters stay trainable float tensors for the straight-through
estimator. `intsim.integerize` derives signs and integer row multipliers;
the importer does not claim a packed 1-bit `model.pt`.

## Verification performed

Run only the individual test needed, with the repository environment:

```sh
/Users/obed/repos/Symbex-1/.venv/bin/python -m pytest -q tests/test_import_cli.py
```

- `test_import_llama2c.py`: all tensors, tied/untied classifiers, GQA/MHA,
  malformed dimensions and truncated/extra bytes (10 tests).
- `test_import_mapping.py`: independent NumPy causal forward, two layers,
  GQA/MHA and tied/untied heads; non-finite rejection (5 tests).
- `test_import_tokenizer.py`: IDs, raw bytes, scores and malformed input (8 tests).
- `test_import_gguf.py`: synthetic F32/F16/Q8_0, metadata types, alignment,
  named unsupported types and malformed files (16 tests).
- `test_import_binarize.py`: exact shapes/signs/scales, integer multipliers,
  unchanged non-linear parameters/source model and finite gradients (2 tests).
- `test_import_cli.py`: subprocess CLI, train/intsim CPU reload, tiny binary
  integer forward with 64-wide heads, GQA expansion and safe rejection (7 tests).

All 48 tests passed individually on CPU, with each test file below 30 seconds.
No model downloads, whole-suite runs, firmware builds or training runs occurred.
The integer smoke test checks execution and finite outputs, not equality between
the quantized and original float model or board bit-exactness.
