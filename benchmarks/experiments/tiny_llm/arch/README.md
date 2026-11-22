# ESP32-S3 architecture planner

**Estimates, not measurements.** This is a standard-library-only sizing and
ranking tool. It does not train, download models, export firmware, access a
board, or establish conversational quality. Tests instantiate the real default
TinyLM on CPU solely to count parameters; there is no forward/backward pass.

## What to pick first

Use **d384, L8, H6, KV6, hidden1024, vocab4096, ctx256, W1A8, int4 tied
embedding/head, int8 KV** as the evidence-backed starting point. Eden measured
10.96 story tokens/s with the current firmware (`chip_run.py --autoload 3`, 3 seeds;
9.52 without autoload) and a 2,765,216-byte export for seed 42; the parent README
reports three-seed validation loss 1.5376 ± 0.0031. This is evidence for short
English TinyStories, **not for answering questions or holding a useful chat**.
The 256-token context includes the prompt and generated conversation; current
firmware stops when full, with no sliding window.

Next architecture experiment: retain the int4 head and W1A8, compare KV2 GQA
against KV6, then consider a longer context. This reduces KV storage but needs
new training/export/attention support and quality checks. Longer context is not
free: attention latency and training memory grow, and current firmware has
context/scratch limits. Test FFN-only LittleBit targets 0.5 then 0.3 BPW as a
separate research choice, not as a guaranteed way to enlarge the model without
losing quality. Int2 head is a measured speed/size alternative with a larger
validation-loss penalty (parent README); the planner cannot price that penalty.

Do not choose the first Pareto row as “best chat model.” The objectives are
**more actual parameters and more decode tokens/s**, not accuracy. Different
vocabularies change tokens per word; different contexts change capability;
factored and dense parameters are not equivalent. Compare these in separate
grids with fixed vocab/context/precision before using the frontier for decisions.
Conversational data, instruction tuning, validation prompts, repetition, factual
accuracy and multi-turn retention remain untested. A Mac fit flag is not proof
that a particular training implementation fits or has the estimated speed.

## Run from the repository root

```sh
python benchmarks/experiments/tiny_llm/arch/search.py --current-only --top 10
python benchmarks/experiments/tiny_llm/arch/search.py --workload full --json
python benchmarks/experiments/tiny_llm/arch/search.py --grid my-grid.json --tokens 100000000 --batch 64 --seq 256
```

The default grid is 36 configurations: d256/384/512, L4/8/12, hidden768/1024,
head4/8, 64-wide MHA heads, vocab4096 and ctx256. `--current-only` removes known
firmware incompatibilities; it does **not** certify an export or exactness.
Every row must fit chip memory/partition budgets and the estimated 24 GiB Mac
training budget. No suitable rows produces an empty frontier and rejection
counts, not a fallback that violates the limits. Tables show the fastest ten
frontier entries; JSON contains the entire frontier and firmware gaps.

Custom grid JSON maps `Config` fields to nonempty lists. Unspecified fields use
the measured int8-head default. All combinations must be valid; invalid input
fails explicitly. Example (four candidates, all require new GQA support):

```json
{
  "layers": [8, 12],
  "kv_heads": [2],
  "head_bits": [4],
  "weight_bits": [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0.5, 0.5, 0.5]]
}
```

`weight_bits` is ordered **q, k, v, o, w1, w3, w2**. Each value is 1 (dense
binary) or a target BPW strictly between 0 and 1 (a factored proposal). Head
and KV bits are 2, 4 or 8. Heads must divide dim into even rotary widths, and
kv_heads must divide heads. The Python API in `planner.py` exposes `Config`,
`size_model`, `latency`, `memory`, `training`; `search.py` exposes `search` and
`pareto`. Import by file path using `importlib.util.spec_from_file_location`
(register the module in `sys.modules` before executing it, for dataclasses).

## Counts and flash contract

The real default model has **15,735,168 parameters**, with **14,155,776** in
block projections. Tied embedding/head is counted once; RMSNorm weights count,
RoPE and nonlinear tables do not count as learned parameters. GQA changes K/V
output width to `kv_heads * (dim / heads)` without shrinking Q or O.

Dense binary rows are padded to 32 bits and have int32 output multipliers.
Flash includes the 32-byte header, tied packed embedding, int32 embedding
scales/norms, int16 RoPE tables, 2048-entry exp and 1024-entry sigmoid uint16
tables, projection scales and metadata. Int4/int2 exports add 16 bytes per
projection. Default int8/int4/int2 head totals reproduce the recorded
**3,550,752 / 2,765,216 / 2,372,000 bytes** exactly.

For LittleBit-like proposals, each matrix is U[out,rank] V[rank,in], both
binary, with FP16 row/column/latent scale vectors. Rank is the largest integer
meeting the requested BPW **including scales and row padding**; impossible
rank-one budgets fail. `achieved_bpw` reports the actual representation rate.
There is no residual factor branch. Trainable parameters count factor entries
and their learned scales; `dense_parameters` reports the unfactored equivalent.
The 16-byte per-projection planning descriptor is additional to target BPW.
These are exact arithmetic counts for this specified proposed layout, **not an
implemented LittleBit file format**. Integer scale conversion, alignment beyond
word padding, metadata encoding and kernels require a real export design.
No SamsungLabs implementation was copied.

## Latency assumptions and evidence

Calibration reads `../results/chip_s3_binary_d384l8_s42_symbex_al3.json` by default
(`--firmware autoload`: the current firmware, `chip_run.py --autoload 3`, data cache
autoload over the model) or `../results/chip_s3_binary_d384l8_s42_symbex.json`
(`--firmware plain`): ESP32-S3 N16R8, 240 MHz, lib_symbex PIE, model and KV in PSRAM.
Rounded parts without autoload are linear 55.8 ms, attention 34.1 ms, head 28.0 ms and
rest 10.6 ms; with autoload linear 43.7 ms, attention 34.0 ms, head 26.2 ms, rest 10.3 ms
(autoload speeds up the linears, not attention).

- Linear cost scales by executed W1A8 binary products, including both factor
  matmuls; all activations stay 8-bit. Factor scale/requantization cost is
  unknown and omitted, so LittleBit speed projections are optimistic.
- Attention separates fixed per-token RoPE/cache work from work proportional
  to occupied context × query heads × head dimension × layers. `mean` averages
  positions 0..ctx-1; `full` uses ctx occupied tokens. GQA and low-bit KV do not
  automatically get proportional speedups; unpacking and repeated queries
  still cost work. They save modeled storage, not guaranteed latency.
- Head cost scales with vocab × dim × head bits. This deliberately simple
  model underestimates packed-head overhead: int4 is predicted at half the
  int8 head time, whereas it measured 14.681 ms; int2 measured 8.435 ms.
  Tests check **whole-token totals** within 5%, not each part within 5%.
- Norm/embedding and SwiGLU scale with layers × dim and layers × hidden.
- `story` infers effective occupancy from the recorded story mean (plain 118.0 ms,
  autoload 103.9 ms)
  and scales it with context capacity. It is a comparison scenario, not a
  prediction for arbitrary prompt lengths. The plain 8.481 story tokens/s must not
  be confused with reciprocal mean-window latency (128.489 ms, ~7.78 tok/s).

Tests compare mean, full and story totals against all three seed-42 head
variants without autoload and the int8 / int4 heads with autoload (no int2 autoload
run exists) within 5%. That is calibration consistency, **not held-out predictive
accuracy**. Cache locality, PSRAM contention, bandwidth saturation, different
head widths and larger workspaces can invalidate extrapolation.

## Placement and limits

Budgets use binary units: 512 KiB SRAM, 8 MiB PSRAM, 16 MiB physical flash.
The existing model partition is only **5 MiB** (`chip/partitions.csv`); the tool
does not silently use all flash. Changing that partition needs separate work.

- PSRAM: full model copy (including RoPE/nonlinear tables), packed K and V,
  int32 per-token/per-KV-head scales, 16-byte read guard, binary row counts.
- SRAM: int32 residual and Q/K/V/output buffers, temporary/requantized vectors,
  int64 scratch/logits, softmax scratch, block descriptors and the 64 KiB
  default linear LUT. Current code allocates that LUT even with lib_symbex.
- Reserved: 160 KiB SRAM, 256 KiB PSRAM, and flash below offset `0x310000`.
  SRAM reserve covers static workspaces, stacks, allocator and application;
  these are planning choices, not measured free-heap guarantees. Adjust
  keyword arguments to `memory()` / `search(memory_kwargs=...)` for a known
  application. Tokenizer/networking/application data need explicit budgets.

No automatic spilling or streaming is modeled because it changes latency.
The default KV payload is 1,572,864 bytes, **plus** 98,304 bytes of scales and
the guard. Default modeled PSRAM allocations agree with the recorded load
within 4 KiB before the safety reserve. Known firmware gaps include GQA,
packed KV, factored projections, non-64-wide heads, dimension alignment,
ctx > 258, hidden < 2×ctx, workspace < 4×ctx and inputs > 1024.
Memory feasibility does not establish numerical correctness or deployability.

## Mac training assumptions

The **user-supplied** M4 Pro 24 GB anchor is 1.59 s/step at batch128 × seq256
for d384L8. No corresponding raw MPS log is present here; the committed
`train_*.json` timings are not substituted as Mac timings. Anchor throughput
is `128*256/1.59` ≈ 20,609 tokens/s and 10,000 such steps cost about 4.42 h.

Training FLOPs/token are `6*(executed projection weights + vocab*dim) +
12*layers*seq*dim`, counting forward/backward and full-square attention.
Seconds/step scales by FLOPs and batch tokens; hours is N/tokens-per-second/3600.
Int2/int4 inference storage does not imply faster floating-point QAT training.
Direct-factor training excludes initialization/SVD, distillation/teacher,
validation, I/O, checkpointing and convergence changes. Constant MPS FLOP/s,
especially across batch sizes, is an unverified approximation.

The training-memory allowance uses 16 bytes/parameter for FP32 weights,
gradients and Adam states; FP32 saved activations; two FP32 logit-sized buffers;
two FP32 attention-sized buffers; and 4 GiB OS/framework reserve. It is neither
a measured peak nor a fit guarantee. Fused attention/checkpointing can reduce
it; QAT intermediates and MPS allocation can increase it. Smaller batches may
fit, but the predicted batch scaling still requires measurement.

## Tiny verification

```sh
python -m pytest -q tests/test_tiny_llm_arch.py
```

This covers real CPU parameter count, three recorded export sizes, packing and
factor budgets, invalid inputs, latency anchors, SRAM/PSRAM/flash boundaries,
Mac arithmetic, strict Pareto dominance/ties, a tiny grid and CLI output.
There is no training, full suite, firmware build or new hardware measurement.
