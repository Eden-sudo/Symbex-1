# LittleBit experimental lane

Independent paper-based implementation; no SamsungLabs code copied:
[LittleBit §3](https://arxiv.org/html/2506.13771v5),
[LittleBit-2 Algorithm 1](https://arxiv.org/html/2603.00042v2).

`littlebit.py` represents W as `diag(h) sign(U) diag(l) sign(V).T diag(g)`.
`FactorLinear.from_dense(W, rank, residual=False, itq_iters=None, seed=0)` splits
truncated SVD singular values evenly, optionally applies shared Joint-ITQ, and
extracts rank-one magnitude envelopes with SVD. A second pair fits the primary
residual. Forward uses two projections, not a dense reconstructed weight.
Signs use identity STE (zero maps to +1), as requested for this experiment;
this is not the paper's SmoothSign training recipe. `dense()` is an oracle.
Higher rank and ITQ are not universal guarantees of lower binary product error.

Explicit opt-in: load `littlebit_model.py`, then call
`replace_linears(model, rank=64, names=("w1", "w3"), residual=False)` before
constructing the optimizer. It replaces those projections in every TinyLM block,
preserves unselected layers and embedding/head, and uses int8 activation fake
quantization. It accepts unrotated 8-bit-input projections only. This is a Python
conversion API, not a new `train.py` conversion CLI/checkpoint format. The round-2
container below archives mixed model payloads separately from `model.bin`.

`littlebit_budget.py` counts byte-padded U and V-transpose signs plus all three
scale vectors (FP16 by default); headers are excluded. `rank_for_bpw` selects the
largest fitting rank and rejects infeasible budgets. Rank is **per pair**: a
residual pair doubles the payload at fixed rank. For 1024×384, rank 64 is
14,208 bytes / 0.2890625 BPW without a residual, or twice that with one.

`littlebit_int.py` converts trained scales with a specified input bound (default
1). Inputs represent `q * input_bound / 127`; output is Q12. It applies signed
Q20 int32 multipliers before V, between factors, and after U, with half-away
rounding and symmetric int8 saturation at the first two stages. Conservative
bounds set the quantization units; they are not activation calibration. Values
outside the input bound can saturate. Overflowing scales/output bounds are
rejected. Both pairs contribute rounded Q12 outputs. This is a layer-only
integer reference, not wired into `intsim.py`'s complete transformer.

`littlebit_blob.py` documents and reads/writes the standalone **LBT1** layout:
40-byte versioned little-endian header, packed LSB-first signs, then int32
input/latent/output multipliers per path. It is not tiny_llm `model.bin`.
`c/littlebit_blob.c` reads this exact layout and reuses `tinylm_factor.h`'s
requantizer. The allocation-free reference recomputes latent values for each
output; it is deliberately not a speed kernel. Host tests compile with
`/usr/bin/clang` and compare every output against Python, including unaligned
blobs, padding, signed scales and residual paths.

Run `python benchmarks/experiments/tiny_llm/littlebit_analysis.py` for the d384,
8-layer, hidden-1024, vocab-4096 budget (`--residual` for two pairs). It separates
FP16-scale payloads from int32-scale blobs, includes one shared int8 head, and
compares with the current 1-bit projections. Operation estimates assume cached
latent execution, **not** the allocation-free C reference. They exclude attention,
norms, nonlinearities, quantization, KV cache and tables. A target can be
infeasible (notably two-pair 0.1 BPW square projections at this width).

## Round 2

### Distillation

`littlebit_distill.distillation_loss(student, teacher, temperature=..., ...)`
computes `T² * KL(teacher || student)`, averaged over all tokens. Teacher logits
and optional hidden targets are detached; student gradients remain connected.
Optional `student_hidden`, `teacher_hidden`, and `hidden_weight` add mean squared
error over identically shaped hidden states. Half/bfloat values are promoted to
float32; double precision is retained for gradient checks.

The existing `train.py --distill` path now uses this helper. Add
`--distill-hidden-weight 0.1` to match the final normalized hidden state, with
the existing `--distill-t` temperature. The complete loss is
`(1-alpha)*CE + alpha*(T²*KL + hidden_weight*MSE)`.
Default hidden weight is zero; ordinary model forward still returns logits.
`model(tokens, return_hidden=True)` returns `(logits, final_normalized_hidden)`.
Hidden matching requires equal teacher/student widths and token shapes.
No checkpoint download or production training was run.

### Sensitivity and allocation

`littlebit_sensitivity.probe(float_model, bpws, residual=False, itq_iters=None)`
visits bias-free float linear projections and returns named candidate lists of
`rank`, `bits`, and `error`. Error is squared Frobenius weight reconstruction
error, **not** validation-loss damage or calibrated output error. Costs include
byte-padded signs and FP16 scales, excluding headers. Infeasible BPWs raise.

`allocate(candidates, budget_bits)` returns `(selection, used_bits)`. It starts
with the cheapest candidate per layer and greedily buys the largest reduction
in summed error per additional bit, considering jumps over poor intermediate
ranks. It is a heuristic, not a globally optimal knapsack solver. It never
exceeds the budget; arbitrary budgets can leave unused bits because ranks have
discrete packed costs. A constructed sensitivity curve uses exactly its budget
and beats uniform ranks; this does not prove superiority on real models.

### Mixed-model container

`littlebit_container.export(model, path, input_bound=1.0)` writes **LBM1**;
`read(path)` returns a dictionary of named typed payloads. In-memory equivalents
are `dumps(model)` and `loads(data)`. Every block's seven projections must be
factored or unrotated W1A8 binary; unsupported projections raise, not disappear.
Factor entries use the existing integer **LBT1** payload (including residual
pairs). Binary entries store packed row signs and float32 row scales. Auxiliary
state includes the embedding/shared-head weights, norms, and rope buffers as
float32. This is mixed-format archival data, not a fully integer transformer.

The 32-byte little-endian header records magic, version, JSON TOC length, entry
count and reserved zeros. A 32-byte SHA256 digest protects the TOC; each entry
records its name, kind, offset, size and payload SHA256. Payloads begin on
16-byte boundaries with zero padding. Readers reject unknown versions/types,
duplicate names, invalid bounds/padding, trailing bytes and checksum failures.
SHA256 detects corruption, not authenticity. Model architecture options are
not reconstructed: the Python reader returns payloads, **not a runnable model**.
There is no C container loader or compatibility with tiny_llm `model.bin`.

### Portable bit-plane C variant

`c/tinylm_factor_popcount.h` adds `tinylm_factor_popcount`, with the reference
factor kernel's inputs/outputs plus a caller-owned workspace of
`8 * ceil(rank/8)` bytes. It computes latent int8 values once, packs their signs
and seven magnitude planes, and uses masked XNOR plus portable SWAR popcount
on the rank dimension. A single binary XNOR would be wrong for int8 latents.
Tail padding is masked by zero magnitude bits. Signed multipliers, saturation,
rounding and output widths match `tinylm_factor`.

The host test compiles with `/usr/bin/clang`, strict C99 warnings and undefined
behavior sanitizer. It compares all latents and outputs against the reference
for 300 randomized tiny cases, including odd ranks, signed multiplier extremes,
shift boundaries, padding and workspace guards. This is a performance candidate:
**no faster-runtime claim, timing, firmware integration or board test**.

### Quality proxy

Run `python benchmarks/experiments/tiny_llm/littlebit_quality.py` for a seeded
512×512 random linear teacher and four random input vectors, CPU/single-thread.
It prints weight MSE and output MSE for a row-scaled 1-bit baseline and LittleBit
at **0.1, 0.3, 0.55 and 1.0 BPW**, with every ITQ/residual combination. Requested
and actual BPW are distinct; FP16 scale and sign-padding overhead are counted.
Infeasible targets are printed rather than silently exceeding their budgets.
Errors use float scales; these are representation-cost estimates, not measured
FP16 serialization error. Raw errors are never replaced by a monotone envelope.

Tests run the smaller 128×128 CLI case and check nonincreasing error (within
floating-point tolerance) only on a constructed exact sign-rank-one teacher.
That case is already reconstructible at the lowest feasible rank; it does not
establish strict improvement or monotonicity for random/pretrained models.

### Evidence and limits

Round-2 focused tests: `tests/test_littlebit_distill.py` (values and gradients),
`tests/test_littlebit_sensitivity.py` (probe oracle, exact constructed budget),
`tests/test_littlebit_container.py` (mixed toy model and corruption),
`tests/test_littlebit_popcount_c.py` (host C equality),
`tests/test_littlebit_quality.py` (script and constructed monotonicity), and
`tests/test_littlebit_docs.py` (documented modules/test paths exist).

Evidence remains synthetic CPU/host-only. No pretrained model conversion,
distillation training run, validation perplexity, board execution, board memory,
latency, energy or end-to-end speedup was measured. No firmware, existing model
format or KV cache changed. The only training CLI addition is optional hidden
distillation; normal logits distillation already existed before round 2.
