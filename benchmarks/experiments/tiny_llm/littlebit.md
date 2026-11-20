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
conversion API, not a new `train.py` CLI/checkpoint format or full-model exporter.

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

Evidence is limited to tiny synthetic CPU tests, eight optimizer steps on a
two-block toy TinyLM, blob round-trips and host C/Python equality. No pretrained
model conversion, distillation, validation perplexity, board execution, board
memory, latency, energy or end-to-end speedup has been measured in this lane.
No firmware, existing model format, KV cache or production training CLI changed.
