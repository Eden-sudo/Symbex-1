# Vocabulary and output-head experiments (host only)

The existing ESP32-S3 int8 head is 4096 x 384 and costs **28 ms**.
The roughly 118 ms story-token average is a different aggregation from the
128.5 ms mean over the timed windows in [README.md](README.md).
Eden already measured int4 at **14.7 ms** and int2 at **8.4 ms**.
Nothing here replaces those measurements or demonstrates a faster board kernel.

## Bounded vocabulary preparation

Run from this experiment directory, using a local **training** text file:

```sh
python prepare_vocab.py /path/to/train.txt --vocab 1024 --max-bytes 65536
```

The printed path is a fresh temporary directory retained for inspection; the
caller owns its cleanup. Supported requested vocabulary sizes are 1024, 2048
and 4096. The default slice is 64 KiB, with a hard maximum of 4 MiB plus one
lookahead byte. A truncated final story is discarded, including an incomplete
UTF-8 suffix. At EOF a final story does not need `<|endoftext|>`.

Every 100th nonempty story, starting with index zero, goes to validation.
SentencePiece sees only the other stories. At least two complete stories are
required. No official validation/test file is opened; the source argument must
be training data. Each stream is little-endian uint16 with BOS 1 before each
story, matching the preparation convention. There is no test.bin.

Tiny text may not support every requested BPE piece: `prepare.json` records
`requested_vocab` and `actual_vocab`; use the latter as the model vocabulary.
Changing tokenization invalidates old token IDs/checkpoints and changes tokens
per character. Compare quality per text unit as well as latency per token.

## Opt-in factorized model

The new module leaves `train.py` unchanged. Its floating head computes
`logits = (h @ A) @ B`, with A[dim, rank] and B[rank, vocab].
It is untied: the dense input embedding remains a separate learned parameter.

```python
from factorized_head import FactorizedTinyLM
import torch

model = FactorizedTinyLM(16, 8, 1, 2, 12, 4, "fp32", rank=3)
logits = model(torch.tensor([[1, 2, 3, 4]]))
assert logits.shape == (1, 4, 16)
```

Construct before creating the optimizer. Only `head="fp32"` is accepted;
block quantization options remain inherited. Rank must be a positive integer
no greater than min(dim, vocab). The head is randomly initialized, not an SVD
conversion preserving a trained dense model. To reload a state_dict, instantiate
the same subclass, architecture and rank first. Legacy train CLI checkpoints,
integer export, C model loading and firmware do not support this head.

## Two-stage integer reference

`cluster_embeddings` runs seeded CPU Lloyd k-means on int8 embedding rows in a
shared scale. It permits at most two million elements, 256 clusters and 32
iterations. Empty centers retain their previous value. Float centers determine
assignments; rounded int8 centers provide approximate routing scores.

```python
from cluster_head import cluster_embeddings, shortlist_logits, full_logits, greedy
import numpy as np

w = np.array([[10, 11], [12, 10], [-10, -11], [-12, -10]], dtype=np.int8)
h = np.array([1, 1], dtype=np.int8)
labels, centers = cluster_embeddings(w, 2, seed=9)
ids, scores, selected = shortlist_logits(h, w, labels, centers, k=2)
assert greedy(ids, scores) == int(full_logits(h, w).argmax())
```

First score all centers, select top-k clusters, then evaluate only their token
rows. Results are sparse ascending IDs with exact int64 dot products, not full
logits with artificial scores. Routing ties favor lower cluster IDs; greedy ties
favor lower token IDs. An empty shortlist raises ValueError.

Candidate scores equal the full integer head's corresponding scores. Greedy
selection equals the full head **only if its winning token is shortlisted**
(the lowest-ID maximizer on ties). Selecting all clusters recovers all logits.
There is no general recall guarantee, full-softmax equivalence or quality claim.
The reference uses unscaled int8 rows, not Eden's per-row scaled int4/int2 ABI.

`c/cluster_head.c` and `.h` expose the same operation without allocation.
Read the header for output capacities, nonaliasing and limits. Invalid inputs
return -1; an empty shortlist returns -2; neither writes outputs. The caller
must supply valid sized buffers. The host tests compile only this new reference
with clang and compare scores, IDs and routing order against Python. They do
not compile firmware or modify `c/tinylm.c` or `chip/src/main.cpp`.

## Cost table, estimates only

```sh
python head_cost.py
python head_cost.py --format json
```

The generated table covers vocab 1024/2048/4096, dense or rank 32/64/128, and
2/4/8-bit weights. Python also accepts 1 bit and other positive dimensions.
Rows are byte-aligned: payload = rows * ceil(columns * bits / 8).
Factors use stored shapes rank x dim and vocab x rank. Scale storage is four
bytes per output row. An untied factor head also counts a separate int8 dense
embedding plus its row scales; a dense head assumes a tied embedding.

These are hypothetical quantized layouts, not the actual float model's resident
memory or an implemented factor codec. Intermediate int32 bytes are listed
separately, and table storage excludes allocator overhead, full logits, routing
tables, KV cache and other transformer parameters.

`estimated_ms_payload = 28 * head_payload_bytes / (4096 * 384)`.
`estimated_ms_macs = 28 * macs / (4096 * 384)`.
These are separate bandwidth/operation hypotheses, **not bounds**, measurements
or calibrated forecasts. Metadata traffic, unpacking and launch overhead are
not modeled. In particular the dense int4 estimate is 14 ms, not Eden's measured
14.7 ms. No whole-token speedup or energy number is inferred.

## Small reproducible checks

From the repository root, use the configured Python environment and run only
the needed file: `python -m pytest -q tests/test_output_head_vocab.py`.
The other focused checks are:

- `tests/test_output_head_factor.py`: formula, shapes, gradients, three toy steps, reload.
- `tests/test_output_head_cluster.py`: clustering, integer oracle, shortlist misses and ties.
- `tests/test_output_head_cluster_c.py`: clang host parity, rejection, extreme accumulation.
- `tests/test_output_head_cost.py`: arithmetic, byte padding and table CLI.
- `tests/test_output_head_doc.py`: executable examples and referenced artifacts.

All evidence here is synthetic CPU/host evidence. Real-corpus quality, shortlist
recall, trained factor ranks, firmware integration, board latency and energy
remain unverified. Do not use the held-out test split to choose these variants.
