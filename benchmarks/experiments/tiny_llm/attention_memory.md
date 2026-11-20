# GQA and rolling KV reference

This lane adds **host references**, not a firmware integration or a speed claim.
Existing `intsim.py`, `c/tinylm.c`, chip code and the binary export ABI are unchanged.
PR #105's sub-8-bit KV implementation is not duplicated here.

## Python entry points

Run from `benchmarks/experiments/tiny_llm` (or add that directory to the import path):

```python
import torch
from train import TinyLM
from gqa_intsim import integerize, forward
from gqa_rolling import RollingModel

torch.manual_seed(4)
model = TinyLM(16, 128, 1, 2, 64, 4, "binary", kv_heads=1).eval()
params = integerize(model, "binary")
scores, scale = forward(params, torch.tensor([[2, 7, 1, 5]]))
state = RollingModel(params, window=4, max_positions=12)
token = 2
for _ in range(12):
    scores, scale = state.step(token)
    token = int(scores.argmax())
```

Only binary/ternary integer weights and **64-wide attention heads** are supported.
Adjacent groups of `heads / kv_heads` query heads share one KV head; MQA is
`kv_heads=1`. The existing float GQA option fixes head width to 64; MHA uses
`dim / heads`, so choose `dim = heads * 64` for comparable architectures.
The additive whole-sequence path matches existing intsim exactly for MHA. Tiny
random binary-model logits are checked against the float GQA model with absolute
tolerance 0.06, zero relative tolerance; this is not trained-model quality evidence.

The legacy checkpoint CLI still rejects GQA and its model.bin writer must not be
used on these parameter dictionaries. Construct the matching float model, load its
state dict, then call the additive `integerize`. No checkpoint migration is provided.

## Rolling semantics and bounds

`RollingAttention.step` accepts one stream: Q `[heads,64]`, K/V `[kv_heads,64]`,
Q12 int64 tensors. Inputs have absolute value at most 2^19. Persistent storage is
int8 K/V and int32 maxima per KV head, token and layer. Window is 1..512; rolling
heads are 1..64 and divisible by positive KV heads. Cache scales are 1..2^20.
Reference kernels validate these numerical domains to bound their accumulators.

RoPE uses **absolute token positions**, never `position % window`. Keys are rotated
once, before storage; ring slots alone wrap. Export-time `rope_tables` uses floats
to produce Q14 integer tables. All attention and generation math then uses integers
(existing intsim uses checked float64 matrix products to implement exact integer
products). `max_positions` is an explicit generation limit: table exhaustion raises
an error, rather than silently restarting positions. RoPE storage grows with this
limit and is **not** included in the KV table.

Before eviction, full-model outputs equal prefix recomputation. After eviction,
this is sliding-window attention over retained layer states, **not** recomputation
of the cropped prompt: retained states still encode earlier histories. Tests check
an independent chronological attention oracle after multiple wraps, plus repeatable
greedy generation beyond the training context. No claim about long-context quality.

## C host reference

`c/gqa_attention.h` exposes one-token attention on quantized, already-rotated Q/K/V;
`c/gqa_rolling.h` adds Q12 input rotation, quantization and ring state. Header comments
define layouts, sizes, return codes and buffer ownership. C takes caller-exported
Q14 RoPE and uint16 exp tables. It does not use floating-point operations.

The ring's persistent buffers are caller-owned; its struct starts at `position=0`.
Reset that field for a new stream. Both implementations gather chronological scratch;
the C reference allocates it per step and has sizeable stack arrays. It is deliberately
**not suitable for the firmware hot path yet**. A direct ring-stride kernel and a
caller-supplied scratch arena are future integration work, not measured optimizations.

Host tests compile with `/usr/bin/clang -std=c99 -O2 -Wall -Wextra -Werror` and compare
every output integer against Python; rolling tests compare cache bytes and maxima too.
They cover MHA/GQA/MQA, single-token windows, zero input, signed extrema, maximum
scales, length 512, repeated wraps, reset and rejection without output mutation.
No board run, sanitizer campaign, firmware build, latency or energy measurement was run.

## Memory table

From the repository root:

```sh
python benchmarks/experiments/tiny_llm/attention_memory.py
python benchmarks/experiments/tiny_llm/attention_memory.py --kv-heads 6 1 --ctx 256 --bits 8 --json
```

Defaults: dim 384, 8 layers, 6 query heads; KV heads 6/3/2/1, contexts 256/512,
bits 8/4/2. Each separately packed K/V head row rounds up to a byte. Two int32
scales per head and layer are included. For default shape and context 256, MHA
uses 1,572,864 payload bytes and 1,671,168 bytes including scales; MQA uses 278,528
bytes including scales (6x smaller). These are script-tested arithmetic, not chip
allocation measurements. The legacy memory budget includes an additional 16-byte
read guard; this table does not. Structs, allocator overhead, RoPE, scratch, model
weights and other working memory are excluded.

Sub-8-bit rows are **hypothetical sizing with the stated scale convention**, not
the ABI or measured allocation of PR #105. No quantizer/packing codec is added.

## Tiny checks and remaining work

Run only the lane's individual test files with the project Python and `-m pytest -q`:
`tests/test_tiny_llm_gqa_intsim.py`, `tests/test_tiny_llm_gqa_c.py`,
`tests/test_tiny_llm_gqa_rolling.py`, `tests/test_tiny_llm_gqa_rolling_c.py`,
`tests/test_tiny_llm_attention_memory.py`, `tests/test_tiny_llm_attention_doc.py`.
All use tiny CPU tensors; C tests compile only the standalone references.

No speculative attention-score early exit is added: dropping nonzero softmax
weights would change the exact contract. The existing zero-tail exp lookup and
explicit causal/window length bounds are sufficient for this reference.

Remaining: GQA-aware checkpoint CLI/export ABI, firmware integration and scratch
budget, trained-model validation, then owner-approved ESP32-S3 exactness and timing.
Local-only scope means no PR, push, Team sync comment or roadmap/UPDATES edit.
