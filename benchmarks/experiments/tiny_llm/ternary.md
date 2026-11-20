# Ternary weights: host projection path

Ternary weights are an opt-in alternative to binary weights, not a new default.
The existing `train.py --weights ternary` selects `QLinear(..., "ternary")`.
`ternary.py` now owns its reusable quantizer: float master weights, row-wise
absmean scale floored at 1e-5, round-to-nearest-even and clamp to {-1, 0, +1}.
The straight-through derivative is identity, including at zero and saturation.
This is the BitNet b1.58 idea with tiny_llm's **per-row**, not per-tensor, scale.
The binary path and checkpoint state-dict parameter names are unchanged.

## CPU projection example

The modules below are loaded by file path, with no package installation.
Run from the repository root. This is a tiny projection, not model training.

```python
import importlib.util
from pathlib import Path
import torch

root = Path("benchmarks/experiments/tiny_llm")
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ternary = load("ternary")
integer = load("ternary_int")
codec = load("ternary_pack")
w = torch.tensor([[-1., 0., 1., 0., 1.]])
codes, sw = ternary.quantize(w)
q, sx = integer.quantize_input(torch.tensor([[-2., 0., 1., 3., 1.]]))
acc = integer.accumulate(q, codes)  # exact int64, no scales
y = integer.dequantize(acc, sx, sw)  # float rescaling outside the kernel
payload = codec.pack(codes[0].tolist(), "base3")
assert codec.unpack(payload, 5, "base3") == codes[0].tolist()
assert y.shape == (1, 1)
```

`accumulate` accepts full int8 inputs (-128 through 127) and ternary int8 codes.
Float agreement is against the quantized activation/weight path, not the
original dense matrix. The tests use absolute and relative tolerance 3e-6.
This is not a fixed-point scale/multiplier export for the full transformer.

## Bytes and packing

The local codecs map -1, 0, +1 to digits 0, 1, 2, low digit first. Each row
starts a new byte. Unused final digits must be zero; they are not weights.
The 2-bit codec rejects code 3; base-3 rejects byte values 243 through 255.
Lengths and padding are validated. Shapes and scales are stored separately.

| Payload | Weights per byte | Large-row bytes/weight | Relative to binary |
|---|---:|---:|---:|
| Binary | 8 | 0.125 | 1x |
| Local 2-bit ternary | 4 | 0.25 | 2x |
| Local base-3 ternary | 5 | 0.2 | 1.6x |

For R rows of C weights, payload bytes are R * ceil(C / weights-per-byte).
Add R * scale_bytes for scales; `bytes_per_weight(C, format, scale_bytes)`
includes row padding and optional scales. For C=384 and a four-byte scale,
binary costs 52 bytes/row, base-3 81, and 2-bit 100. Metadata, alignment,
embeddings, KV cache, activations and runtime scratch are excluded.
Five trits per byte is 1.6 bits/weight, not the information limit log2(3)
(approximately 1.585 bits). Zero weights still occupy a digit.

## Cost versus 1-bit — estimate, not a measurement

For the same row width, a scalar binary dot product needs C signed adds;
a ternary implementation that skips zeros needs approximately (1-z)*C adds,
where z is the zero fraction. This C reference does not implement sparse
skipping: it performs C decoded products/accumulations and base-3 extraction.
It reads about 1.6x the binary payload (2x for 2-bit), before scales/padding.
Base-3 division/remainder adds decoding work; 2-bit uses shifts and masks.
These are operation and storage **estimates**, not latency predictions.
They do not compare directly with the existing binary LUT8 or ESP32 PIE
kernels. Cache behavior, zero distribution and compiler lowering matter.
No tokens/s, energy saving or model-quality benefit is claimed by this lane.

## C and GGUF boundaries

`c/ternary.c` / `c/ternary.h` consume local base-3 rows, returning int64 dots.
They validate the whole payload before output writes and allocate no heap.
The host test compiles with `/usr/bin/clang`; this is not an ESP32 board test.
Callers own correctly sized, nonoverlapping input/output allocations.

GGUF `importers/gguf.py` additionally reads TQ1_0 (54 bytes/256 weights,
0.2109375 bytes/weight) and TQ2_0 (66 bytes/256, 0.2578125 bytes/weight).
Those totals include their fp16 block scales. Both require row widths
divisible by 256. Their interleaved layouts are **not** the local codecs.
The decoder cites llama.cpp's format reference in its comment. TQ2 code 3
follows GGML's decoder (+2 times scale); normal ternary quantizers do not
emit it. GGUF tensor decoding does not supply model architecture mapping.

No real ThoxMicro-1bit-9M file was inspected or loaded. A model's marketing
name does not establish its GGML type, tokenizer or activation semantics.
The GGUF CLI still rejects whole-model import. Legacy `intsim.py` model.bin
export, full C transformer, firmware, board speed, energy and real pretrained
quality remain unverified and are not enabled by these projection helpers.

## Focused verification

Run one named file with the existing environment, for example:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /Users/obed/repos/Symbex-1/.venv/bin/python -m pytest -q tests/test_ternary_c.py
```

- `test_ternary_quant.py`: forward, identity backward and 12 toy SGD steps.
- `test_ternary_int.py`: Python-integer oracle, float agreement and boundaries.
- `test_ternary_pack.py`: exhaustive single-byte patterns, tails and errors.
- `test_ternary_c.py`: exact host parity, int8 extremes and rejection safety.
- `test_import_gguf.py`: synthetic TQ blocks plus existing GGUF regressions.
- `test_ternary_doc.py`: executable example and storage arithmetic.
