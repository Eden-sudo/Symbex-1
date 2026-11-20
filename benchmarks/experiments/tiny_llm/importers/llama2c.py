"""Read little-endian llama2.c run.c v0 checkpoints, without fetching files."""
import math
import struct
from pathlib import Path

import numpy as np


def read_checkpoint(path):
    """Return config and named float32 arrays (read-only mmap views).

    v0 has seven int32s, then layer-major tensors in run.c memory_map_weights
    order, including the two obsolete rotary tables before an optional head.
    Positive vocab ties the classifier; negative vocab stores it separately.
    """
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(28)
    if len(header) != 28:
        raise ValueError("truncated llama2.c v0 header")
    d, h, layers, heads, kv, vocab, seq = struct.unpack("<7i", header)
    if min(d, h, layers, heads, kv, seq) <= 0 or vocab == 0:
        raise ValueError("llama2.c dimensions must be positive (vocab may be negative)")
    if d % heads or heads % kv or (d // heads) % 2:
        raise ValueError("invalid attention dimensions: require even head width and heads divisible by KV heads")
    config = dict(dim=d, hidden_dim=h, n_layers=layers, n_heads=heads,
                  n_kv_heads=kv, vocab_size=abs(vocab), seq_len=seq,
                  shared_classifier=vocab > 0)
    kd, hd = d // heads * kv, d // heads
    shapes = [
        ("token_embedding_table", (abs(vocab), d)),
        ("rms_att_weight", (layers, d)),
        ("wq", (layers, d, d)), ("wk", (layers, kd, d)),
        ("wv", (layers, kd, d)), ("wo", (layers, d, d)),
        ("rms_ffn_weight", (layers, d)),
        ("w1", (layers, h, d)), ("w2", (layers, d, h)),
        ("w3", (layers, h, d)), ("rms_final_weight", (d,)),
        ("freq_cis_real", (seq, hd // 2)), ("freq_cis_imag", (seq, hd // 2)),
    ]
    if vocab < 0:
        shapes.append(("wcls", (abs(vocab), d)))
    expected = 28 + 4 * sum(math.prod(shape) for _, shape in shapes)
    if path.stat().st_size != expected:
        raise ValueError(f"llama2.c v0 size mismatch: expected {expected} bytes, got {path.stat().st_size}")
    data = np.memmap(path, dtype="<f4", mode="r", offset=28)
    tensors, offset = {}, 0
    for name, shape in shapes:
        count = math.prod(shape)
        tensors[name] = data[offset:offset + count].reshape(shape)
        offset += count
    if vocab > 0:
        tensors["wcls"] = tensors["token_embedding_table"]
    return config, tensors
