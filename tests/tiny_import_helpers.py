"""Synthetic format writers and file-path imports; no network or model assets."""
import importlib.util
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "benchmarks/experiments/tiny_llm"


def load(name):
    path = TINY / (name + ".py")
    spec = importlib.util.spec_from_file_location("test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(path, *, shared=True, kv=1, dim=8, heads=2):
    """Independently spell out run.c v0 layout, with distinguishable weights."""
    hidden, layers, vocab, seq = 12, 2, 7, 4
    kd, hd = dim // heads * kv, dim // heads
    shapes = [
        ("token_embedding_table", (vocab, dim)), ("rms_att_weight", (layers, dim)),
        ("wq", (layers, dim, dim)), ("wk", (layers, kd, dim)),
        ("wv", (layers, kd, dim)), ("wo", (layers, dim, dim)),
        ("rms_ffn_weight", (layers, dim)), ("w1", (layers, hidden, dim)),
        ("w2", (layers, dim, hidden)), ("w3", (layers, hidden, dim)),
        ("rms_final_weight", (dim,)), ("freq_cis_real", (seq, hd // 2)),
        ("freq_cis_imag", (seq, hd // 2)),
    ]
    if not shared:
        shapes.append(("wcls", (vocab, dim)))
    rng = np.random.default_rng(17)
    values = {}
    with path.open("wb") as f:
        f.write(struct.pack("<7i", dim, hidden, layers, heads, kv, vocab if shared else -vocab, seq))
        for name, shape in shapes:
            a = rng.normal(0, .15, shape).astype("<f4")
            if name.startswith("rms"):
                a += 1
            values[name] = a
            f.write(a.tobytes())
    if shared:
        values["wcls"] = values["token_embedding_table"]
    return values
