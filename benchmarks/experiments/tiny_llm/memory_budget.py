"""Size the default integer exporter layout and the S3 working buffers."""
import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("attention_memory", Path(__file__).with_name("attention_memory.py"))
_attention = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attention)


def _packed_linear_bytes(n_out, n_in):
    return 4 * n_out * ((n_in + 31) // 32)


def budget(dim, layers, heads, hidden, seq, vocab, weights, head):
    """Return layout and allocation sizes, not a firmware compatibility guarantee.

    Both binary and ternary use one bit per linear weight in the current writer:
    its positive mask loses ternary zeros. The bit count excludes row padding and
    multipliers. The fp32 training head exports an int8 shared table.
    """
    for name, value in (("dim", dim), ("layers", layers), ("heads", heads),
                        ("hidden", hidden), ("seq", seq), ("vocab", vocab)):
        if type(value) is not int or not 0 < value <= 2**31 - 1:
            raise ValueError(f"{name} must be a positive signed int32")
    if weights not in ("binary", "ternary"):
        raise ValueError("weights must be binary or ternary")
    if head not in ("fp32", "int4", "int2"):
        raise ValueError("head must be fp32, int4 or int2")
    if dim % heads or (dim // heads) % 2:
        raise ValueError("dim must be divisible by heads with an even head width")
    if head == "int4" and dim % 32:
        raise ValueError("int4 head requires dim divisible by 32")
    if head == "int2" and dim % 64:
        raise ValueError("int2 head requires dim divisible by 64")

    linear = layers * (4 * _packed_linear_bytes(dim, dim)
                       + 2 * _packed_linear_bytes(hidden, dim)
                       + _packed_linear_bytes(dim, hidden))
    head_bits = {"fp32": 8, "int4": 4, "int2": 2}[head]
    embedding = vocab * dim * head_bits // 8 + 4 * vocab
    # Header, shared table, final norm, rotary and lookup tables, block norms,
    # padded linear matrices, row multipliers and version 3/4 metadata.
    model = (32 + embedding + 4 * dim + 2 * seq * (dim // heads)
             + 2 * 2048 + 2 * 1024 + layers * 8 * dim + linear
             + layers * 4 * (5 * dim + 2 * hidden)
             + (layers * 7 * 16 if head != "fp32" else 0))
    big = max(dim, hidden)
    # FAST buffers, output scores and four persistent S3 scratch arrays.
    internal = (20 * dim + 21 * big + 2 * seq + 512 * (big // 8)
                + 4 * (big // 32 + 1) + 8 * vocab + 4096)
    return {"model_bin_bytes": model, "linear_weight_bytes": linear,
            "embedding_head_bytes": embedding,
            # The legacy allocator adds a 16-byte guard to the common KV layout.
            "kv_cache_bytes": _attention.kv_bytes(heads, heads, seq, 8, dim, layers)["kv_total_bytes"] + 16,
            "per_token_internal_ram_bytes": internal, "bits_per_linear_weight": 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("dim", 384), ("layers", 8), ("heads", 6),
                          ("hidden", 1024), ("seq", 256), ("vocab", 4096)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--weights", choices=("binary", "ternary"), default="binary")
    parser.add_argument("--head", choices=("fp32", "int4", "int2"), default="fp32")
    args = parser.parse_args(argv)
    try:
        result = budget(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
