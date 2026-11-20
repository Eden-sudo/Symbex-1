"""KV sizing, not a firmware allocator or a sub-8-bit codec implementation."""
import argparse
import json


def kv_bytes(heads=6, kv_heads=6, ctx=256, bits=8, dim=384, layers=8):
    """Packed per-head rows + two int32 scales per KV head and layer.

    Payload rows round up independently to bytes. No guard, struct, allocator,
    RoPE tables, temporary attention scratch or model weights are included.
    Lower bit widths are hypothetical storage with this scale convention.
    """
    for name, value in (("heads", heads), ("kv_heads", kv_heads), ("ctx", ctx),
                        ("bits", bits), ("dim", dim), ("layers", layers)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if heads % kv_heads or dim % heads or bits not in (1, 2, 4, 8):
        raise ValueError("require kv_heads dividing heads, heads dividing dim, bits in 1/2/4/8")
    head_dim = dim // heads
    payload = 2 * layers * kv_heads * ((head_dim * bits + 7) // 8)
    scales = 2 * layers * kv_heads * 4
    return {"heads": heads, "kv_heads": kv_heads, "ctx": ctx, "bits": bits,
            "payload_bytes_per_token": payload, "scale_bytes_per_token": scales,
            "kv_bytes_per_token": payload + scales,
            "payload_total_bytes": payload * ctx, "kv_total_bytes": (payload + scales) * ctx}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, nargs="+", default=[6, 3, 2, 1])
    parser.add_argument("--ctx", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4, 2])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    a = parser.parse_args(argv)
    try:
        rows = [kv_bytes(a.heads, kv, ctx, bits, a.dim, a.layers)
                for kv in a.kv_heads for ctx in a.ctx for bits in a.bits]
    except ValueError as exc:
        parser.error(str(exc))
    if a.json:
        print(json.dumps(rows))
    else:
        print("heads kv_heads ctx bits payload_B/token scales_B/token KV_B/token KV_total_B")
        for r in rows:
            print(*(r[k] for k in ("heads", "kv_heads", "ctx", "bits", "payload_bytes_per_token",
                                  "scale_bytes_per_token", "kv_bytes_per_token", "kv_total_bytes")))


if __name__ == "__main__":
    main()
