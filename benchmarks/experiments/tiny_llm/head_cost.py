"""Head-only storage and latency ESTIMATES, not kernel or board measurements."""
import argparse
import itertools
import json

ANCHOR_MS = 28.0
ANCHOR_WEIGHTS = 4096 * 384


def packed_rows(rows, columns, bits):
    """Each row starts at a byte boundary; no metadata or allocator padding."""
    return rows * ((columns * bits + 7) // 8)


def head_cost(vocab, rank=None, bits=8, dim=384):
    """Dense or untied factored head, with hypothetical quantized row layouts.

    Factors are stored as rank x dim and vocab x rank (transposes of A/B).
    One int32 row multiplier is counted per output row of each projection.
    A separate int8 embedding is included for the untied factored variant.
    Dense storage assumes its embedding is tied to its head. No codecs implied.
    """
    if any(type(n) is not int or n <= 0 for n in (vocab, dim)):
        raise ValueError("vocab and dim must be positive integers")
    if type(bits) is not int or bits not in (1, 2, 4, 8):
        raise ValueError("bits must be 1, 2, 4 or 8")
    if rank is not None and (type(rank) is not int or not 1 <= rank <= min(vocab, dim)):
        raise ValueError("rank must be in [1, min(vocab, dim)]")
    shapes = [(vocab, dim)] if rank is None else [(rank, dim), (vocab, rank)]
    weights = sum(rows * columns for rows, columns in shapes)
    payload = sum(packed_rows(rows, columns, bits) for rows, columns in shapes)
    scales = 4 * sum(rows for rows, _ in shapes)
    embedding = 0 if rank is None else vocab * dim + 4 * vocab
    # Normalize against payload only to preserve the measured dense int8 anchor.
    # Scale metadata remains explicit rather than implying calibrated bandwidth.
    return {"vocab": vocab, "dim": dim, "rank": rank, "bits": bits,
            "head_payload_bytes": payload, "head_scale_bytes": scales,
            "separate_embedding_bytes": embedding,
            "model_table_bytes": payload + scales + embedding,
            "macs": weights, "intermediate_int32_bytes": 0 if rank is None else 4 * rank,
            "estimated_ms_payload": ANCHOR_MS * payload / ANCHOR_WEIGHTS,
            "estimated_ms_macs": ANCHOR_MS * weights / ANCHOR_WEIGHTS,
            "estimate_only": True}


def cost_table(vocabs=(1024, 2048, 4096), ranks=(None, 32, 64, 128), bits=(2, 4, 8)):
    return [head_cost(v, r, b) for v, r, b in itertools.product(vocabs, ranks, bits)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    rows = cost_table()
    if args.format == "json":
        print(json.dumps({"anchor": "Eden ESP32-S3: 4096 x 384 int8 head, 28 ms",
                          "rows": rows}, indent=2))
    else:
        print("Estimates only; payload and MAC ratios are separate hypotheses, not bounds.")
        print("| Vocab | Rank (dense = —) | Bits | Head payload B | Tables B | Est. payload ms | Est. MAC ms |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['vocab']} | {r['rank'] or '—'} | {r['bits']} | "
                  f"{r['head_payload_bytes']} | {r['model_table_bytes']} | "
                  f"{r['estimated_ms_payload']:.3f} | {r['estimated_ms_macs']:.3f} |")


if __name__ == "__main__":
    main()
