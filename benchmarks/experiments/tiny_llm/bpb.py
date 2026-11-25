"""Bits per byte of text for tiny_llm results, so models with different tokenizers compare on one scale.

    python bpb.py results/train_binary_d768l8_head4_s42.json results/intsim_binary_d768l8_head4_s42.json ...

Counts the UTF-8 bytes of the tokens that train.py's evaluate() scores (the same fixed, non-overlapping windows
of the validation split: --windows x --seq target tokens), then bits per byte = loss / ln 2 x tokens / bytes.
A story's first piece loses its leading space (SentencePiece's dummy prefix, which decoding drops); BOS, the story
separator, counts as one byte, like the newline between stories in a text file. Validation only: never reads test.
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "tinystories")
LOSS_KEYS = ("val_loss", "val_loss_float", "val_loss_integer")


def piece_bytes(sp):
    """UTF-8 length of every vocabulary id, and the same without a leading space (a story's first piece)."""
    full = np.zeros(sp.get_piece_size(), dtype=np.int64)
    first = np.zeros_like(full)
    for i in range(sp.get_piece_size()):
        if i == sp.bos_id():
            full[i] = first[i] = 1
        elif sp.is_byte(i):
            full[i] = first[i] = 1
        elif sp.is_control(i) or sp.is_unknown(i):
            continue
        else:
            text = sp.id_to_piece(i).replace("▁", " ")
            full[i] = len(text.encode("utf-8"))
            first[i] = len(text[1:].encode("utf-8")) if text.startswith(" ") else full[i]
    return full, first


def scored_bytes(ids, seq, windows, sizes, bos):
    """Bytes and tokens of the targets of the first `windows` windows (train.py evaluate's layout)."""
    full, first = sizes
    usable = min(windows, (len(ids) - 1) // seq)
    targets = np.concatenate([ids[r * seq + 1:r * seq + seq + 1] for r in range(usable)]).astype(np.int64)
    prev = np.concatenate([ids[r * seq:r * seq + seq] for r in range(usable)]).astype(np.int64)
    after_bos = prev == bos
    nbytes = np.where(after_bos, first[targets], full[targets]).sum()
    return int(nbytes), int(targets.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="train_*.json / intsim_*.json with a validation loss")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--windows", type=int, default=5120, help="train.py: --eval-batches x --batch (40 x 128)")
    ap.add_argument("--intsim-windows", type=int, default=1024, help="intsim.py scores 1,024 windows")
    ap.add_argument("--out", help="write every figure to this JSON")
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.data, "tok4096.model"))
    sizes = piece_bytes(sp)
    ids = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    ratio, out = {}, {"split": "val", "seq": args.seq, "windows": {}, "results": {}}
    for w in {args.windows, args.intsim_windows}:
        b, t = scored_bytes(ids, args.seq, w, sizes, sp.bos_id())
        ratio[w] = (b, t)
        out["windows"][str(w)] = {"tokens": t, "bytes": b, "bytes_per_token": round(b / t, 4)}
        print(f"{w} windows: {t:,} tokens, {b:,} bytes, {b / t:.3f} bytes per token")
    for path in args.results:
        r = json.load(open(path))
        w = args.intsim_windows if os.path.basename(path).startswith("intsim") else args.windows
        b, t = ratio[w]
        for k in LOSS_KEYS:
            if k in r:
                bits = r[k] / math.log(2) * t / b
                out["results"].setdefault(os.path.basename(path), {})[k] = {"nats_per_token": r[k],
                                                                              "bits_per_byte": round(bits, 4)}
                print(f"{os.path.basename(path)} {k} {r[k]:.4f} nats/token -> {bits:.4f} bits/byte")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
