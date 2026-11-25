"""
attn_cut.py -- reading fewer keys: drop the attention weights p < p_max e^-n (exp-table index >= 128 n) in the
integer model (intsim.py) and measure the loss and the share of keys before the last 32 positions still read.

    python attn_cut.py runs/binary_d384l8_s42 runs/binary_attn_k1v2_tau_r32_s42 ...

Runs trained with --attn-temp get the KV they were trained with (T2: 1-bit keys x pi / 2, 2-bit values, 32
int8 recent). Validation only (1024 windows, as intsim.py). Writes results/attn_cut.json (README "Reading the cache").
"""
import argparse
import json
import os

import torch

import intsim as I
import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
CUTS = (None, 8, 6, 4, 3, 2, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--windows", type=int, default=1024)
    args = ap.parse_args()
    import sentencepiece as spm
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = T.pick_device(allow_mps=False)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(T.DATA, "tok4096.model"))
    val = T.load_split("val", dev)
    out_path = os.path.join(HERE, "results", "attn_cut.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        model, a = I.load_model(run, sp.get_piece_size(), dev)
        P = I.to_device(I.integerize(model, a["weights"]), dev)
        base_kv = {"k_bits": a["k_bits"], "v_bits": a["v_bits"], "k_scale": a["k_scale"],
                   "recent": a["kv_recent"]} if a.get("attn_temp") else {}
        rows = []
        for n in CUTS:
            P["kv"] = dict(base_kv, p_cut=n)
            I.ATTN_STATS = []
            loss = I.int_loss(P, val, a["seq"], args.windows, 16)
            kept = sum(I.ATTN_STATS) / len(I.ATTN_STATS)
            I.ATTN_STATS = None
            rows.append({"cut_nats": n, "val_loss_integer": round(loss, 4), "far_keys_kept": round(kept, 3)})
            print(name, rows[-1], flush=True)
        res[name] = {"kv": base_kv or "int8", "val_windows": args.windows, "cuts": rows}
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
