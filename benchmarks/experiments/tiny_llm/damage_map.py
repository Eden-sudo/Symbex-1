"""damage_map.py -- where does W1A1 lose, without training: switch the input precision of one group of linear layers
in one block of a TRAINED model and measure the validation loss (seconds per cell instead of 25 min per run):

    python damage_map.py binary_d384l8_s42 --to 1       # W1A8 model: binarize one input at a time (damage)
    python damage_map.py binary_a1p_k1_s42 --to 8       # W1A1 model: one input back to 8 bits (recovery)

Groups: qkv (q, k, v share one input), o, w13 (w1, w3), w2. Cells: every group x block, every group in all blocks,
and all groups at once (to check whether the single effects add up). No retraining: the network has not adapted to
the change, so this is a map to choose what to train, checked against the trained runs (LOGBOOK.md)."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import HERE, TinyLM, W2_ACTS, evaluate, load_split  # noqa: E402

GROUPS = {"qkv": ("q", "k", "v"), "o": ("o",), "w13": ("w1", "w3"), "w2": ("w2",)}


def load(run, dev):
    ck = torch.load(os.path.join(HERE, "runs", run, "model.pt"), map_location=dev, weights_only=False)
    a = ck["args"]
    m = TinyLM(4096, a["dim"], a["layers"], a["heads"], a["hidden"], a["seq"], a["weights"], a.get("head", "fp32"),
               tuple(a.get("int8_blocks", [])), a.get("acts", 8),
               {"ste_mode": a.get("ste", "identity"), "ste_k": a.get("ste_k", 2.0),
                "learn_thr": a.get("learn_thr", False)},
               tuple(a.get("a8_inputs", [])), W2_ACTS[a.get("w2_acts", "same")], a.get("ffn_act", "silu"),
               tuple(a.get("hadamard", [])), tuple(a.get("a8_blocks", []))).to(dev)
    m.load_state_dict(ck["model"])
    return m, a["seq"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run folder under runs/")
    ap.add_argument("--to", type=int, choices=(1, 8), required=True, help="precision set on the chosen inputs")
    ap.add_argument("--batches", type=int, default=20, help="validation batches of 32 windows")
    args = ap.parse_args()
    dev = "cuda"
    model, seq = load(args.run, dev)
    val = load_split("val", dev)
    layers = {(bi, g): [getattr(b, n) for n in names] for bi, b in enumerate(model.blocks)
              for g, names in GROUPS.items()}
    orig = {id(l): l.acts for ls in layers.values() for l in ls}
    for ls in layers.values():  # 1 bit with the clipped-STE options at inference is plain sign(x) x mean|x|
        for l in ls:
            l.act_opts = l.act_opts or {"ste_mode": "clip", "ste_k": 1.0}

    def loss(cells):
        for c in cells:
            for l in layers[c]:
                l.acts = args.to
        with torch.no_grad():
            v = evaluate(model, val, seq, args.batches, 32)
        for c in cells:
            for l in layers[c]:
                l.acts = orig[id(l)]
        return v

    nb = len(model.blocks)
    base = loss([])
    out = {"run": args.run, "to": args.to, "base": base, "cell": {}, "group": {}}
    print(f"{args.run}: base {base:.4f}, inputs set to {args.to} bit(s); delta loss per cell")
    print("group  " + " ".join(f"  blk{b}" for b in range(nb)) + "    all blocks  sum of blocks")
    for g in GROUPS:
        row = [loss([(b, g)]) - base for b in range(nb)]
        out["cell"][g] = row
        out["group"][g] = loss([(b, g) for b in range(nb)]) - base
        print(f"{g:5s}  " + " ".join(f"{d:+6.3f}" for d in row) + f"    {out['group'][g]:+.3f}       {sum(row):+.3f}")
    out["all"] = loss(list(layers)) - base
    print(f"all groups, all blocks: {out['all']:+.3f} (sum of groups {sum(out['group'].values()):+.3f})")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"damage_map_{args.run}_to{args.to}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
