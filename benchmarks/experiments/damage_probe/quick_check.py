"""
quick_check.py -- a fast check (one seed, ~3 min), not a result: does the per-layer drop measured BEFORE binarizing
(layer_table.json, on the trained full-precision twin) name the layers that cost the most AFTER binarizing?

    1. prediction: drop of each layer (weights + activation) from layer_table.json, same seed
    2. B = the twin with every layer binary, fine-tuned --epochs (layer_table.finetune), BatchNorm recalibrated
    3. on B, no training: each layer made real again (its own fine-tuned weights, hardtanh), BatchNorm recalibrated
       -> gain = accuracy - accuracy of B; and the cosine between B's and the twin's BatchNorm output of that layer
    4. Spearman of the prediction with the gain and with (1 - cosine)
Validation split only.

    python quick_check.py --seed 42 --runs <training_gap>/runs/train_gap --data <repo>/benchmarks/data/cifar10
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import layer_table as LT
from layer_table import G, LAYERS, resnet8

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def bn_outputs(net, sw, x):
    """every layer's BatchNorm output (fc: the logits) on x, one batch"""
    outs = {}
    hooks = [bn.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o.flatten(1)))
             for k, (_, bn) in LT.modules_of(net).items() if bn is not None]
    net.eval()
    outs["fc"] = net(x).flatten(1)
    for h in hooks:
        h.remove()
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--calib", type=int, default=10000)
    ap.add_argument("--table", default=os.path.join(HERE, "results", "layer_table.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "quick_check.json"))
    args = ap.parse_args()
    args.runs, args.data = os.path.expanduser(args.runs), os.path.expanduser(args.data)
    from scipy.stats import spearmanr
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    xs, ys, xv, yv = LT.load_tensors(args.seed, args, dev)
    xc = xs[torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:args.calib].to(dev)]
    table = json.load(open(args.table))["seeds"][str(args.seed)]
    assert table["provenance"] == LT.provenance(args.seed, args, dev, (xs, ys, xv, yv)), "table from other inputs"
    predicted = {k: v.get("wa", v["w"]) for k, v in table["drop"].items()}

    net = G.build("twin")
    net.load_state_dict(torch.load(os.path.join(args.runs, f"twin_s{args.seed}.pth"), map_location="cpu"))
    net = net.to(dev).eval()
    sw = LT.Switch(net)
    resnet8.sign = sw
    sw.set([], [])
    LT.recalibrate(net, xc)
    twin_acc = G._accuracy(net, xv, yv)
    twin_out = bn_outputs(net, sw, xv[:1000])

    sw.set(LAYERS, LAYERS)
    LT.recalibrate(net, xc)
    acc_oneshot = G._accuracy(net, xv, yv)
    LT.finetune(net, xs, ys, args.epochs, args.lr, args.seed)
    LT.recalibrate(net, xc)
    b_acc = G._accuracy(net, xv, yv)
    b_out = bn_outputs(net, sw, xv[:1000])
    state = {k: v.clone() for k, v in net.state_dict().items()}

    rows = {}
    for k in LAYERS:
        rest = [x for x in LAYERS if x != k]
        net.load_state_dict(state)
        sw.set(rest, rest)
        LT.recalibrate(net, xc)
        cos = F.cosine_similarity(b_out[k], twin_out[k], dim=1).mean().item()
        rows[k] = {"predicted_drop": predicted[k], "gain_real_in_B": round(G._accuracy(net, xv, yv) - b_acc, 2),
                   "one_minus_cos": round(1 - cos, 4)}
    pred = [rows[k]["predicted_drop"] for k in LAYERS]
    out = {"seed": args.seed, "args": {k: v for k, v in vars(args).items() if k not in ("runs", "data")},
           "twin_recalibrated": twin_acc, "all_binary_one_shot": acc_oneshot, "B_finetuned": b_acc, "layers": rows,
           "spearman_pred_vs_gain": round(float(spearmanr(pred, [rows[k]["gain_real_in_B"] for k in LAYERS]).statistic), 3),
           "spearman_pred_vs_cos": round(float(spearmanr(pred, [rows[k]["one_minus_cos"] for k in LAYERS]).statistic), 3),
           "spearman_gain_vs_cos": round(float(spearmanr([rows[k]["gain_real_in_B"] for k in LAYERS],
                                                         [rows[k]["one_minus_cos"] for k in LAYERS]).statistic), 3),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"twin {twin_acc:.2f} | all binary one shot {acc_oneshot:.2f} -> B fine-tuned {b_acc:.2f}")
    print(f"{'layer':5s} {'predicted drop':>14s} {'gain real in B':>14s} {'1 - cos':>8s}")
    for k in LAYERS:
        r = rows[k]
        print(f"{k:5s} {r['predicted_drop']:14.2f} {r['gain_real_in_B']:+14.2f} {r['one_minus_cos']:8.4f}")
    print(f"Spearman pred vs gain {out['spearman_pred_vs_gain']} | pred vs 1-cos {out['spearman_pred_vs_cos']} | "
          f"gain vs 1-cos {out['spearman_gain_vs_cos']} | {out['seconds']} s")


if __name__ == "__main__":
    main()
