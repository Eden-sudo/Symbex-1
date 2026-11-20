"""
loss_tails.py -- README part 4b check: is the ratio (all 10 layers binary / sum of the single increases) driven by a
few confidently wrong images? Recomputes the 57 subsets it needs (none, the 10 singles, the 45 pairs, all 10) as
interactions.py does (same calibration images, same recalibration), keeps every validation image's cross-entropy and
gives the ratio and the count of positive pair interactions with three statistics: mean (= part 4b), median, and the
mean of the loss clipped at --clip per image. Validation only.

    python loss_tails.py --seeds 42 43 44 --runs <training_gap>/runs/train_gap --data <repo>/benchmarks/data/cifar10
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

import layer_table as LT
from interactions import HERE, N, subset
from layer_table import G, resnet8


@torch.no_grad()
def image_losses(net, x, y, bs=1000):
    net.eval()
    return torch.cat([torch.nn.functional.cross_entropy(net(x[i:i + bs]).float(), y[i:i + bs], reduction="none")
                      for i in range(0, len(x), bs)]).cpu().numpy()


def seed_losses(seed, args, dev, masks):
    xs, ys, xv, yv = LT.load_tensors(seed, args, dev)
    xc = xs[torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:args.calib].to(dev)]
    net = G.build("twin")
    net.load_state_dict(torch.load(os.path.join(args.runs, f"twin_s{seed}.pth"), map_location="cpu"))
    net = net.to(dev).eval()
    state = {k: v.clone() for k, v in net.state_dict().items()}
    sw = LT.Switch(net)
    resnet8.sign = sw
    out = {}
    for m in masks:
        net.load_state_dict(state)
        sw.set(subset(m), subset(m))
        LT.recalibrate(net, xc)
        out[m] = image_losses(net, xv, yv)
    return out


def summary(loss, stat):
    d = {m: stat(v) - stat(loss[0]) for m, v in loss.items()}
    singles = sum(d[1 << i] for i in range(N))
    pairs = [d[(1 << i) | (1 << j)] - d[1 << i] - d[1 << j] for i, j in itertools.combinations(range(N), 2)]
    return {"sum_of_singles": round(float(singles), 4), "all_binary": round(float(d[(1 << N) - 1]), 4),
            "ratio": round(float(d[(1 << N) - 1] / singles), 3), "positive_pairs": f"{sum(p > 0 for p in pairs)}/{len(pairs)}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--calib", type=int, default=10000)
    ap.add_argument("--clip", type=float, default=10.0, help="per-image loss cap of the clipped statistic")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "loss_tails.json"))
    args = ap.parse_args()
    args.runs, args.data = os.path.expanduser(args.runs), os.path.expanduser(args.data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = [0, *(1 << i for i in range(N)), *((1 << i) | (1 << j) for i, j in itertools.combinations(range(N), 2)),
             (1 << N) - 1]
    res = json.load(open(args.out)) if os.path.exists(args.out) else {"args": {"calib": args.calib, "clip": args.clip},
                                                                      "seeds": {}}
    for seed in args.seeds:
        loss = seed_losses(seed, args, dev, masks)
        all_l = loss[(1 << N) - 1]
        res["seeds"][str(seed)] = {
            "mean": summary(loss, np.mean), "median": summary(loss, np.median),
            "clipped": summary(loss, lambda v: np.minimum(v, args.clip).mean()),
            "all_binary_images_above_clip": round(float((all_l > args.clip).mean() * 100), 2),
            "all_binary_loss_share_of_top_5pct": round(float(np.sort(all_l)[-len(all_l) // 20:].sum() / all_l.sum()), 3)}
        print(f"s{seed}: " + " | ".join(f"{k} ratio {res['seeds'][str(seed)][k]['ratio']} pairs+ "
                                         f"{res['seeds'][str(seed)][k]['positive_pairs']}" for k in ("mean", "median", "clipped"))
              + f" | above clip {res['seeds'][str(seed)]['all_binary_images_above_clip']} %", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
