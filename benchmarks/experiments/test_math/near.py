"""
near.py -- why the sums of a binary convolution sit closer to the threshold than the MLP's (section 6 of the README).

For every binary layer that reads bits (the layers of spread.py: ResNet-8's main convs, the 1D nets' convs after the
first, the MLP's hidden layers after the first), clean pass, validation split, per neuron (an output channel over all
its positions, or a dense unit):
    n          its nonzero weights (the terms of its sum)
    mu, sigma  mean and standard deviation of z - T
    c          sigma / sqrt(n): 1 for independent +-1 terms, larger when they move together
    near       share of z - T in [-2, 1] (measured)
    pred       4 * phi(mu / sigma) / sigma: the bell-curve share (z - T steps by 2, so 2 grid points in [-2, 1])
Per network: means over the neurons of its layers; sigma, sqrt(n), c and phi over the neurons whose bit moves on the
data (a constant neuron counts in near / pred and is reported apart; the CDT gives dead neurons a threshold of
+-2^31, so the moments are taken on the raw sum, never on z - T); sigma, sqrt(n), c as geometric means.
Exact integer simulation of the CDT (as damage.py / spread.py), test never read.

    python near.py resnet8 --seeds 42 43 44 --out results/near.json
    python near.py har --nets max --seeds 42 43 44 --out results/near.json
    python near.py ecg --nets m --seeds 42 43 44 --out results/near.json
    python near.py mlp [--depth 3] --seeds 42 43 44 --out results/near.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import damage as D                                                                            # noqa: E402
import deep_mlp                                                                            # noqa: E402
import spread as S                                                                            # noqa: E402
import tie_rule as R                                                                          # noqa: E402


def _phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def near_stats(first, stages, weights, x):
    """per layer and per neuron: n, mu, sigma, near, pred"""
    acc = [None] * len(stages)
    for i in range(0, len(x), S.BATCH):
        h = first(x[i:i + S.BATCH])
        for j, f in enumerate(stages):
            raw, z, h = f(h)
            raw, z = raw.numpy(), z.numpy()
            axes = (0,) + tuple(range(2, z.ndim))                            # samples and positions
            # moments of the raw sum (small integers: exact), not of z - T (dead neurons carry T = +-2^31)
            part = np.array([raw.sum(axes), (raw * raw).sum(axes), ((z >= -2) & (z <= 1)).sum(axes),
                             (z >= 0).sum(axes)])
            cnt = z.size // z.shape[1]
            if acc[j] is None:
                acc[j] = [part, cnt]
            else:
                acc[j][0] += part
                acc[j][1] += cnt
    layers = []
    for (part, cnt), (w, thr) in zip(acc, weights):
        w = np.asarray(w)
        n = (w != 0).reshape(w.shape[0], -1).sum(1).astype(np.float64)
        mean_raw = part[0] / cnt
        sigma = np.sqrt(np.maximum(part[1] / cnt - mean_raw * mean_raw, 0))
        mu = mean_raw - np.asarray(thr, dtype=np.float64)
        live = (part[3] > 0) & (part[3] < cnt)                               # the bit moves on these samples
        sig = np.where(sigma > 0, sigma, 1.0)
        pred = np.where(sigma > 0, 4 * _phi(mu / sig) / sig, ((mu >= -2) & (mu <= 1)).astype(np.float64))
        layers.append(dict(n=n, mu=mu, sigma=sig, near=part[2] / cnt, pred=pred, live=live))
    return layers


def summary(layers):
    """network means over all neurons of its layers (sigma, sqrt n, c: geometric)"""
    def stats(l):
        v = l["live"]
        gm = lambda a: float(np.exp(np.log(a[v]).mean()))                                    # noqa: E731
        return dict(near=100 * float(l["near"].mean()), pred=100 * float(l["pred"].mean()),
                    near_moving=100 * float(l["near"][v].mean()), near_constant=100 * float(l["near"][~v].sum() / len(v)),
                    constant=100 * float(1 - v.mean()), sigma=gm(l["sigma"]), sqrt_n=gm(np.sqrt(l["n"])),
                    c=gm(l["sigma"] / np.sqrt(l["n"])), phi=float(_phi(l["mu"][v] / l["sigma"][v]).mean()),
                    abs_mu_over_sigma=float(np.median(np.abs(l["mu"][v] / l["sigma"][v]))))
    out = stats({k: np.concatenate([l[k] for l in layers]) for k in layers[0]})
    out["per_layer"] = [stats(l) for l in layers]
    return out


def show(tag, r):
    print(f"  {tag}: near {r['near']:.2f}% (pred {r['pred']:.2f}%), constant {r['constant']:.1f}% (near among moving {r['near_moving']:.2f}%), sigma {r['sigma']:.1f} = sqrt(n) "
          f"{r['sqrt_n']:.1f} x c {r['c']:.2f}, phi {r['phi']:.3f}, median |mu|/sigma {r['abs_mu_over_sigma']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", choices=("resnet8", "har", "ecg", "mlp"))
    ap.add_argument("--nets", nargs="+", default=[], help="har: min max max5 max10; ecg: s m l")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--depth", type=int, help="mlp: the DeepMLP_<n> runs of deep_mlp.py")
    ap.add_argument("--out", help="JSON to add these results to (e.g. results/near.json)")
    ap.add_argument("--fixed-init", action="store_true", help="mlp: the runs of the fixed init_thresholds (section 8)")
    args = ap.parse_args()
    if args.fixed_init:
        deep_mlp.use_fixed_init()
    res = {}
    for seed in args.seeds:
        if args.bench == "resnet8":
            net, xv = R.load_resnet8(os.path.join(D.BENCH, "conv2d", "cnn_cifar_resnet8", "runs", f"w1_base_s{seed}"))[:2]
            items = [(f"resnet8_s{seed}", lambda x, n=net: S.first_resnet8(n, x), S.stages_resnet8(net),
                      [(b["conv"]["w"], b["conv"]["thr"]) for b in net["blocks"]], xv)]
        elif args.bench == "mlp":
            layers, xv = D.load_mlp(seed, args.depth)[:2]
            tag = deep_mlp.mlp_tag(args.depth)
            items = [(f"{tag}_s{seed}", lambda x, l=layers: S.first_mlp(l, x), S.stages_mlp(layers),
                      [(d["signs"], d["threshold"]) for d in layers[1:] if not d["last"]], xv)]
        else:
            if not args.nets:
                ap.error(f"{args.bench}: --nets is required")
            folder = os.path.join(D.BENCH, "conv1d", "har_ultracompact" if args.bench == "har" else "ecg_mitbih")
            items = []
            for arch in args.nets:
                sfx = "" if args.bench == "har" else "_rr"
                net, xv = R.load_1d(args.bench, os.path.join(folder, "runs", f"{arch}_binary_e100_s{seed}{sfx}"))[:2]
                items.append((f"{args.bench}_{arch}_s{seed}", lambda x, n=net: S.first_1d(n, x), S.stages_1d(net),
                              [(c["w"], c["thr"]) for c in net["layers"][1:]], xv))
        for key, first, stages, weights, xv in items:
            res[key] = summary(near_stats(first, stages, weights, xv))
            show(key, res[key])
    if args.out:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        old.update(res)
        with open(args.out, "w") as f:
            json.dump(old, f, indent=1)
        print(f"-> {args.out} ({len(old)} entries)")


if __name__ == "__main__":
    main()
