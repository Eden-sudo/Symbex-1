"""
interactions.py -- README part 4: the damage of binarizing layers does not add up (tiny_llm: all units at once 2.2-2.5x
the sum of single units). Do the interactions between ResNet-8 layers, measured without training, rank part 3's
trained configurations better than the plain sum of single-layer drops?

One shot table: the twin with every one of the 2^10 subsets of layer_table.LAYERS binary (weights and activation
together, as part 3), BatchNorm recalibrated on the same 10,000 training images, validation accuracy. From it, per seed:
    additive   sum of the single drops d_i of the configuration's binary layers (part 3's drop_sum, recomputed here)
    pairs      additive + the pair interactions I_ij = d_ij - d_i - d_j of every pair inside the configuration
    shapley    sum of the exact Shapley values phi_i of its binary layers (the game: drop of a subset; phi_i = the
               average extra drop of layer i over every order of adding the layers)
    popcount   mean one-shot drop of all the subsets with the same stem state and the same number of other binary layers
               (the layers other than the stem taken as interchangeable)
    acc0       the configuration's own one-shot accuracy (part 3's acc0, recomputed: the information all the above
               approximate)
Truth: part 3's trained accuracies (results/config_table.json). Statistic: config_analysis.partial (Spearman with the
number of binary layers held fixed) and Spearman over all 40. Also the interchangeability check of popcount: inside
each (stem, count) group of the 1,024 subsets, the spread of the one-shot accuracy vs the spread of the same subset
between seeds. Validation split only; the test set is never read.

    python interactions.py --seeds 42 43 44 --runs <training_gap>/runs/train_gap --data <repo>/benchmarks/data/cifar10
    python interactions.py --analyze                    # only the analysis, from results/interactions.json
    python interactions.py --metric loss --out results/interactions_loss.json ...   # part 4b: the game in loss
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

import layer_table as LT
from config_analysis import partial
from layer_table import G, LAYERS, resnet8

HERE = os.path.dirname(os.path.abspath(__file__))
N = len(LAYERS)
STEM = LAYERS.index("stem")


def subset(mask):
    return [LAYERS[i] for i in range(N) if mask >> i & 1]


def mask_of(binary):
    return sum(1 << LAYERS.index(k) for k in binary)


@torch.no_grad()
def accuracy_and_loss(net, x, y, bs=1000):
    """validation accuracy (%, as G._accuracy) and mean cross-entropy (natural log) in one pass"""
    net.eval()
    hits, loss = 0, 0.0
    for i in range(0, len(x), bs):
        out = net(x[i:i + bs])
        hits += (out.argmax(1) == y[i:i + bs]).sum().item()
        loss += torch.nn.functional.cross_entropy(out.float(), y[i:i + bs], reduction="sum").item()
    return 100.0 * hits / len(x), loss / len(x)


def one_shot_table(seed, args, dev):
    """accuracy and loss of the twin with every subset binary, index = bit mask over LAYERS"""
    xs, ys, xv, yv = LT.load_tensors(seed, args, dev)
    prov = LT.provenance(seed, args, dev, (xs, ys, xv, yv))
    xc = xs[torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:args.calib].to(dev)]
    net = G.build("twin")
    net.load_state_dict(torch.load(os.path.join(args.runs, f"twin_s{seed}.pth"), map_location="cpu"))
    net = net.to(dev).eval()
    state = {k: v.clone() for k, v in net.state_dict().items()}
    sw = LT.Switch(net)
    resnet8.sign = sw
    acc, loss = [], []
    t0 = time.time()
    for m in range(2 ** N):
        net.load_state_dict(state)
        sw.set(subset(m), subset(m))
        LT.recalibrate(net, xc)
        a, l = accuracy_and_loss(net, xv, yv)
        acc.append(a)
        loss.append(l)
        if m % 128 == 127:
            print(f"  s{seed}: {m + 1}/{2 ** N} subsets ({time.time() - t0:.0f} s)", flush=True)
    return {"provenance": prov, "accuracy": acc, "loss": loss, "seconds": round(time.time() - t0, 1)}


def predictors(acc, metric="accuracy"):
    """per configuration mask: the predicted DROP of each model (higher = worse); with the loss, drop = increase"""
    acc = np.asarray(acc, dtype=np.float64)
    drop = acc[0] - acc if metric == "accuracy" else acc - acc[0]   # the game: v(S) = damage of subset S
    d = np.array([drop[1 << i] for i in range(N)])
    inter = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        inter[i, j] = drop[(1 << i) | (1 << j)] - d[i] - d[j]
    phi = np.zeros(N)
    for i in range(N):                                           # exact Shapley: weight |S|! (N - |S| - 1)! / N!
        for m in range(2 ** N):
            if m >> i & 1:
                continue
            s = bin(m).count("1")
            phi[i] += math.factorial(s) * math.factorial(N - s - 1) / math.factorial(N) * (drop[m | 1 << i] - drop[m])
    group = {}
    for m in range(2 ** N):
        key = (m >> STEM & 1, bin(m & ~(1 << STEM)).count("1"))
        group.setdefault(key, []).append(drop[m])

    def predict(m):
        idx = [i for i in range(N) if m >> i & 1]
        return {"additive": d[idx].sum(),
                "pairs": d[idx].sum() + sum(inter[i, j] for i, j in itertools.combinations(idx, 2)),
                "shapley": phi[idx].sum(),
                "popcount": float(np.mean(group[(m >> STEM & 1, bin(m & ~(1 << STEM)).count("1"))])),
                "acc0": drop[m]}
    return predict, d, inter, phi, group


def analyze(res, configs, metric="accuracy"):
    seeds = sorted(res["seeds"])
    out = {"seeds": {}}
    for s in seeds:
        rows = configs["seeds"][s]["configs"]
        predict, d, inter, phi, group = predictors(res["seeds"][s][metric], metric)
        trained = [r["trained"] for r in rows]
        n = [len(r["binary"]) for r in rows]
        preds = [predict(mask_of(r["binary"])) for r in rows]
        o = {}
        for k in preds[0]:
            score = [-p[k] for p in preds]                       # higher = better, as the trained accuracy
            o[k] = {"all": round(float(spearmanr(score, trained).statistic), 3), "partial_count": partial(score, trained, n)}
        acc0_saved = [r["acc0"] for r in rows]
        recomputed = [res["seeds"][s]["accuracy"][0] - predict(mask_of(r["binary"]))["acc0"] for r in rows] \
            if metric == "accuracy" else [res["seeds"][s]["accuracy"][mask_of(r["binary"])] for r in rows]
        o["acc0_vs_part3_max_abs_diff"] = round(float(np.max(np.abs(np.subtract(acc0_saved, recomputed)))), 2)
        full = (1 << N) - 1
        nd = 2 if metric == "accuracy" else 4
        o["sum_single_vs_all"] = {"sum_of_singles": round(float(d.sum()), nd), "all_binary": round(predict(full)["acc0"], nd),
                                  "ratio": round(float(predict(full)["acc0"] / d.sum()), 3)}
        pairs = [inter[i, j] for i, j in itertools.combinations(range(N), 2)]
        o["positive_pairs"] = f"{sum(v > 0 for v in pairs)}/{len(pairs)}"
        o["phi"] = dict(zip(LAYERS, np.round(phi, nd).tolist()))
        o["single"] = dict(zip(LAYERS, np.round(d, nd).tolist()))
        top = sorted(((inter[i, j], LAYERS[i], LAYERS[j]) for i, j in itertools.combinations(range(N), 2)),
                     key=lambda t: -abs(t[0]))[:5]
        o["largest_pair_interactions"] = [[a, b, round(float(v), nd)] for v, a, b in top]
        o["group_sd_mean"] = round(float(np.mean([np.std(v, ddof=1) for v in group.values() if len(v) > 1])), nd)
        out["seeds"][s] = o
    acc = np.array([res["seeds"][s][metric] for s in seeds])
    out["seed_sd_mean"] = round(float(acc.std(0, ddof=1).mean()), 4) if len(seeds) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs", help="training_gap's runs/train_gap (twin_s<seed>.pth)")
    ap.add_argument("--data", help="the CIFAR-10 folder (benchmarks/data/cifar10)")
    ap.add_argument("--calib", type=int, default=10000, help="training images for the BatchNorm recalibration")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "interactions.json"))
    ap.add_argument("--configs", default=os.path.join(HERE, "results", "config_table.json"))
    ap.add_argument("--analyze", action="store_true", help="skip the measuring: only the analysis of --out")
    ap.add_argument("--metric", choices=("accuracy", "loss"), default="accuracy",
                    help="the game the models are built on: validation accuracy (part 4) or cross-entropy (part 4b)")
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {"args": {"calib": args.calib}, "layers": LAYERS,
                                                                      "seeds": {}}
    configs = json.load(open(args.configs))
    if not args.analyze:
        if not (args.runs and args.data):
            sys.exit("--runs and --data are needed to measure (or --analyze)")
        if res["args"].get("calib") != args.calib:
            sys.exit(f"{args.out} was measured with --calib {res['args'].get('calib')}: use another --out")
        args.runs, args.data = os.path.expanduser(args.runs), os.path.expanduser(args.data)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in args.seeds:
            if str(seed) in res["seeds"] and args.metric in res["seeds"][str(seed)]:
                print(f"  s{seed}: already in {args.out}")
                continue
            row = one_shot_table(seed, args, dev)
            if row["provenance"] != configs["seeds"][str(seed)]["provenance"]:
                sys.exit(f"s{seed}: other checkpoints or split than {args.configs}")
            res["seeds"][str(seed)] = row
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(res, open(args.out, "w"))
    done = {k: v for k, v in res["seeds"].items() if k in configs["seeds"] and args.metric in v}
    if not done:
        return
    out = analyze({"seeds": done}, configs, args.metric)
    res["analysis"] = out
    json.dump(res, open(args.out, "w"))
    seeds = sorted(done)
    print(f"{'model':9s} {'all 40':>20s} {'same count':>20s}")
    for k in ("additive", "pairs", "shapley", "popcount", "acc0"):
        print(f"{k:9s} " + " ".join(f"{' / '.join('%.2f' % out['seeds'][s][k][g] for s in seeds):>20s}"
                                    for g in ("all", "partial_count")))
    for s in seeds:
        o = out["seeds"][s]
        print(f"s{s}: sum of singles {o['sum_single_vs_all']['sum_of_singles']} vs all binary "
              f"{o['sum_single_vs_all']['all_binary']} (ratio {o['sum_single_vs_all']['ratio']}, positive pairs "
              f"{o['positive_pairs']}) | acc0 check {o['acc0_vs_part3_max_abs_diff']} | group sd "
              f"{o['group_sd_mean']} | largest pairs {o['largest_pair_interactions']}")
    print(f"seed sd (same subset, mean) {out['seed_sd_mean']}")


if __name__ == "__main__":
    main()
