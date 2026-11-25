"""
damage.py -- where does a binary network's fragility live: does the damage grow layer by layer, or does it stay small
inside and only the output lack margin? And is it a property of the convolutions?

Binary networks only, in the CDT's exact integer simulation (the clean predictions are checked equal to it). Two
perturbations, each compared with the clean pass of the same samples:
    input   +-1 step on every 8-bit input value (random sign per value, clipped; as fragility.py)
    inside  1 % of the first binary layer's output bits flipped at random (the same share for every network, so the
            MLP and the convolutions compare directly)
Measured: the share of bits that change in each binary layer (the outputs the next layer reads: after the OR of
ResNet-8's blocks, after the pool of the 1D nets), the predictions that change (among the right and the wrong ones)
and the clean output margin (top logit - second) of the samples that change and of those that do not.

    resnet8   the CDT runs conv2d/cnn_cifar_resnet8/runs/w1_base_s<seed> (validation: the 2000 saved images)
    har, ecg  the fragility.py runs conv1d/<bench>/runs/<net>_binary_e100_s<seed>[_rr] (the whole validation split)
    mlp       mlp/mlp_fashion/runs/sizing/h128_512_s<seed>_nodistill (784-128-512-10, labels only; 2000 saved images)
              --depth n: deep_mlp.py's runs/deep_mlp/d<n>_s<seed> (784-128-512 x (n - 1)-10, same recipe)
Training data: the first 5000 samples of each run's own training split. The test set is never read.

    python damage.py resnet8 --seeds 42 43 44 --out results/damage.json
    python damage.py har --nets max --seeds 42 43 44 --out results/damage.json
    python damage.py ecg --nets m --seeds 42 43 44 --out results/damage.json
    python damage.py mlp --seeds 42 43 44 --out results/damage.json
    python damage.py mlp --depth 3 --seeds 42 43 44 --out results/damage.json
"""
import argparse
import json
import os
import sys
import tomllib

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", ".."))
TOOLS = os.path.normpath(os.path.join(BENCH, "..", "tools"))
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)
import tie_rule as R                                                                          # noqa: E402
from fragility import TRAIN_SAMPLES, noisy                                                    # noqa: E402

FLIP = 0.01                                              # share of the first binary layer's bits flipped (inside)


class Flipper:
    """flips a fixed random share of the +-1 values it is given (seeded: the same positions on every call order)"""
    def __init__(self, share, seed=0):
        self.share, self.rng = share, np.random.default_rng(seed)

    def __call__(self, x):
        if not self.share:
            return x
        m = torch.as_tensor(self.rng.random(tuple(x.shape)) < self.share)
        return torch.where(m, -x, x)


def _bits(x):
    return (x > 0).flatten(1).numpy()


def trace_resnet8(net, x_u8, flip):
    """sim_resnet8 (H(0) = 1) returning the bits of the stem and of every block output, and the logits"""
    t = torch.as_tensor((np.arange(net["levels"]) + 1) * 256 // (net["levels"] + 1), dtype=torch.float64)

    def layer(x, c):
        raw = F.conv2d(x, torch.as_tensor(c["w"]), stride=c["stride"], padding=c["k"] // 2)
        thr = torch.as_tensor(c["thr"], dtype=torch.float64).view(1, -1, 1, 1)
        return torch.where(raw >= thr, 1.0, -1.0).double()

    bits, logits = [[] for _ in range(1 + len(net["blocks"]))], []
    for i in range(0, len(x_u8), 500):
        x = torch.as_tensor(np.ascontiguousarray(x_u8[i:i + 500])).permute(0, 3, 1, 2).double()
        if net["stem"].get("int8"):
            x = layer(x - 128, net["stem"])
        else:
            x = (x.unsqueeze(2) >= t.view(1, 1, -1, 1, 1)).flatten(1, 2).double() * 2 - 1
            x = layer(x, net["stem"])
        x = flip(x)
        bits[0].append(_bits(x))
        for j, b in enumerate(net["blocks"]):
            y = layer(x, b["conv"])
            a = layer(F.max_pool2d(x, 2) if b["pool"] else x, b["shortcut"]) if b["shortcut"] is not None else x
            x = torch.maximum(y, a)
            bits[j + 1].append(_bits(x))
        if net.get("pool"):
            ph = net["pool"]
            counts = F.avg_pool2d((x + 1) / 2, ph["window"]) * ph["window"] ** 2
            dot = counts.flatten(1) @ torch.as_tensor(ph["q"].reshape(len(ph["q"]), -1), dtype=torch.float64).T
            logits.append(dot.numpy() * net["fc_scale"][None] + net["fc_bias"][None])
        else:
            raw = x.flatten(1) @ torch.as_tensor(net["fc_w"]).T
            logits.append(raw.numpy() * net["fc_scale"][None] + net["fc_bias"][None])
    return [np.concatenate(b) for b in bits], np.concatenate(logits)


def trace_1d(net, x_int8, flip):
    """sim_1d (H(0) = 1) returning the bits every conv layer passes on (after its pool), and the logits"""
    bits, logits = [[] for _ in net["layers"]], []
    for i in range(0, len(x_int8), 1000):
        x = torch.as_tensor(np.asarray(x_int8[i:i + 1000], dtype=np.float64)).transpose(1, 2)
        for j, c in enumerate(net["layers"]):
            raw = F.conv1d(x, torch.as_tensor(c["w"]), stride=c["stride"])
            x = torch.where(raw >= torch.as_tensor(c["thr"], dtype=torch.float64).view(1, -1, 1), 1.0, -1.0).double()
            if c["pool"] > 1:
                x = F.max_pool1d(x, c["pool"])
            if j == 0:
                x = flip(x)
            bits[j].append(_bits(x))
        feats = x.transpose(1, 2).flatten(1)
        logits.append((feats @ torch.as_tensor(net["fc_w"]).T).numpy() * net["fc_scale"][None] + net["fc_bias"][None])
    return [np.concatenate(b) for b in bits], np.concatenate(logits)


def trace_mlp(layers, x_u8, flip):
    """dense.simulate returning the hidden bits and the integer logits"""
    from compiler_core.export.dense import Q_ACT, Q_MULT
    act, bits = x_u8.astype(np.int64), []
    for i, d in enumerate(layers):
        raw = act @ d["signs"].T
        if not d["last"]:
            act = np.where(raw >= d["threshold"][None, :], 1, -1)
            if i == 0:
                act = flip(torch.as_tensor(act)).numpy()
            bits.append(act > 0)
        else:
            sh = Q_MULT - Q_ACT
            return bits, ((raw * d["scale"][None, :] + (1 << (sh - 1))) >> sh) + d["shift"][None, :]


def margin(logits):
    s = np.sort(np.asarray(logits, dtype=np.float64), 1)
    return s[:, -1] - s[:, -2]


def compare(trace, x, y, lo, hi):
    """clean pass vs the two perturbations: per-layer changed bits, moved predictions, margins"""
    b0, l0 = trace(x, Flipper(0))
    p0, m0 = l0.argmax(1), margin(l0)
    right = p0 == y
    out = {"accuracy": 100 * float(right.mean()), "layers": len(b0)}
    for name, xs, fl in (("input", noisy(x, 1, lo, hi), Flipper(0)), ("inside", x, Flipper(FLIP))):
        b1, l1 = trace(xs, fl)
        ch = l1.argmax(1) != p0
        mm, ms = float(np.median(m0[ch])) if ch.any() else None, float(np.median(m0[~ch]))
        out[name] = dict(
            bits_changed=[100 * float((u != v).mean()) for u, v in zip(b0, b1)],
            moved=100 * float(ch.mean()), moved_right=100 * float(ch[right].mean()),
            moved_wrong=100 * float(ch[~right].mean()) if (~right).any() else None,
            margin_moved=mm, margin_still=ms, margin_ratio=(mm / ms) if mm is not None and ms > 0 else None)
    return out, p0


def show(tag, r):
    for name in ("input", "inside"):
        d = r[name]
        mr = "-" if d["margin_ratio"] is None else f"{d['margin_ratio']:.2f}"
        print(f"  {tag} {name:6s} bits changed % by layer " + " ".join(f"{v:.2f}" for v in d["bits_changed"])
              + f" | moved {d['moved']:.1f}% (right {d['moved_right']:.1f}) | margin moved / still {mr}")


def _cdt_train_u8(cfg, n, folder):
    """the first n samples of the run's own CDT training split, as the 8-bit values the network reads"""
    import cdt
    from compiler_core.export.dense import to_uint8
    cwd = os.getcwd()
    os.chdir(folder)                                     # [data] directory may be relative to the benchmark folder
    try:
        train = cdt.load_data(cfg)[0]
    finally:
        os.chdir(cwd)
    xs = torch.stack([train[i][0] for i in range(n)])
    ys = np.array([int(train[i][1]) for i in range(n)])
    return xs, to_uint8(xs, cfg["data"]["mean"], cfg["data"]["std"]), ys, train


def study_resnet8(seeds):
    folder = os.path.join(BENCH, "conv2d", "cnn_cifar_resnet8")
    out = {}
    for seed in seeds:
        run = os.path.join(folder, "runs", f"w1_base_s{seed}")
        net, xv, yv, ref, _, _ = R.load_resnet8(run)
        cfg = tomllib.load(open(os.path.join(run, "config_used.toml"), "rb"))
        _, u8t, yt, _ = _cdt_train_u8(cfg, TRAIN_SAMPLES, folder)
        u8t = u8t.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        def trace(x, flip):
            return trace_resnet8(net, x, flip)
        res = {}
        for split, x, y in (("validation", xv, yv), ("train", u8t, yt)):
            res[split], p0 = compare(trace, x, y, 0, 255)
            if split == "validation":
                assert (p0 == ref).all(), "the clean pass is not the CDT simulation"
            show(f"resnet8 s{seed} {split[:5]}", res[split])
        out[f"resnet8_s{seed}"] = res
    return out


def study_1d(bench, nets, seeds):
    folder = os.path.join(BENCH, "conv1d", "har_ultracompact" if bench == "har" else "ecg_mitbih")
    sys.path.insert(0, folder)
    if bench == "har":
        from har_data import load
    else:
        from ecg_data import load
    out = {}
    for arch in nets:
        for seed in seeds:
            sfx = "" if bench == "har" else "_rr"
            net, xv, yv, ref, _, _ = R.load_1d(bench, os.path.join(folder, "runs", f"{arch}_binary_e100_s{seed}{sfx}"))
            xt, yt = load("train", rr=True) if bench == "ecg" else load("train")
            xt, yt = xt[:TRAIN_SAMPLES], yt[:TRAIN_SAMPLES]

            def trace(x, flip):
                return trace_1d(net, x, flip)
            res = {}
            for split, x, y in (("validation", xv, yv), ("train", xt, yt)):
                res[split], p0 = compare(trace, x, y, -128, 127)
                if split == "validation":
                    assert (p0 == ref).all(), "the clean pass is not the CDT simulation"
                show(f"{bench} {arch} s{seed} {split[:5]}", res[split])
            out[f"{bench}_{arch}_s{seed}"] = res
    return out


def study_mlp(seeds, depth=None):
    import cdt
    from compiler_core.export import dense
    folder = os.path.join(BENCH, "mlp", "mlp_fashion")
    sys.path.insert(0, folder)
    import model as M
    import deep_mlp
    out = {}
    for seed in seeds:
        run = deep_mlp.run_dir(depth, seed) if depth else os.path.join(folder, "runs", "sizing", f"h128_512_s{seed}_nodistill")
        tag = f"mlp_d{depth}" if depth else "mlp"
        cfg = tomllib.load(open(os.path.join(run, "config_used.toml"), "rb"))
        xs, u8t, yt, train = _cdt_train_u8(cfg, TRAIN_SAMPLES, folder)
        plan = json.load(open(os.path.join(run, "train.json")))["plan"]
        est = {"rows": [{"name": next(iter(plan))}]}           # load_trained reads only the first layer's name
        base = getattr(deep_mlp if depth else M, cfg["model"]["definition"].split(":")[1])()
        model = cdt.load_trained(base, train, est, cfg, run)["model"]
        layers = dense.fold(dense.analyze(model, xs[:2]), cfg["data"]["mean"], cfg["data"]["std"])
        xv = np.fromfile(os.path.join(run, "validation_x.u8"), np.uint8).reshape(-1, u8t.shape[1])
        yv = np.fromfile(os.path.join(run, "validation_y.u8"), np.uint8)
        ref = np.fromfile(os.path.join(run, "validation_sim.u8"), np.uint8)

        def trace(x, flip):
            return trace_mlp(layers, x, flip)
        res = {}
        for split, x, y in (("validation", xv, yv), ("train", u8t, yt)):
            res[split], p0 = compare(trace, x, y, 0, 255)
            if split == "validation":
                assert (p0 == ref).all(), "the clean pass is not the CDT simulation"
            show(f"{tag} s{seed} {split[:5]}", res[split])
        out[f"{tag}_s{seed}"] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", choices=("resnet8", "har", "ecg", "mlp"))
    ap.add_argument("--nets", nargs="+", default=[], help="har: min max max5 max10; ecg: s m l")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--depth", type=int, help="mlp: the DeepMLP_<n> runs of deep_mlp.py instead of the 784-128-512-10")
    ap.add_argument("--out", help="JSON to add these results to (e.g. results/damage.json)")
    args = ap.parse_args()
    if args.bench == "resnet8":
        res = study_resnet8(args.seeds)
    elif args.bench == "mlp":
        res = study_mlp(args.seeds, args.depth)
    else:
        if not args.nets:
            ap.error(f"{args.bench}: --nets is required")
        res = study_1d(args.bench, args.nets, args.seeds)
    if args.out:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        old.update(res)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(old, f, indent=1)
        print(f"-> {args.out} ({len(old)} entries)")


if __name__ == "__main__":
    main()
