"""
resnet20_symbex.py -- the cnn_cifar_resnet20 benchmark: binarize a PUBLISHED ResNet-20 and measure what it keeps.

    python resnet20_symbex.py --train --seeds 3                 # compress: binary, distilled from the original
    python resnet20_symbex.py --train --seeds 3 --no-distill    # compress: binary, trained on the labels only
    python resnet20_symbex.py                                   # the results (reads runs/, trains nothing)
    python resnet20_symbex.py --fidelity                        # the binary model in integers vs. floating point

Run from benchmarks/cnn_cifar_resnet20/ (main environment, .venv; a GPU helps: ~4 s per epoch with the data in VRAM).
The original is a ResNet-20 trained by someone else (model.py, 92.59% on CIFAR-10). --train runs the CDT (symbex.toml)
on it: every one of its 22 layers becomes 1 bit -- same architecture and width -- and is retrained for 60 epochs.
Each seed lands in runs/symbex_s<seed>[_nd]/, with predictions.npz (labels, original and binary predictions on the
10,000 test images). From them, the results:
    recovery    binarized accuracy / original accuracy
    identity    images where the binary model predicts the same class as the original
    retained    of the images the original gets right, the share the binary model still gets right
    gained      images the binary model gets right and the original got wrong
    per class   the same for each class: where the knowledge is kept and where it is lost
    confusions  the most frequent NEW mistakes (true class -> predicted) on images the original got right
and, with both trainings, what distillation adds (results/resnet20_distill.json, results/resnet20_no_distill.json).
--fidelity: the trained model recomputed from its weights must predict what the CDT predicted (checks the reading of
the model), then folded into integers (tools/compiler_core/export/conv.py: what the chip will run) and compared
with it for a few fixed-point precisions.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
CDT = os.path.join(TOOLS, "cdt.py")


def mean_std(v):
    v = np.asarray(v, dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def fmt(ms):
    return f"{ms[0]:.2f} +- {ms[1]:.2f}"


def analyze(pred, classes):
    """Statistics of one seed from its predictions.npz."""
    y, o, b = pred["labels"], pred["original"], pred["binarized"]
    oc = o == y
    out = dict(orig_acc=100 * oc.mean(), bin_acc=100 * (b == y).mean(), identity=100 * (b == o).mean(),
               retained=100 * (b[oc] == y[oc]).mean(), gained=int(((b == y) & ~oc).sum()))
    out["recovery"] = 100 * out["bin_acc"] / out["orig_acc"]
    per = []
    for c in range(len(classes)):
        m = y == c
        mc = m & oc
        per.append(dict(orig=100 * oc[m].mean(), bin=100 * (b[m] == c).mean(),
                        retained=100 * (b[mc] == c).mean() if mc.any() else 0.0))
    out["per_class"] = per
    lost = oc & (b != y)                                  # the original got it right, the binary model does not
    pairs = {}
    for t, p in zip(y[lost], b[lost]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    out["new_confusions"] = sorted(([t, p, n] for (t, p), n in pairs.items()), key=lambda r: -r[2])
    return out


def summarize(stats, classes):
    """Mean +- std over seeds; new confusions summed over seeds."""
    s = {k: mean_std([st[k] for st in stats]) for k in ("orig_acc", "bin_acc", "recovery", "identity", "retained", "gained")}
    s["per_class"] = [{k: mean_std([st["per_class"][c][k] for st in stats]) for k in ("orig", "bin", "retained")}
                      for c in range(len(classes))]
    conf = {}
    for st in stats:
        for t, p, n in st["new_confusions"]:
            conf[(t, p)] = conf.get((t, p), 0) + n
    s["new_confusions"] = sorted(([t, p, n / len(stats)] for (t, p), n in conf.items()), key=lambda r: -r[2])[:8]
    s["seeds"] = len(stats)
    return s


def print_summary(name, s, classes):
    print(f"\n    {name.upper()}  ({s['seeds']} seed(s), 10,000 CIFAR-10 test images)")
    print(f"    original (published) accuracy : {s['orig_acc'][0]:.2f}%")
    print(f"    binarized accuracy            : {fmt(s['bin_acc'])} %")
    print(f"    RECOVERY                      : {fmt(s['recovery'])} %  (binarized / original)")
    print(f"    IDENTITY                      : {fmt(s['identity'])} %  (same class as the original)")
    print(f"    RETAINED                      : {fmt(s['retained'])} %  (of what the original gets right)")
    print(f"    GAINED                        : {s['gained'][0]:.0f} images the original got wrong")
    print(f"\n    {'class':<12}{'original':>10}{'binarized':>18}{'retained':>18}   lost")
    print("    " + "-" * 70)
    for c, name_c in enumerate(classes):
        p = s["per_class"][c]
        lost = p["orig"][0] - p["bin"][0]
        print(f"    {name_c:<12}{p['orig'][0]:>9.1f}%{fmt(p['bin']):>17}%{fmt(p['retained']):>17}%   "
              + "#" * max(0, int(round(lost))))
    print("\n    most frequent NEW mistakes (images the original gets right), per seed:")
    for t, p, n in s["new_confusions"]:
        print(f"      {classes[t]:>10} -> {classes[p]:<10} {n:6.1f}")


def train(args, run_dir, get_pretrained):
    get_pretrained()                                   # downloads the published model the first time
    for seed in [42 + i for i in range(args.seeds)]:
        out_dir = run_dir(seed, not args.no_distill)
        if os.path.exists(os.path.join(out_dir, "predictions.npz")) and not args.redo:
            print(f"[*] seed {seed}: already trained in {out_dir} (--redo to train it again)")
            continue
        cmd = [sys.executable, CDT, "symbex.toml", "--seed", str(seed), "--output-directory", out_dir, "--to", "train",
               *(["--no-distill"] if args.no_distill else []), *(["--epochs", str(args.epochs)] if args.epochs else [])]
        print("\n[*] " + " ".join(os.path.relpath(c) if c == CDT else c for c in cmd[1:]))
        if subprocess.run(cmd).returncode != 0:
            sys.exit("[X] the CDT failed")


def results(classes, run_dir):
    found = {}
    for distill, key, title in ((True, "distill", "binary ResNet-20, distilled from the original"),
                                (False, "no_distill", "binary ResNet-20, trained on the labels only")):
        seeds = sorted(int(mt.group(1)) for d in os.listdir("runs") if (mt := re.fullmatch(r"symbex_s(\d+)(_nd)?", d))
                       and run_dir(int(mt.group(1)), distill) == os.path.join("runs", d)
                       and os.path.exists(os.path.join("runs", d, "predictions.npz")))
        if not seeds:
            print(f"[!] no trained seeds for: {title}  (python resnet20_symbex.py --train --seeds 3"
                  f"{'' if distill else ' --no-distill'})")
            continue
        stats = [analyze(np.load(os.path.join(run_dir(s, distill), "predictions.npz")), classes) for s in seeds]
        s = summarize(stats, classes)
        print_summary(title, s, classes)
        json.dump(s, open(os.path.join("results", f"resnet20_{key}.json"), "w"), indent=1)
        found[key] = s
    if len(found) == 2:
        d, n = found["distill"], found["no_distill"]
        print(f"\n    WHAT DISTILLATION ADDS  (the teacher is the published model, {d['orig_acc'][0]:.2f}%)")
        print(f"    {'':<12}{'distilled':>18}{'labels only':>18}")
        for k, label in (("bin_acc", "accuracy"), ("recovery", "recovery"), ("identity", "identity"), ("retained", "retained")):
            print(f"    {label:<12}{fmt(d[k]):>17}%{fmt(n[k]):>17}%")
        print(f"\n    With a teacher this much better than the binary student, distillation adds "
              f"{d['bin_acc'][0] - n['bin_acc'][0]:+.1f} points and makes training {n['bin_acc'][1] / max(d['bin_acc'][1], 1e-9):.0f}x "
              f"more stable across seeds.")


def fidelity(args, m):
    import torch
    sys.path.insert(0, TOOLS)
    from compiler_core.export import conv as R
    run = m.run_dir(args.seed, not args.no_distill)
    sd = torch.load(os.path.join(run, "model_binarized.pth"), map_location="cpu", weights_only=False)
    pred = np.load(os.path.join(run, "predictions.npz"))
    x, y = m.cifar_u8(train=False)
    if len(pred["labels"]) != len(y) or (pred["labels"] != y).any():
        sys.exit("[X] predictions.npz is not in the order of the CIFAR-10 test batch: the comparison would be wrong")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    pf = R.forward_float(sd, x, m.MEAN, m.STD, device=dev)
    agree = 100 * (pf == pred["binarized"]).mean()
    acc_f, acc_cdt = 100 * (pf == y).mean(), 100 * (pred["binarized"] == y).mean()
    print(f"\n    {run}: THE BINARY MODEL IN INTEGERS  (10,000 test images, {dev})")
    print(f"    1. reading the model: recomputed in float64 {acc_f:.2f}% | the CDT (float32 on its device) {acc_cdt:.2f}% "
          f"({time.time() - t:.0f} s)")
    print(f"       same class on {agree:.2f}% of the images: a binary network is chaotic under rounding -- a value a hair")
    print(f"       from a threshold flips a bit with the arithmetic's precision, and the flip spreads; the accuracy is what")
    print(f"       must agree. In integers there is one result only, and the chip reproduces it bit for bit.")
    if abs(acc_f - acc_cdt) > 0.5:
        print("    [!] the accuracies differ by more than 0.5 points: the recomputed model is not the one the CDT trained")
    print(f"    2. folded into integers:\n    {'Q_ACT':>6}{'Q_MULT':>8}{'accuracy':>11}{'same class as float64':>24}")
    for q, qm in ((12, 20), (14, 22), (16, 24), (18, 26)):
        pi = R.simulate(R.fold(sd, m.MEAN, m.STD, q_act=q, q_mult=qm), x, device=dev)
        print(f"    {q:>6}{qm:>8}{100 * (pi == y).mean():>10.2f}%{100 * (pi == pf).mean():>23.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="compress: train the binary model with the CDT")
    ap.add_argument("--seeds", type=int, default=1, help="with --train: seeds 42, 43, ... (default 1)")
    ap.add_argument("--no-distill", action="store_true", help="labels only, no teacher (--train and --fidelity)")
    ap.add_argument("--epochs", type=int, default=0, help="with --train: override [training] epochs")
    ap.add_argument("--redo", action="store_true", help="with --train: train again seeds that are already trained")
    ap.add_argument("--fidelity", action="store_true", help="the trained model in integers vs. floating point")
    ap.add_argument("--seed", type=int, default=42, help="with --fidelity: the seed to check (default 42)")
    args = ap.parse_args()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    import model as m
    os.makedirs("results", exist_ok=True)
    if args.train:
        train(args, m.run_dir, m.get_pretrained)
    elif args.fidelity:
        fidelity(args, m)
    else:
        results(m.CLASSES, m.run_dir)


if __name__ == "__main__":
    main()
