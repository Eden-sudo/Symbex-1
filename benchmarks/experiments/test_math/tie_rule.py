"""
tie_rule.py -- does the value of the step function at zero, H(0), matter in a binary network?

Every binary neuron computes bit = H(z - T): z = the integer sum of +-1 products, T = its integer threshold. The
library (and training: PyTorch's x >= 0 -> +1) uses H(0) = 1, so a tie z == T gives +1. This study changes only the
ties of an already trained network, in the CDT's exact integer simulation, and counts what moves:
    "1"     z == T -> +1   (the library; checked equal to the CDT simulation, prediction by prediction)
    "0"     z == T -> -1   (the same as threshold T + 1)
    "half"  z == T ->  0   (a ternary value, carried into the next layer: OR = max, pool = max, counts of halves)
Validation split only (the test set is never read). Networks: the boolean ResNet-8 (export/boolean.py simulate,
the 2000 validation images the export saved in the run), the HAR and ECG 1D nets (export/bool1d.py simulate, the
whole validation split).

    python tie_rule.py resnet8 ../../conv2d/cnn_cifar_resnet8/runs/w1_base_s4{2,3,4}
    python tie_rule.py har ../../conv1d/har_ultracompact/runs/max_binary_e100_s4{2,3,4}
    python tie_rule.py ecg ../../conv1d/ecg_mitbih/runs/m_binary_e100_s4{2,3,4}_rr
    ... --out results/tie_rule.json   (adds / replaces these runs in the JSON)
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.normpath(os.path.join(BENCH, "..", "tools")))
from compiler_core.export import bool1d, boolean                                             # noqa: E402

MODES = ("1", "0", "half")


def threshold(raw, thr, mode, ties, name):
    """+-1 bits (0 on a tie for 'half'), and the count of ties of this layer"""
    eq = raw == thr
    t = ties.setdefault(name, [0, 0])
    t[0] += int(eq.sum())
    t[1] += eq.numel()
    out = torch.where(raw >= thr, 1.0, -1.0).double()
    if mode == "0":
        out = torch.where(eq, -1.0, out)
    elif mode == "half":
        out = torch.where(eq, 0.0, out)
    return out


def sim_resnet8(net, x_u8, mode, ties):
    """boolean.simulate with the tie rule as a parameter (classification heads only, not the per-cell FOMO head)"""
    if net.get("cells"):
        raise SystemExit("[X] per-cell (FOMO) heads are not supported: their output is a grid, not a class")
    t = torch.as_tensor((np.arange(net["levels"]) + 1) * 256 // (net["levels"] + 1), dtype=torch.float64)

    def layer(x, c, name):
        raw = F.conv2d(x, torch.as_tensor(c["w"]), stride=c["stride"], padding=c["k"] // 2)
        return threshold(raw, torch.as_tensor(c["thr"], dtype=torch.float64).view(1, -1, 1, 1), mode, ties, name)

    preds = []
    for i in range(0, len(x_u8), 500):
        x = torch.as_tensor(np.ascontiguousarray(x_u8[i:i + 500])).permute(0, 3, 1, 2).double()
        if net["stem"].get("int8"):
            x = layer(x - 128, net["stem"], "stem")
        else:
            x = (x.unsqueeze(2) >= t.view(1, 1, -1, 1, 1)).flatten(1, 2).double() * 2 - 1
            x = layer(x, net["stem"], "stem")
        for j, b in enumerate(net["blocks"]):
            y = layer(x, b["conv"], f"block{j}")
            if b["shortcut"] is not None:
                a = layer(F.max_pool2d(x, 2) if b["pool"] else x, b["shortcut"], f"block{j}_shortcut")
            else:
                a = x
            x = torch.maximum(y, a)                                                      # OR of bits
        if net.get("pool"):                                                              # pooled int8 classifier
            ph = net["pool"]
            counts = F.avg_pool2d((x + 1) / 2, ph["window"]) * ph["window"] ** 2           # 'half': half counts
            dot = counts.flatten(1) @ torch.as_tensor(ph["q"].reshape(len(ph["q"]), -1), dtype=torch.float64).T
            logit = dot.numpy() * net["fc_scale"][None] + net["fc_bias"][None]
        else:
            raw = x.flatten(1) @ torch.as_tensor(net["fc_w"]).T
            logit = raw.numpy() * net["fc_scale"][None] + net["fc_bias"][None]
        preds.append(np.argmax(logit, 1))
    return np.concatenate(preds)


def sim_1d(net, x_int8, mode, ties):
    """bool1d.simulate with the tie rule as a parameter"""
    preds = []
    for i in range(0, len(x_int8), 1000):
        x = torch.as_tensor(np.asarray(x_int8[i:i + 1000], dtype=np.float64)).transpose(1, 2)
        for j, c in enumerate(net["layers"]):
            raw = F.conv1d(x, torch.as_tensor(c["w"]), stride=c["stride"])
            x = threshold(raw, torch.as_tensor(c["thr"], dtype=torch.float64).view(1, -1, 1), mode, ties, f"conv{j}")
            if c["pool"] > 1:
                x = F.max_pool1d(x, c["pool"])
        feats = x.transpose(1, 2).flatten(1)
        logit = (feats @ torch.as_tensor(net["fc_w"]).T).numpy() * net["fc_scale"][None] + net["fc_bias"][None]
        preds.append(np.argmax(logit, 1))
    return np.concatenate(preds)


def load_resnet8(run):
    """the trained boolean network of a CDT run, folded; the run's saved validation images and their simulation"""
    import tomllib
    cfg = tomllib.load(open(os.path.join(run, "config.toml"), "rb"))
    path, cls = cfg["model"]["definition"].rsplit(":", 1)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("resnet8_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = getattr(mod, cls)()
    model.load_state_dict(torch.load(os.path.join(run, "model_binarized.pth"), map_location="cpu"))
    net = boolean.fold(model.eval())
    x = np.fromfile(os.path.join(run, "validation_x.u8"), np.uint8).reshape(-1, 32, 32, 3)
    y = np.fromfile(os.path.join(run, "validation_y.u8"), np.uint8)
    ref = np.fromfile(os.path.join(run, "validation_sim.u8"), np.uint8)
    return net, x, y, ref, lambda p: 100 * float((p == y).mean()), "accuracy"


def load_1d(bench, run):
    """the trained HAR / ECG network of a run, folded; the whole validation split and its CDT simulation"""
    folder = os.path.join(BENCH, "conv1d", "har_ultracompact" if bench == "har" else "ecg_mitbih")
    sys.path.insert(0, folder)
    info = json.load(open(os.path.join(run, "train.json")))
    if bench == "har":
        from har_data import load
        from model import BinaryHAR
        model = BinaryHAR(info["arch"], info["first"])
        x, y = load("validation")
        metric, name = (lambda p: 100 * float((p == y).mean())), "accuracy"
    else:
        from ecg_data import load, scores
        from ecg_train import make_model
        rr = bool(info.get("rr", False))
        model = make_model(info["arch"], info["first"], rr)
        x, y = load("validation", rr=rr)
        metric, name = (lambda p: float(scores(y, p)["balanced"])), "balanced accuracy"
    model.load_state_dict(torch.load(os.path.join(run, "model.pth"), map_location="cpu"))
    net = bool1d.fold(model.eval())
    return net, x, y, bool1d.simulate(net, x), metric, name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", choices=("resnet8", "har", "ecg"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", help="JSON to add these runs to (e.g. results/tie_rule.json)")
    args = ap.parse_args()
    results = {}
    for run in args.runs:
        if args.bench == "resnet8":
            net, x, y, ref, metric, mname = load_resnet8(run)
            sim = sim_resnet8
        else:
            net, x, y, ref, metric, mname = load_1d(args.bench, run)
            sim = sim_1d
        preds, ties = {}, {}
        for mode in MODES:
            t = {}
            preds[mode] = sim(net, x, mode, t)
            if mode == "1":
                ties = t
        if not (preds["1"] == ref).all():
            raise SystemExit(f"[X] {run}: H(0)=1 is not the CDT simulation ({(preds['1'] != ref).sum()} differ)")
        name = os.path.basename(os.path.normpath(run))
        r = dict(bench=args.bench, n=len(y), metric=mname,
                 ties_percent=100 * sum(v[0] for v in ties.values()) / sum(v[1] for v in ties.values()),
                 ties_per_layer={k: 100 * v[0] / v[1] for k, v in ties.items()},
                 score={m: metric(preds[m]) for m in MODES},
                 changed={m: int((preds[m] != preds["1"]).sum()) for m in MODES})
        results[name] = r
        print(f"{name}  n={r['n']}  H(0)=1 == CDT simulation OK  ties {r['ties_percent']:.2f}% ("
              + " ".join(f"{k}:{v:.2f}" for k, v in r["ties_per_layer"].items()) + ")")
        for m in MODES:
            print(f"   H(0)={m:<4s} {mname} {r['score'][m]:.2f}%  predictions changed vs H(0)=1: {r['changed'][m]}")
    if args.out:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        old.update(results)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(old, f, indent=1)
        print(f"-> {args.out} ({len(old)} runs)")


if __name__ == "__main__":
    main()
