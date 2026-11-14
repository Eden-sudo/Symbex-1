"""
har_shift.py -- har_ultracompact benchmark: do the fixed BatchNorm + sign thresholds break on new people? A diagnostic
for README section 9, with NO training: it reads the saved networks of har_train.py (Adam, 100 epochs) and the FP32
networks TFLite quantizes (models/har_fp32_<arch>_s<seed>.pth, FloatHAR of har_lowbit.py).

    python har_shift.py                     # 4 nets x (binary, int8 first layer, FP32) x seeds 42, 43, 44
    python har_shift.py --arch max --seeds 42

Run from benchmarks/conv1d/har_ultracompact/ (environment env_bench). For every network:
    stored      accuracy on train / validation / test with the BatchNorm statistics of training (as deployed)
    AdaBN       the BatchNorm running statistics recomputed on the target windows, labels unused (Li et al. 2016):
                  all   one recalibration on all the windows of the split
                  subj  one recalibration per subject (each person calibrates their own device)
                the validation subjects are the measurement; the test subjects are a DIAGNOSTIC only (recalibrating
                on them is not a deployable result and does not change any choice: nothing here is exported)
    bits        (binary networks) per conv layer, per channel, the share of +1 after sign on each split; drift =
                mean over channels of |share(split) - share(train)|; saturated = channels with share < 5 % or > 95 %;
                per pool, the share of +1 after the OR and the share of windows of 4 with two or more +1 (ties)
    input       per split, the mean and spread of each axis (gravity and phone orientation show here)
    orientation each subject's phone "up" or "down" in the pocket (sign of the mean y axis over all their windows:
                about +0.9 g or -1 g while walking) and the stored accuracy of each group, per split
Results: results/symbex/har_shift.json. Accuracy in %.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from har_data import OUT as DATA                                                             # noqa: E402
from model import ARCHS, TABLE1, BinaryHAR, int8_weights, sign                               # noqa: E402

SEEDS = (42, 43, 44)
SPLITS = {"train": "train_subjects", "validation": "val_subjects", "test": "test_subjects"}


def load_splits(dev):
    """{split: (x, y, subject)} on the device"""
    d = np.load(DATA)
    out = {}
    for name, key in SPLITS.items():
        m = np.isin(d["subject"], d[key])
        out[name] = tuple(torch.as_tensor(a[m], device=dev) for a in (d["x"], d["y"], d["subject"]))
    return out


def binary_net(arch, first, seed, dev):
    model = BinaryHAR(arch, first).to(dev)
    model.load_state_dict(torch.load(os.path.join(HERE, "runs", f"{arch}_{first}_e100_s{seed}", "model.pth"),
                                     map_location=dev))
    return model.eval()


def float_net(arch, seed, dev):
    from har_lowbit import FloatHAR
    model = FloatHAR(arch).to(dev)
    model.load_state_dict(torch.load(os.path.join(HERE, "models", f"har_fp32_{arch}_s{seed}.pth"), map_location=dev))
    return model.eval()


def run(model, x):
    """logits; the float network takes x / 127 channel-major, the binary one the int8 windows"""
    if isinstance(model, BinaryHAR):
        return model(x)
    return model(x.float().transpose(1, 2) / 127)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    return 100 * (run(model, x).argmax(1) == y).float().mean().item()


@torch.no_grad()
def adabn(model, x):
    """BatchNorm running statistics := the statistics of x (one batch with every window, so every layer is normalized
    with the exact statistics of the target windows as it collects them); the weights and BN scales are unchanged"""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = 1.0
    model.train()
    run(model, x)
    model.eval()


def orientation(data):
    """{subject: "up" | "down"}: the sign of the mean y axis over all the subject's windows"""
    out = {}
    for x, _, subj in data.values():
        for k in torch.unique(subj):
            out[int(k)] = "up" if x[subj == k][..., 1].float().mean().item() > 0 else "down"
    return out


def accuracies(make, data, orient):
    """stored, AdaBN on all the windows of the split, AdaBN per subject; validation and test; stored per orientation"""
    model = make()
    res = {"stored": {s: accuracy(model, data[s][0], data[s][1]) for s in SPLITS}, "orientation": {}}
    for s in SPLITS:
        x, y, subj = data[s]
        for o in ("up", "down"):
            m = torch.as_tensor([orient[int(k)] == o for k in subj.tolist()], device=x.device)
            res["orientation"][f"{s}_{o}"] = accuracy(model, x[m], y[m])
    for split in ("validation", "test"):
        x, y, subj = data[split]
        model = make()
        adabn(model, x)
        res.setdefault("adabn_all", {})[split] = accuracy(model, x, y)
        hits = 0
        for s in torch.unique(subj):
            m = subj == s
            model = make()
            adabn(model, x[m])
            hits += (run(model, x[m]).argmax(1) == y[m]).sum().item()
        res.setdefault("adabn_subj", {})[split] = 100 * hits / len(x)
    return res


@torch.no_grad()
def bit_stats(model, x):
    """per conv layer, the share of +1 per channel after sign; per pool, the share of +1 after the OR and of windows
    with two or more +1 (BinaryHAR.forward step by step, pool after sign as the chip runs it)"""
    convs, pools = [], []
    h = x.float().transpose(1, 2)
    for kind, i in model.plan:
        if kind == "pool":
            n = h.shape[-1] // i * i
            win = (h[..., :n] > 0).reshape(*h.shape[:-1], -1, i).sum(-1)
            pools.append({"ones_after": (win > 0).float().mean().item(), "ties": (win >= 2).float().mean().item()})
            h = F.max_pool1d(h, i)
            continue
        w = model.convs[i].weight
        if i == 0:
            w = int8_weights(w) if model.first == "int8" else sign(w)
            h = h / 127
        else:
            w = sign(w)
        h = sign(model.bns[i](F.conv1d(h, w)))
        convs.append((h > 0).float().mean(dim=(0, 2)).cpu().numpy())
    return convs, pools


def bits(model, data):
    stats = {s: bit_stats(model, data[s][0]) for s in SPLITS}
    layers = []
    for j, ref in enumerate(stats["train"][0]):
        row = {"channels": len(ref)}
        for s in SPLITS:
            p = stats[s][0][j]
            row[s] = {"ones": float(p.mean()), "saturated": float(((p < 0.05) | (p > 0.95)).mean()),
                      "drift": float(np.abs(p - ref).mean())}
        layers.append(row)
    return {"conv": layers, "pool": {s: stats[s][1] for s in SPLITS}}


def input_stats(data):
    """per split, mean and std of each axis in g (int8 127 = 2 g), and the spread of the per-subject means"""
    out = {}
    for s in SPLITS:
        x, _, subj = data[s]
        g = x.float() / 127 * 2
        per = torch.stack([g[subj == k].mean(dim=(0, 1)) for k in torch.unique(subj)])
        out[s] = {"mean_g": g.mean(dim=(0, 1)).tolist(), "std_g": g.std(dim=(0, 1)).tolist(),
                  "subject_mean_spread_g": per.std(0).tolist()}
    return out


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), help="one architecture (default: the four of Table 1)")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "symbex", "har_shift.json"))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    data = load_splits(dev)
    seeds = [int(s) for s in args.seeds.split(",")]
    archs = [args.arch] if args.arch else list(TABLE1)
    orient = orientation(data)
    result = {"input": input_stats(data), "orientation": {str(k): v for k, v in sorted(orient.items())}, "nets": {}}
    for arch in archs:
        for kind in ("binary", "int8", "fp32"):
            runs = []
            for seed in seeds:
                make = ((lambda: float_net(arch, seed, dev)) if kind == "fp32"
                        else (lambda: binary_net(arch, kind, seed, dev)))
                r = {"seed": seed, **accuracies(make, data, orient)}
                if kind != "fp32":
                    r["bits"] = bits(make(), data)
                runs.append(r)
            result["nets"][f"{arch}_{kind}"] = runs
            line = f"[*] {arch:<6} {kind:<6}"
            for key, split in (("stored", "validation"), ("adabn_all", "validation"), ("adabn_subj", "validation"),
                               ("stored", "test"), ("adabn_all", "test"), ("adabn_subj", "test")):
                m, sd = mean_sd([r[key][split] for r in runs])
                line += f" | {key[:10]} {split[:4]} {m:5.1f} ± {sd:3.1f}"
            print(line)
            o = {k: mean_sd([r["orientation"][k] for r in runs])[0] for k in runs[0]["orientation"]}
            print("      orientation: " + "  ".join(f"{k} {v:.1f}" for k, v in o.items()))
            if kind != "fp32":
                for j in range(len(runs[0]["bits"]["conv"])):
                    d = {s: mean_sd([r["bits"]["conv"][j][s]["drift"] for r in runs])[0] for s in ("validation", "test")}
                    o = {s: mean_sd([r["bits"]["conv"][j][s]["ones"] for r in runs])[0] for s in SPLITS}
                    sat = mean_sd([r["bits"]["conv"][j]["train"]["saturated"] for r in runs])[0]
                    print(f"      conv{j}: +1 train/val/test {o['train']:.2f}/{o['validation']:.2f}/{o['test']:.2f}"
                          f"  drift val {d['validation']:.3f} test {d['test']:.3f}  saturated (train) {sat:.2f}")
                for j in range(len(runs[0]["bits"]["pool"]["train"])):
                    p = runs[0]["bits"]["pool"]["train"][j]
                    print(f"      pool{j} (seed {seeds[0]}, train): +1 after OR {p['ones_after']:.2f}, ties {p['ties']:.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
