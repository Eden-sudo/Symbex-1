"""
fragility.py -- is the binary network's sensitivity to ties its own, or do the INT8 / float networks share it?

tie_rule.py found that changing only the ties z == T of a trained binary network moves 9-30 % of its predictions,
mostly the wrong ones. This study applies one perturbation that does not depend on the architecture, the same to the
binary network and to its comparison network on the same inputs: noise of +-k steps on every input value (random
sign per value, clipped to the input range; the 8-bit value both networks read). It counts the predictions that move,
among the right and the wrong ones, and whether the two networks move the same samples. For the binary network it
also counts the tie moves (H(0) = 0 or 1/2) on training data: fragile there too means the network cannot separate
its own training set (capacity), not only new data.

    resnet8   training_gap's binary / twin (real weights, hardtanh) / relu ResNet-8, the same recipe and thermometer
              input (runs/train_gap/<kind>_s<seed>.pth), PyTorch; ties: the CDT runs w1_base_s<seed>
    har, ecg  the binary runs (<net>_binary_e100_s<seed>[_rr]) in the CDT simulation and the TFLite INT8 models of the
              same net and seed (models/har_int8_<net>_s<seed>.tflite, models/ecg_int8_<net>_s<seed>_rr.tflite)

Validation split and training data only (the test set is never read).

    python fragility.py resnet8 --seeds 42 43 44 --out results/fragility.json
    python fragility.py har --nets min max --seeds 42 43 44 --out results/fragility.json
    python fragility.py ecg --nets s m l --seeds 42 43 44 --out results/fragility.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import tie_rule as R                                                                          # noqa: E402

NOISE = (1, 2, 4, 8)
TRAIN_SAMPLES = 5000                                     # training samples scored (the first ones, fixed)


def noisy(x, k, lo, hi, seed=0):
    """x + k * (random sign per value), clipped to [lo, hi], same dtype"""
    s = np.random.default_rng(seed).integers(0, 2, size=x.shape) * 2 - 1
    return np.clip(x.astype(np.int16) + k * s, lo, hi).astype(x.dtype)


def moves(p0, p1, y):
    """share of predictions that move, among the right and the wrong ones (right = p0 == y)"""
    ch, right = p1 != p0, p0 == y
    return dict(moved=100 * float(ch.mean()), moved_right=100 * float(ch[right].mean()) if right.any() else None,
                moved_wrong=100 * float(ch[~right].mean()) if (~right).any() else None, accuracy=100 * float(right.mean()))


def overlap(a, b):
    """the two networks' moved sets: Jaccard and how much more often b moves where a moved than overall (lift)"""
    inter = float((a & b).sum())
    return dict(jaccard=inter / max(1.0, float((a | b).sum())),
                lift=(inter / max(1.0, float(a.sum()))) / max(1e-9, float(b.mean())))


def tie_moves(sim, net, x, y):
    """moved by H(0) = 0 or 1/2 against H(0) = 1, in the CDT simulation"""
    p = {m: sim(net, x, m, {}) for m in R.MODES}
    r = moves(p["1"], np.where(p["0"] != p["1"], p["0"], p["half"]), y)
    return r, p["1"]


def study_resnet8(seeds, noise):
    sys.path.insert(0, os.path.join(BENCH, "experiments", "training_gap"))
    import train_gap_study as G
    out = {}
    for seed in seeds:
        train, val = G.load_split(seed)
        xv, yv = G._to_gpu(val, "cpu")
        xt, yt = G._to_gpu(torch.utils.data.Subset(train, range(TRAIN_SAMPLES)), "cpu")
        u8v = (xv * 255).round().to(torch.uint8).numpy()
        u8t = (xt * 255).round().to(torch.uint8).numpy()
        yv, yt = yv.numpy(), yt.numpy()
        res, moved = {}, {}
        for kind in ("binary", "twin", "relu"):
            net = G.build(kind)
            net.load_state_dict(torch.load(os.path.join(G.RUNS, f"{kind}_s{seed}.pth"), map_location="cpu"))
            net.eval()

            @torch.no_grad()
            def pred(u8):
                G.resnet8.sign = G._sign_for(kind)            # model.py reads this global at every forward
                x = torch.as_tensor(u8).float() / 255
                return torch.cat([net(x[i:i + 1000]).argmax(1) for i in range(0, len(x), 1000)]).numpy()
            r = {}
            for split, u8, y in (("validation", u8v, yv), ("train", u8t, yt)):
                p0 = pred(u8)
                r[split] = {"accuracy": 100 * float((p0 == y).mean())}
                for k in noise:
                    p1 = pred(noisy(u8, k, 0, 255))
                    r[split][f"noise{k}"] = moves(p0, p1, y)
                    moved[(kind, split, k)] = p1 != p0
            res[kind] = r
            print(f"  resnet8 s{seed} {kind:6s} val {r['validation']['accuracy']:.2f}% train "
                  f"{r['train']['accuracy']:.2f}% | val moved " + " ".join(
                      f"+-{k}: {r['validation'][f'noise{k}']['moved']:.1f}%" for k in noise))
        res["overlap_binary_twin"] = {f"noise{k}": overlap(moved[("binary", "validation", k)],
                                                           moved[("twin", "validation", k)]) for k in noise}
        # the CDT's w1_base run of this seed: tie moves on validation (its 2000 saved images) and on training data
        run = os.path.join(BENCH, "conv2d", "cnn_cifar_resnet8", "runs", f"w1_base_s{seed}")
        net, x, y, ref, _, _ = R.load_resnet8(run)
        tv, p = tie_moves(R.sim_resnet8, net, x, y)
        assert (p == ref).all(), "H(0)=1 is not the CDT simulation"
        tt, _ = tie_moves(R.sim_resnet8, net, u8t.transpose(0, 2, 3, 1), yt)
        res["ties_cdt_w1_base"] = {"validation": tv, "train": tt}
        print(f"  resnet8 s{seed} ties (CDT w1_base): moved val {tv['moved']:.1f}% train {tt['moved']:.1f}% "
              f"(accuracy val {tv['accuracy']:.1f} train {tt['accuracy']:.1f})")
        out[f"resnet8_s{seed}"] = res
    return out


def study_1d(bench, nets, seeds, noise):
    folder = os.path.join(BENCH, "conv1d", "har_ultracompact" if bench == "har" else "ecg_mitbih")
    sys.path.insert(0, os.path.join(BENCH, "conv1d", "har_ultracompact"))
    sys.path.insert(0, folder)
    from har_tflite import predict_tflite
    if bench == "har":
        from har_data import load
    else:
        from ecg_data import load
    out = {}
    for arch in nets:
        for seed in seeds:
            sfx = "" if bench == "har" else "_rr"
            run = os.path.join(folder, "runs", f"{arch}_binary_e100_s{seed}{sfx}")
            tfl = open(os.path.join(folder, "models", f"{bench}_int8_{arch}_s{seed}{sfx}.tflite"), "rb").read()
            net, xv, yv, ref, _, _ = R.load_1d(bench, run)
            xt, yt = load("train", rr=True) if bench == "ecg" else load("train")
            xt, yt = xt[:TRAIN_SAMPLES], yt[:TRAIN_SAMPLES]
            # the TFLite input is the chip's uint8 (int8 + 128) of the ECG / accelerometer channels; ECG with RR
            # channels: the binary network reads them as int8 too (bool1d), the same array goes to both

            def tpred(x):
                return predict_tflite(tfl, (x.astype(np.int16) + 128).astype(np.uint8).reshape(len(x), -1))[0]

            def bpred(x):
                return R.sim_1d(net, x, "1", {})
            res = {}
            for name, pred in (("binary", bpred), ("int8", tpred)):
                res[name] = {}
                for split, x, y in (("validation", xv, yv), ("train", xt, yt)):
                    p0 = pred(x)
                    if name == "binary" and split == "validation":
                        assert (p0 == ref).all(), "H(0)=1 is not the CDT simulation"
                    res[name][split] = {"accuracy": 100 * float((p0 == y).mean())}
                    for k in noise:
                        p1 = pred(noisy(x, k, -128, 127))
                        res[name][split][f"noise{k}"] = moves(p0, p1, y)
                        res[name][split].setdefault("_moved", {})[k] = p1 != p0
            res["overlap_binary_int8"] = {f"noise{k}": overlap(res["binary"]["validation"]["_moved"][k],
                                                               res["int8"]["validation"]["_moved"][k]) for k in noise}
            for name in ("binary", "int8"):
                for split in ("validation", "train"):
                    res[name][split].pop("_moved")
            res["ties_binary"] = {"validation": tie_moves(R.sim_1d, net, xv, yv)[0],
                                  "train": tie_moves(R.sim_1d, net, xt, yt)[0]}
            v = res["binary"]["validation"], res["int8"]["validation"]
            print(f"  {bench} {arch} s{seed}: acc bin {v[0]['accuracy']:.1f} int8 {v[1]['accuracy']:.1f} | moved bin / "
                  "int8 " + " ".join(f"+-{k}: {v[0][f'noise{k}']['moved']:.1f}/{v[1][f'noise{k}']['moved']:.1f}"
                                     for k in noise)
                  + f" | ties val {res['ties_binary']['validation']['moved']:.1f}% train "
                    f"{res['ties_binary']['train']['moved']:.1f}%")
            out[f"{bench}_{arch}_s{seed}"] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", choices=("resnet8", "har", "ecg"))
    ap.add_argument("--nets", nargs="+", default=[], help="har: min max max5 max10; ecg: s m l")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--noise", nargs="+", type=int, default=list(NOISE), help="input noise steps k (+-k)")
    ap.add_argument("--out", help="JSON to add these results to (e.g. results/fragility.json)")
    args = ap.parse_args()
    if args.bench == "resnet8":
        res = study_resnet8(args.seeds, args.noise)
    else:
        if not args.nets:
            ap.error(f"{args.bench}: --nets is required")
        res = study_1d(args.bench, args.nets, args.seeds, args.noise)
    if args.out:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        old.update(res)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(old, f, indent=1)
        print(f"-> {args.out} ({len(old)} entries)")


if __name__ == "__main__":
    main()
