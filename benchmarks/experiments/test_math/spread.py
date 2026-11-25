"""
spread.py -- how one changed bit spreads from a binary layer to the next (section 5b of the README).

For every pair of consecutive binary layers l -> l+1 (the classifier excluded): one random bit of layer l's output
is flipped in every sample, layer l+1 is run once on the clean and on the flipped input, and we count
    reach         sums of layer l+1 that change (the neurons that read the flipped bit; ResNet-8: the main conv of
                  the block; the shortcut acts only through the amplification)
    flip chance   share of those whose output bit flips (reach x flip chance = changed bits before the pool / OR)
    amplification bits of layer l+1's output (what layer l+2 reads: after the pool of the 1D nets, after the OR of
                  ResNet-8's blocks) that change per flipped bit
    share growth  amplification x bits of layer l / bits of layer l+1's output: how the SHARE of changed bits (the
                  measure of damage.py) grows; = reach x flip chance x pool keep x size ratio
    near          share of all layer l+1 neurons (main conv) with z - T in [-2, 1] on the clean pass: a +-2 step
                  (one +-1 input times a +-1 weight) can flip them
Binary networks, the CDT's exact integer simulation (as damage.py), validation split, test never read.

    python spread.py resnet8 --seeds 42 43 44 --out results/spread.json
    python spread.py har --nets max --seeds 42 43 44 --out results/spread.json
    python spread.py ecg --nets m --seeds 42 43 44 --out results/spread.json
    python spread.py mlp [--depth 3] --seeds 42 43 44 --out results/spread.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import damage as D                                                                            # noqa: E402
import deep_mlp                                                                            # noqa: E402
import tie_rule as R                                                                          # noqa: E402

BATCH = 250


def _pm(t):
    return torch.where(t, 1.0, -1.0).double()


def stages_resnet8(net):
    """stage(x) -> (main-conv sums, main-conv z - T, output bits after the OR) for every block"""
    def conv(x, c):
        return F.conv2d(x, torch.as_tensor(c["w"]), stride=c["stride"], padding=c["k"] // 2)

    def thr(c):
        return torch.as_tensor(c["thr"], dtype=torch.float64).view(1, -1, 1, 1)

    def block(b):
        def f(x):
            rc = conv(x, b["conv"])
            y = rc >= thr(b["conv"])
            if b["shortcut"] is not None:
                a = conv(F.max_pool2d(x, 2) if b["pool"] else x, b["shortcut"]) >= thr(b["shortcut"])
            else:
                a = x > 0
            return rc, rc - thr(b["conv"]), _pm(y | a)
        return f
    return [block(b) for b in net["blocks"]]


def first_resnet8(net, x_u8):
    t = torch.as_tensor((np.arange(net["levels"]) + 1) * 256 // (net["levels"] + 1), dtype=torch.float64)
    x = torch.as_tensor(np.ascontiguousarray(x_u8)).permute(0, 3, 1, 2).double()
    st = net["stem"]
    if st.get("int8"):
        x = x - 128
    else:
        x = (x.unsqueeze(2) >= t.view(1, 1, -1, 1, 1)).flatten(1, 2).double() * 2 - 1
    raw = F.conv2d(x, torch.as_tensor(st["w"]), stride=st["stride"], padding=st["k"] // 2)
    return _pm(raw >= torch.as_tensor(st["thr"], dtype=torch.float64).view(1, -1, 1, 1))


def stages_1d(net):
    def layer(c):
        def f(x):
            raw = F.conv1d(x, torch.as_tensor(c["w"]), stride=c["stride"])
            z = raw - torch.as_tensor(c["thr"], dtype=torch.float64).view(1, -1, 1)
            out = _pm(z >= 0)
            return raw, z, (F.max_pool1d(out, c["pool"]) if c["pool"] > 1 else out)
        return f
    return [layer(c) for c in net["layers"][1:]]


def first_1d(net, x_int8):
    x = torch.as_tensor(np.asarray(x_int8, dtype=np.float64)).transpose(1, 2)
    return stages_1d({"layers": [None] + net["layers"][:1]})[0](x)[2]


def stages_mlp(layers):
    def dense(d):
        def f(x):
            raw = x @ torch.as_tensor(d["signs"].T, dtype=torch.float64)
            z = raw - torch.as_tensor(d["threshold"], dtype=torch.float64)[None, :]
            return raw, z, _pm(z >= 0)
        return f
    return [dense(d) for d in layers[1:] if not d["last"]]


def first_mlp(layers, x_u8):
    d = layers[0]
    raw = torch.as_tensor(x_u8.astype(np.int64) @ d["signs"].T, dtype=torch.float64)
    return _pm(raw >= torch.as_tensor(d["threshold"], dtype=torch.float64)[None, :])


def spread(first, stages, x, seed=0):
    """per transition: reach, flip chance, amplification, near (sums over every sample, then the ratios)"""
    rng = np.random.default_rng(seed)
    acc = [dict(reach=0, flipped=0, amp=0, near=0, neurons=0, n=0, bits_in=0, bits_pre=0, bits_out=0) for _ in stages]
    for i in range(0, len(x), BATCH):
        h = first(x[i:i + BATCH])
        for j, f in enumerate(stages):
            n = h.shape[0]
            flat = h.reshape(n, -1).clone()
            pos = torch.as_tensor(rng.integers(0, flat.shape[1], n))
            flat[torch.arange(n), pos] *= -1
            s0, z0, o0 = f(h)
            s1, z1, o1 = f(flat.reshape(h.shape))
            a = acc[j]
            a["reach"] += int((s0 != s1).sum())
            a["flipped"] += int(((z0 >= 0) != (z1 >= 0)).sum())
            a["amp"] += int((o0 != o1).sum())
            a["near"] += int(((z0 >= -2) & (z0 <= 1)).sum())
            a["neurons"] += z0.numel()
            a["bits_in"], a["bits_pre"], a["bits_out"] = h[0].numel(), z0[0].numel(), o0[0].numel()
            a["n"] += n
            h = o0
    out = []
    for a in acc:
        reach, amp = a["reach"] / a["n"], a["amp"] / a["n"]
        flip = a["flipped"] / max(a["reach"], 1)
        out.append(dict(reach=reach, flip_chance=flip, amplification=amp, near=100 * a["near"] / a["neurons"],
                        bits_in=a["bits_in"], bits_pre=a["bits_pre"], bits_out=a["bits_out"],
                        pool_keep=amp / max(reach * flip, 1e-12), size_ratio=a["bits_in"] / a["bits_out"],
                        share_growth=amp * a["bits_in"] / a["bits_out"]))
    return out


def show(tag, rows):
    print(f"  {tag}: " + " | ".join(f"{i}->{i + 1} reach {r['reach']:.1f} flip {100 * r['flip_chance']:.1f}% "
                                    f"keep {r['pool_keep']:.2f} size {r['size_ratio']:.2f} = share x{r['share_growth']:.2f}"
                                    f" (amp {r['amplification']:.2f}, near {r['near']:.1f}%)" for i, r in enumerate(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", choices=("resnet8", "har", "ecg", "mlp"))
    ap.add_argument("--nets", nargs="+", default=[], help="har: min max max5 max10; ecg: s m l")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--depth", type=int, help="mlp: the DeepMLP_<n> runs of deep_mlp.py")
    ap.add_argument("--out", help="JSON to add these results to (e.g. results/spread.json)")
    ap.add_argument("--fixed-init", action="store_true", help="mlp: the runs of the fixed init_thresholds (section 8)")
    args = ap.parse_args()
    if args.fixed_init:
        deep_mlp.use_fixed_init()
    res = {}
    for seed in args.seeds:
        if args.bench == "resnet8":
            net, xv, _, _, _, _ = R.load_resnet8(os.path.join(D.BENCH, "conv2d", "cnn_cifar_resnet8", "runs",
                                                                f"w1_base_s{seed}"))
            items = [(f"resnet8_s{seed}", lambda x, n=net: first_resnet8(n, x), stages_resnet8(net), xv)]
        elif args.bench == "mlp":
            layers, xv = D.load_mlp(seed, args.depth)[:2]
            tag = deep_mlp.mlp_tag(args.depth)
            items = [(f"{tag}_s{seed}", lambda x, l=layers: first_mlp(l, x), stages_mlp(layers), xv)]
        else:
            if not args.nets:
                ap.error(f"{args.bench}: --nets is required")
            folder = os.path.join(D.BENCH, "conv1d", "har_ultracompact" if args.bench == "har" else "ecg_mitbih")
            items = []
            for arch in args.nets:
                sfx = "" if args.bench == "har" else "_rr"
                net, xv = R.load_1d(args.bench, os.path.join(folder, "runs", f"{arch}_binary_e100_s{seed}{sfx}"))[:2]
                items.append((f"{args.bench}_{arch}_s{seed}", lambda x, n=net: first_1d(n, x), stages_1d(net), xv))
        for key, first, stages, xv in items:
            res[key] = spread(first, stages, xv)
            show(key, res[key])
    if args.out:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        old.update(res)
        with open(args.out, "w") as f:
            json.dump(old, f, indent=1)
        print(f"-> {args.out} ({len(old)} entries)")


if __name__ == "__main__":
    main()
