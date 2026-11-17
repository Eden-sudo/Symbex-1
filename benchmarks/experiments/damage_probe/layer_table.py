"""
layer_table.py -- per-layer accuracy table of the binary ResNet-8 (README part 2): for each layer (stem, b0..b5, ds2,
ds4, fc) and part (w = weights, a = output activation, wa = both),

    drop       accuracy(twin) - accuracy(twin with only that part binary)
    recovery   accuracy(binary with only that part real) - accuracy(binary)

one shot, BatchNorm statistics recomputed after every change on 10,000 fixed training images. Then the check: every
layer real (wa) in the binary network, fine-tuned --ft-epochs, minus the same fine-tune of the unchanged network.
Validation split only (the CDT's 45,000 / 5,000); the test set is never read.

    python layer_table.py --seeds 42 43 44 --runs <training_gap>/runs/train_gap --data <repo>/benchmarks/data/cifar10

The networks are training_gap's twin_s<seed>.pth and binary_s<seed>.pth (gitignored, in the main folder). Resumable:
seeds already in --out are kept if the settings match.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "experiments", "training_gap"))
import train_gap_study as G                                # noqa: E402  (puts model.py and tools/ on the path)
import model as resnet8                                    # noqa: E402  (the same module object G patches)
from compiler_core.train.boolean import _augment           # noqa: E402

LAYERS = ["stem", "b0", "b1", "b2", "ds2", "b3", "b4", "ds4", "b5", "fc"]
PARTS = ["w", "a", "wa"]


def modules_of(net):
    """layer name -> (the module holding its weights, the BatchNorm whose output goes through sign, or None)"""
    out = {"stem": (net.stem, net.stem_bn), "fc": (net.fc, None)}
    for i, b in enumerate(net.blocks):
        out[f"b{i}"] = (b.conv, b.bn)
        if b.changes:
            out[f"ds{i}"] = (b.ds_conv, b.ds_bn)
    assert set(out) == set(LAYERS), sorted(out)
    return out


class Switch:
    """what model.py's sign() becomes: per layer, 1-bit or real weights, and sign or hardtanh after its BatchNorm.
    Weights are known by their Parameter; an activation is the output of the BatchNorm that ran last (model.py calls
    sign right on it: sign(bn(conv(x))))"""
    def __init__(self, net):
        self.weight_of, self.binary_w, self.binary_a, self.last = {}, set(), set(), None
        for name, (mod, bn) in modules_of(net).items():
            self.weight_of[id(mod.weight)] = name
            if bn is not None:
                bn.register_forward_hook(lambda m, i, o, name=name: setattr(self, "last", (name, o)))

    def set(self, binary_w, binary_a):
        self.binary_w, self.binary_a = set(binary_w), set(binary_a)

    def __call__(self, x):
        if isinstance(x, nn.Parameter):
            return G._binary_sign(x) if self.weight_of[id(x)] in self.binary_w else x
        name, out = self.last
        assert x is out, "sign() on a tensor that is not the last BatchNorm's output"
        return G._binary_sign(x) if name in self.binary_a else F.hardtanh(x)


def load_split(seed, data_dir):
    """the CDT's (train, validation) split of the CIFAR-10 training set, as training_gap.load_split (data folder given)"""
    import cdt
    cfg = {"data": dict(cdt.DEFAULTS["data"], format="cifar10", directory=data_dir), "training": {"seed": seed}}
    train, val, _ = cdt.load_data(cfg)
    return train, val


@torch.no_grad()
def recalibrate(net, x, bs=500):
    """every BatchNorm's running statistics recomputed (cumulative average) on x, no augmentation"""
    bns = [m for m in net.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.train()
    for i in range(0, len(x), bs):
        net(x[i:i + bs])
    for m, s in zip(bns, saved):
        m.momentum = s
    net.eval()


def parts_of(layer, part):
    """(weights, activations) touched by part"""
    w = [layer] if "w" in part else []
    a = [layer] if "a" in part and layer != "fc" else []
    return w, a


def finetune(net, xs, ys, epochs, lr, seed, bs=128):
    """Adam + cosine, crop + flip, labels only, BatchNorm momentum 0.3 (the benchmark's recipe, short)"""
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0.3
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    torch.manual_seed(seed)
    gen = torch.Generator(device=xs.device).manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xs), device=xs.device, generator=gen)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(net(_augment(xs[idx])), ys[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()


def measure_seed(seed, args, dev):
    train, val = load_split(seed, args.data)
    xs, ys = G._to_gpu(train, dev)
    xv, yv = G._to_gpu(val, dev)
    pick = torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:args.calib].to(dev)
    xc = xs[pick]
    row = {"n_train": len(xs), "n_validation": len(xv), "n_calibration": len(xc)}
    nets, switches, states = {}, {}, {}
    for kind in ("twin", "binary"):
        net = G.build(kind)
        net.load_state_dict(torch.load(os.path.join(args.runs, f"{kind}_s{seed}.pth"), map_location="cpu"))
        net = net.to(dev).eval()
        reference = G._sign_for(kind)                      # what training_gap evaluates
        resnet8.sign = reference
        row[f"{kind}_stored_bn"] = G._accuracy(net, xv, yv)
        with torch.no_grad():
            want = net(xv[:1000])
        sw = Switch(net)
        sw.set(LAYERS if kind == "binary" else [], LAYERS if kind == "binary" else [])
        resnet8.sign = sw
        with torch.no_grad():
            got = net(xv[:1000])
        assert torch.equal(got, want), f"{kind}: the switch does not reproduce training_gap's network"
        nets[kind], switches[kind] = net, sw
        states[kind] = {k: v.clone() for k, v in net.state_dict().items()}

    def accuracy(kind, binary_w, binary_a):
        net, sw = nets[kind], switches[kind]
        net.load_state_dict(states[kind])
        resnet8.sign = sw
        sw.set(binary_w, binary_a)
        recalibrate(net, xc)
        return G._accuracy(net, xv, yv)

    t0 = time.time()
    twin = accuracy("twin", [], [])
    binary = accuracy("binary", LAYERS, LAYERS)
    row.update(twin_recalibrated=twin, binary_recalibrated=binary, drop={}, recovery={})
    for layer in LAYERS:
        row["drop"][layer], row["recovery"][layer] = {}, {}
        for part in PARTS:
            if layer == "fc" and part != "w":
                continue
            w, a = parts_of(layer, part)
            row["drop"][layer][part] = round(twin - accuracy("twin", w, a), 2)
            keep_w = [x for x in LAYERS if x not in w]
            keep_a = [x for x in LAYERS if x not in a]
            row["recovery"][layer][part] = round(accuracy("binary", keep_w, keep_a) - binary, 2)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    row["table_seconds"] = round(time.time() - t0, 1)
    print(f"  s{seed}: twin {twin:.2f} binary {binary:.2f} (stored BN {row['twin_stored_bn']:.2f} / "
          f"{row['binary_stored_bn']:.2f}) | table {row['table_seconds']} s")
    if not args.finetune:
        return row

    def tuned(layer):
        net, sw = nets["binary"], switches["binary"]
        net.load_state_dict(states["binary"])
        resnet8.sign = sw
        w, a = parts_of(layer, "wa") if layer else ([], [])
        sw.set([x for x in LAYERS if x not in w], [x for x in LAYERS if x not in a])
        recalibrate(net, xc)
        finetune(net, xs, ys, args.ft_epochs, args.ft_lr, seed)
        recalibrate(net, xc)
        return G._accuracy(net, xv, yv)

    t0 = time.time()
    control = tuned(None)
    row["finetune"] = {"control": control, "gain": {}}
    for layer in LAYERS:
        row["finetune"]["gain"][layer] = round(tuned(layer) - control, 2)
        print(f"    {layer}: one shot {row['recovery'][layer].get('wa', row['recovery'][layer]['w']):+.2f} | "
              f"fine-tuned {row['finetune']['gain'][layer]:+.2f}", flush=True)
    row["finetune_seconds"] = round(time.time() - t0, 1)
    return row


def spearman(res):
    from scipy.stats import spearmanr
    out = {}
    for seed, row in res["seeds"].items():
        rec = [row["recovery"][x].get("wa", row["recovery"][x]["w"]) for x in LAYERS]
        drop = [row["drop"][x].get("wa", row["drop"][x]["w"]) for x in LAYERS]
        o = {"drop_vs_recovery": round(spearmanr(drop, rec).statistic, 3)}
        if "finetune" in row:
            gain = [row["finetune"]["gain"][x] for x in LAYERS]
            o["recovery_vs_finetune"] = round(spearmanr(rec, gain).statistic, 3)
            o["drop_vs_finetune"] = round(spearmanr(drop, gain).statistic, 3)
        out[seed] = o
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs", required=True, help="training_gap's runs/train_gap (twin_s<seed>.pth, binary_s<seed>.pth)")
    ap.add_argument("--data", required=True, help="the CIFAR-10 folder (benchmarks/data/cifar10)")
    ap.add_argument("--calib", type=int, default=10000, help="training images for the BatchNorm recalibration")
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--ft-lr", type=float, default=5e-4)
    ap.add_argument("--no-finetune", dest="finetune", action="store_false")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "layer_table.json"))
    args = ap.parse_args()
    args.runs, args.data = os.path.expanduser(args.runs), os.path.expanduser(args.data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    same = {k: v for k, v in vars(args).items() if k not in ("seeds", "out", "runs", "data")}
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    old = res.get("args", {})
    if old and {k: old.get(k) for k in same} != same:
        sys.exit(f"{args.out} was measured with other settings ({old} vs {same}): use another --out")
    res.update(args=same, layers=LAYERS, parts=PARTS)
    res.setdefault("seeds", {})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for seed in args.seeds:
        if str(seed) in res["seeds"]:
            print(f"  s{seed}: already in {args.out}")
            continue
        res["seeds"][str(seed)] = measure_seed(seed, args, dev)
        res["spearman"] = spearman(res)
        json.dump(res, open(args.out, "w"), indent=1)
    res["spearman"] = spearman(res)
    json.dump(res, open(args.out, "w"), indent=1)
    for seed, o in res["spearman"].items():
        print(f"  s{seed}: " + ", ".join(f"{k} {v}" for k, v in o.items()))


if __name__ == "__main__":
    main()
