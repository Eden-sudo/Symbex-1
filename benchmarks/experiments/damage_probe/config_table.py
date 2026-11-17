"""
config_table.py -- README part 3: do cheap measures rank random mixed binary / real configurations of ResNet-8 the
way a short training does? (FIT's validation protocol, Zandonati et al. ICLR 2023, at 1 bit.)

Truth: --configs random configurations (every layer of layer_table.LAYERS binary with probability 1/2, weights and
activation together, the same configurations for every seed), each trained --epochs from the twin checkpoint (Adam
--lr + cosine, crop + flip, BatchNorm momentum 0.3), BatchNorm recalibrated, validation accuracy.
Measures, all on the trained twin without training:
    drop_sum   sum of the configuration's per-layer drops (layer_table.json, part 2, same seed)
    fit_w      sum over its binary layers of Tr(EF of the weights) / n x ||w - alpha sign(w)||^2 (alpha = mean |w| per
               output channel: the BatchNorm after the layer absorbs a per-channel scale)
    fit_a      the same for the activation: Tr(EF w.r.t. the activation) / n x E||a - alpha sign(a)||^2 (alpha = mean
               |a| of the tensor: the next BatchNorm absorbs a global scale); fit = fit_w + fit_a
    acc0       accuracy of the twin with the configuration applied and the BatchNorm recalibrated (no training)
    cos_end    1 - mean cosine between the logits of that network and the twin's (validation images)
EF = empirical Fisher, per-sample gradients of the cross-entropy (batch of 1, --fisher-samples training images).
Validation split only (the CDT's); the test set is never read. Resumable (same settings and inputs only).

    python config_table.py --seeds 42 43 44 --runs <training_gap>/runs/train_gap --data <repo>/benchmarks/data/cifar10
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import layer_table as LT
from layer_table import G, LAYERS, resnet8

HERE = os.path.dirname(os.path.abspath(__file__))


def configurations(n, seed=0):
    """n distinct random subsets of LAYERS (the binary ones), each layer in with probability 1/2"""
    gen, seen, out = torch.Generator().manual_seed(seed), set(), []
    while len(out) < n:
        bits = tuple(int(b) for b in torch.randint(0, 2, (len(LAYERS),), generator=gen))
        if bits not in seen:
            seen.add(bits)
            out.append([x for x, b in zip(LAYERS, bits) if b])
    return out


class Recorder(LT.Switch):
    """the Switch, also keeping each activation's output (real mode: hardtanh) with its gradient"""
    def __init__(self, net):
        super().__init__(net)
        self.acts, self.record = {}, False

    def __call__(self, x):
        y = super().__call__(x)
        if self.record and not isinstance(x, nn.Parameter):
            y.retain_grad()
            self.acts[self.last[0]] = y
        return y


def fisher_terms(net, sw, xs, ys, n):
    """per layer: (fit_w, fit_a) on the real network (sw all real), n training images one at a time"""
    mods = LT.modules_of(net)
    sw.set([], [])
    net.eval()
    gw = {k: 0.0 for k in LAYERS}
    ga = {k: 0.0 for k in LAYERS if k != "fc"}
    na, noise_a = {}, {k: 0.0 for k in ga}
    sw.record = True
    for i in range(n):
        sw.acts = {}
        net.zero_grad(set_to_none=True)
        F.cross_entropy(net(xs[i:i + 1]), ys[i:i + 1]).backward()
        for k, (mod, _) in mods.items():
            gw[k] += mod.weight.grad.pow(2).sum().item()
        for k, a in sw.acts.items():
            ga[k] += a.grad.pow(2).sum().item()
            na[k] = a.numel()
            alpha = a.detach().abs().mean()
            noise_a[k] += (a.detach() - alpha * torch.sign(a.detach())).pow(2).sum().item()
    sw.record, sw.acts = False, {}
    net.zero_grad(set_to_none=True)
    out = {}
    for k, (mod, _) in mods.items():
        w = mod.weight.detach()
        alpha = w.abs().flatten(1).mean(1).view(-1, *[1] * (w.dim() - 1))
        noise_w = (w - alpha * torch.where(w >= 0, 1.0, -1.0)).pow(2).sum().item()
        fit_w = gw[k] / n / w.numel() * noise_w
        fit_a = ga[k] / n / na[k] * noise_a[k] / n if k in ga else 0.0
        out[k] = {"fit_w": fit_w, "fit_a": fit_a}
    return out


@torch.no_grad()
def logits(net, x, bs=1000):
    net.eval()
    return torch.cat([net(x[i:i + bs]) for i in range(0, len(x), bs)])


def measure_seed(seed, args, dev, configs, table):
    xs, ys, xv, yv = LT.load_tensors(seed, args, dev)
    pick = torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:args.calib].to(dev)
    xc = xs[pick]
    net = G.build("twin")
    net.load_state_dict(torch.load(os.path.join(args.runs, f"twin_s{seed}.pth"), map_location="cpu"))
    net = net.to(dev).eval()
    state = {k: v.clone() for k, v in net.state_dict().items()}
    sw = Recorder(net)
    resnet8.sign = sw
    row = {"provenance": LT.provenance(seed, args, dev, (xs, ys, xv, yv)), "configs": []}

    t0 = time.time()
    fisher_x = xs[torch.randperm(len(xs), generator=torch.Generator().manual_seed(1))[:args.fisher_samples].to(dev)]
    fisher_y = ys[torch.randperm(len(xs), generator=torch.Generator().manual_seed(1))[:args.fisher_samples].to(dev)]
    row["fisher"] = fisher_terms(net, sw, fisher_x, fisher_y, args.fisher_samples)
    row["fisher_seconds"] = round(time.time() - t0, 1)
    sw.acts = {}
    drops = {k: v.get("wa", v["w"]) for k, v in table["seeds"][str(seed)]["drop"].items()}

    def apply(binary):
        net.load_state_dict(state)
        sw.set(binary, binary)
        recalibrate_quiet()

    def recalibrate_quiet():
        with torch.no_grad():
            LT.recalibrate(net, xc)
        sw.acts = {}

    apply([])
    ref = logits(net, xv)
    for c, binary in enumerate(configs):
        t1 = time.time()
        apply(binary)
        out = logits(net, xv)
        rec = {"binary": binary,
               "drop_sum": round(sum(drops[k] for k in binary), 2),
               "fit_w": sum(row["fisher"][k]["fit_w"] for k in binary),
               "fit_a": sum(row["fisher"][k]["fit_a"] for k in binary),
               "acc0": round(100.0 * (out.argmax(1) == yv).float().mean().item(), 2),
               "cos_end": 1 - F.cosine_similarity(out, ref, dim=1).mean().item()}
        rec["fit"] = rec["fit_w"] + rec["fit_a"]
        rec["measure_seconds"] = round(time.time() - t1, 2)
        sw.acts = {}
        LT.finetune(net, xs, ys, args.epochs, args.lr, seed)
        sw.acts = {}
        recalibrate_quiet()
        rec["trained"] = G._accuracy(net, xv, yv)
        rec["train_seconds"] = round(time.time() - t1, 1)
        row["configs"].append(rec)
        print(f"  s{seed} config {c + 1}/{len(configs)} {'+'.join(binary) or '(none)'}: acc0 {rec['acc0']:.2f} "
              f"drop_sum {rec['drop_sum']:.1f} fit {rec['fit']:.3g} -> trained {rec['trained']:.2f} "
              f"({rec['train_seconds']} s)", flush=True)
        if args.limit and c + 1 >= args.limit:
            break
    return row


MEASURES = {"drop_sum": -1, "fit_w": -1, "fit_a": -1, "fit": -1, "acc0": 1, "cos_end": -1}   # sign: higher = better


def spearman(res):
    """per seed and measure: over all configurations, and within the stem-real / stem-binary groups"""
    from scipy.stats import spearmanr
    out = {}
    for seed, row in res["seeds"].items():
        o = {}
        for m, s in MEASURES.items():
            o[m] = {}
            for group, keep in (("all", lambda r: True), ("stem_real", lambda r: "stem" not in r["binary"]),
                                ("stem_binary", lambda r: "stem" in r["binary"])):
                rows = [r for r in row["configs"] if keep(r)]
                if len(rows) >= 4:
                    o[m][group] = round(spearmanr([s * r[m] for r in rows], [r["trained"] for r in rows]).statistic, 3)
        out[seed] = o
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs", required=True, help="training_gap's runs/train_gap (twin_s<seed>.pth)")
    ap.add_argument("--data", required=True, help="the CIFAR-10 folder (benchmarks/data/cifar10)")
    ap.add_argument("--configs", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--calib", type=int, default=10000)
    ap.add_argument("--fisher-samples", type=int, default=512)
    ap.add_argument("--table", default=os.path.join(HERE, "results", "layer_table.json"), help="part 2's drops")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: stop after this many configurations (not saved)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "config_table.json"))
    args = ap.parse_args()
    args.runs, args.data = os.path.expanduser(args.runs), os.path.expanduser(args.data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = json.load(open(args.table))
    configs = configurations(args.configs)
    same = {k: v for k, v in vars(args).items() if k not in ("seeds", "out", "runs", "data", "table", "limit")}
    res = json.load(open(args.out)) if os.path.exists(args.out) and not args.limit else {}
    old = res.get("args", {})
    if old and {k: old.get(k) for k in same} != same:
        sys.exit(f"{args.out} was measured with other settings ({old} vs {same}): use another --out")
    res.update(args=same, layers=LAYERS, configurations=configs)
    res.setdefault("seeds", {})
    for seed in args.seeds:
        if str(seed) in res["seeds"]:
            if res["seeds"][str(seed)]["provenance"] != LT.provenance(seed, args, dev):
                sys.exit(f"s{seed} in {args.out} was measured on other inputs: use another --out")
            print(f"  s{seed}: already in {args.out}, same checkpoints and split")
            continue
        row = measure_seed(seed, args, dev, configs, table)
        if args.limit:
            print(json.dumps({k: v for k, v in row.items() if k != "provenance"}, indent=1)[:3000])
            return
        res["seeds"][str(seed)] = row
        res["spearman"] = spearman(res)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    res["spearman"] = spearman(res)
    json.dump(res, open(args.out, "w"), indent=1)
    for seed, o in res["spearman"].items():
        print(f"  s{seed}: " + " | ".join(f"{m} {v}" for m, v in o.items()))


if __name__ == "__main__":
    main()
