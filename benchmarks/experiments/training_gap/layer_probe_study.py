"""
layer_probe_study.py -- where, layer by layer, does the binary ResNet-8 gain or lose what it needs to classify?
It reads the networks train_gap_study.py trained (runs/train_gap/<name>_s<seed>.pth, the same graph, validation split)
and measures, on the 5,000 validation images (the test set is never read):

    probe        linear probe (Alain & Bengio 2016): a logistic regression on each layer's output (average-pooled
                 to 4x4, standardized), trained on 10,000 fixed training images, accuracy on validation. Where it
                 rises, the layer adds class information a linear readout can use; where it falls, the layer loses it
    margin       binary convolutions only: the distance between each output's count and its neuron's threshold
                 (the BatchNorm folded, as the export does), in units of the count's noise for random bits
                 (sqrt(N) on the +-1 dot product, sqrt(N) / 2 matching bits; N = input channels x k x k, padding
                 ignored). near = share of outputs within 1 noise unit of the threshold
    or_masked    binary blocks: share of the conv's output bits the OR shortcut overrides (shortcut bit = 1: the
                 conv's decision is not used)
    null         every convolution (binary, int8 or real weights): each filter's weights shuffled within the filter
                 (same values, same count of +1s), on the same images, its output moved to the real filter's mean and
                 standard deviation (first batch), then the layer's own BatchNorm and activation (the bit, hardtanh,
                 ReLU: the shuffled neuron fires about as often as the real one). F = between-class / within-class
                 variance of the channel's mean activation per image (its firing rate in a binary network). sig = share of channels whose F beats every one of the
                 --shuffles shuffled filters (p < 1 / (shuffles + 1)): a channel that does not detects nothing a
                 random filter of the same weights would not
    flip         binary networks only: 1 % of the stem's output bits flipped at random; share of each block's output
                 bits that change, and share of predictions that change. Rising through the blocks: the network
                 amplifies noise; falling: it absorbs it

    python layer_probe_study.py --nets binary binary_w2 binary_i8 twin relu --seeds 42 43 44
    python layer_probe_study.py --nets binary_bop_g0.0001_t1e-08 --seeds 42

The network's name is train_gap_study.py's file name without _s<seed>: its first word the kind (binary, twin, relu),
_i8 the int8 stem, _w2 twice the channels; every setting is read back from its results JSON. Blocks
that change the shortcut or the activation (_sa, _sp, _sap, _sx, _og, _la2) are refused. Results: results/layer_probe/<name>_s<seed>.json, and a table of means over
the seeds. Run from anywhere; env_bench (CUDA) recommended (~1-2 min per network).
"""
import argparse
import json
import os
import re
import statistics
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_gap_study as tg                               # noqa: E402

resnet8 = tg.resnet8
OUT = os.path.join(HERE, "results", "layer_probe")
N_PROBE_TRAIN = 10000


DEAD_FIX_TAGS = {"_og": "or_grad", "_ws": "wstd", "_dl": "dead_loss"}


def restore_variant(name, seed):
    """train_gap_study.py's globals as they were when <name>_s<seed> was trained, read from its results JSON (fields
    missing in older JSONs take their defaults); stops if the rebuilt name differs from <name>"""
    meta = os.path.join(tg.OUT, f"{name}_s{seed}.json")
    if not os.path.exists(meta):
        raise SystemExit(f"[X] {meta} not found: the network's settings come from it")
    info = json.load(open(meta))
    kind = info["model"]
    tg.STEM, tg.WIDTH, tg.LAST = info.get("stem", "thermometer"), info.get("width", 1), info.get("last_stage", "x1")
    tg.BLOCKS = tuple(info.get("blocks", (2, 2, 2)))
    custom = re.search(r"_c\d", name) and info.get("channels")   # "_c<channels>", not the "_clip" recipe
    tg.CHANNELS = tuple(custom) if custom else None
    tg.SHORTCUT2, tg.SHORTCUT_OP = info.get("shortcut2", "or"), info.get("shortcut_op", "or")
    tg.PAD, tg.HEAD = info.get("pad", "zero"), info.get("head", "binary")
    parts = name.split("_")                                # older JSONs lack dead_fix: its tag in the name
    tg.DEAD_FIX = info.get("dead_fix") or next((v for k, v in DEAD_FIX_TAGS.items() if k[1:] in parts), "")
    recipe = info.get("recipe", "base")
    rebuilt = tg.tag(kind) if recipe == "base" else f"{tg.tag(kind)}_{recipe}"
    if rebuilt != name:
        raise SystemExit(f"[X] {name}: its JSON rebuilds as {rebuilt}; not probing a different network")
    return kind


def load_net(name, seed, dev):
    """the trained network <name>_s<seed>, its kind; model.py's global sign() set for it"""
    kind = restore_variant(name, seed)
    path = os.path.join(tg.RUNS, f"{name}_s{seed}.pth")
    if not os.path.exists(path):
        raise SystemExit(f"[X] {path} not found: train it with train_gap_study.py")
    net = tg.build(kind)                                   # sets resnet8.sign for this kind
    wrapped = sorted({type(b).__name__ for b in net.blocks if not hasattr(b, "conv")})
    if wrapped:                                            # they change the shortcut or the activation
        raise SystemExit(f"[X] {name}: the probes cover the benchmark's blocks only, not {', '.join(wrapped)}")
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(dev).eval(), kind


def padded_input(conv, inp):
    """(input, padding) for F.conv2d as the trained forward pass sees them: --pad minus / plus borders the binary
    convs' input with -1 / +1 (train_gap_study._padded_forward), the zero padding stays in F.conv2d"""
    if tg.PAD == "zero" or not isinstance(conv, resnet8.BinaryConv) or conv.padding == (0, 0):
        return inp, conv.padding
    p = conv.padding
    return F.pad(inp, (p[1], p[1], p[0], p[0]), value=-1.0 if tg.PAD == "minus" else 1.0), 0


def effective_weight(conv, kind):
    """the weights the forward pass uses: signs for a binary network's binary convs, the int8 values otherwise"""
    w = conv.weight.detach()
    if isinstance(conv, resnet8.Int8Conv):
        q, s = conv.int8_weights()
        return q * s.view(-1, 1, 1, 1)
    if kind == "binary":
        if tg.DEAD_FIX == "wstd":                          # the forward pass signs the standardized weights
            w = w - w.mean(dim=(1, 2, 3), keepdim=True)
        return torch.where(w >= 0, 1.0, -1.0)
    return w


def convs_of(net):
    """(name, conv, its BatchNorm) of every convolution, in forward order"""
    out = [("stem", net.stem, net.stem_bn)]
    for i, b in enumerate(net.blocks):
        out.append((f"block{i}.conv", b.conv, b.bn))
        if b.changes:
            out.append((f"block{i}.ds", b.ds_conv, b.ds_bn))
    return out


def threshold(bn):
    """BN(d) >= 0  <=>  d >= t (gamma > 0) or d <= t (gamma < 0): t per channel, on the +-1 dot product"""
    sd = torch.sqrt(bn.running_var + bn.eps)
    return bn.running_mean - bn.bias * sd / bn.weight


def activation(t, kind):
    """the layer's activation after its BatchNorm: the bit (0 / 1) in a binary network, hardtanh, ReLU"""
    if kind == "binary":
        return (t >= 0).float()
    return F.hardtanh(t) if kind == "twin" else F.relu(t)


def f_ratio(resp, y, classes=10):
    """resp (images, channels): between-class / within-class variance of each channel"""
    mean = resp.mean(0)
    between = torch.zeros_like(mean)
    within = torch.zeros_like(mean)
    for c in range(classes):
        r = resp[y == c]
        if len(r) == 0:
            continue
        between += len(r) * (r.mean(0) - mean) ** 2
        within += ((r - r.mean(0)) ** 2).sum(0)
    return (between / (classes - 1)) / (within / (len(resp) - classes)).clamp_min(1e-12)


class Taps:
    """forward hooks: each conv's input, each block's input and output; the stem's output = block 0's input"""
    def __init__(self, net):
        self.conv_in, self.block_in, self.block_out, self.handles = {}, {}, {}, []
        for name, conv, _ in convs_of(net):
            self.handles.append(conv.register_forward_hook(
                lambda m, i, o, n=name: self.conv_in.__setitem__(n, i[0])))
        for i, b in enumerate(net.blocks):
            self.handles.append(b.register_forward_hook(self._block_hook(i)))

    def _block_hook(self, k):
        def hook(m, inp, out):                             # returns None: the output stays the block's
            self.block_in[k], self.block_out[k] = inp[0], out
        return hook

    def layers(self):
        """the layer outputs, in order: the stem's, then each block's"""
        return [("stem", self.block_in[0])] + [(f"block{k}", self.block_out[k]) for k in sorted(self.block_out)]

    def remove(self):
        for h in self.handles:
            h.remove()


def pooled(t):
    return F.adaptive_avg_pool2d(t.float(), 4).flatten(1)


@torch.no_grad()
def features(net, x, bs=500):
    """the pooled outputs of every layer (and the input, as the stem sees it) for the images x"""
    taps = Taps(net)
    feats = {}
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        net(xb)
        u8 = torch.round(xb * 255) if xb.is_floating_point() else xb
        inp = (u8.float() - 128) / 128 if net.stem_kind == "int8" else resnet8.thermometer(u8, net.levels)
        for name, t in [("input", inp)] + taps.layers():
            feats.setdefault(name, []).append(pooled(t))
    taps.remove()
    return {k: torch.cat(v) for k, v in feats.items()}


def probe(ftr, ytr, fva, yva, seed, steps=500, classes=10):
    """a logistic regression, full batch, Adam; validation accuracy"""
    mu, sd = ftr.mean(0), ftr.std(0).clamp_min(1e-6)
    a, b = (ftr - mu) / sd, (fva - mu) / sd
    g = torch.Generator(device="cpu").manual_seed(seed)
    lin = torch.nn.Linear(a.shape[1], classes).to(a.device)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(lin.weight.shape, generator=g).to(a.device) * 0.01)
        lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(lin(a), ytr).backward()
            opt.step()
    with torch.no_grad():
        return 100.0 * (lin(b).argmax(1) == yva).float().mean().item()


@torch.no_grad()
def conv_stats(net, kind, x, y, shuffles, seed, bs=500):
    """per convolution: margin (binary), OR masking (binary blocks), the shuffled-filter null test"""
    taps = Taps(net)
    g = torch.Generator(device="cpu").manual_seed(seed)
    convs = convs_of(net)
    perms, weights = {}, {}
    for name, conv, _ in convs:
        w = effective_weight(conv, kind)
        weights[name] = w
        n = w[0].numel()
        perms[name] = [torch.stack([torch.randperm(n, generator=g) for _ in range(w.shape[0])]).to(w.device)
                       for _ in range(shuffles)]
    resp = {n: [] for n, _, _ in convs}
    null = {n: [[] for _ in range(shuffles)] for n, _, _ in convs}
    near = {n: [0, 0] for n, _, _ in convs}
    zsum = {n: [] for n, _, _ in convs}
    masked, match = {}, {}
    for i in range(0, len(x), bs):
        net(x[i:i + bs])
        for name, conv, bn in convs:
            w = weights[name]
            inp, pad = padded_input(conv, taps.conv_in[name])
            d = F.conv2d(inp, w, None, conv.stride, pad)
            resp[name].append(activation(bn(d), kind).mean((2, 3)))
            flat = w.flatten(1)
            for k, p in enumerate(perms[name]):
                ds = F.conv2d(inp, torch.gather(flat, 1, p).view_as(w), None, conv.stride, pad)
                if (name, k) not in match:                 # the shuffled filter moved to the real one's scale
                    match[name, k] = (ds.mean((0, 2, 3)), ds.std((0, 2, 3)).clamp_min(1e-6),
                                      d.mean((0, 2, 3)), d.std((0, 2, 3)))
                ms, ss, mr, sr = (v.view(1, -1, 1, 1) for v in match[name, k])
                null[name][k].append(activation(bn((ds - ms) / ss * sr + mr), kind).mean((2, 3)))
            if kind == "binary" and not isinstance(conv, resnet8.Int8Conv):
                z = (d - threshold(bn).view(1, -1, 1, 1)).abs() / (w[0].numel() ** 0.5)
                near[name][0] += (z < 1).sum().item()
                near[name][1] += z.numel()
                zsum[name].append(z.flatten()[::97])                    # a sample for the median
        if kind == "binary":
            for k, b in enumerate(net.blocks):
                sc = b.shortcut(taps.block_in[k])
                m = masked.setdefault(k, [0, 0])
                m[0] += (sc > 0).sum().item()
                m[1] += sc.numel()
    taps.remove()
    out = {}
    for name, conv, _ in convs:
        f_real = f_ratio(torch.cat(resp[name]), y)
        f_null = torch.stack([f_ratio(torch.cat(v), y) for v in null[name]])          # (shuffles, channels)
        row = dict(n_bits=weights[name][0].numel(), f_real=f_real.median().item(), f_null=f_null.median().item(),
                   sig=(f_real > f_null.max(0).values).float().mean().item())
        if near[name][1]:
            row.update(near=near[name][0] / near[name][1], z_median=torch.cat(zsum[name]).median().item())
        out[name] = row
    if kind == "binary":
        for k, (a, n) in masked.items():
            out[f"block{k}.conv"]["or_masked"] = a / n
    return out


@torch.no_grad()
def flip_test(net, x, p, seed, bs=500):
    """1 % of the stem's output bits flipped: share of each block's output bits that change, of predictions"""
    g = torch.Generator(device=x.device).manual_seed(seed)
    taps = Taps(net)
    changed, total, pred_changed = {}, {}, 0
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        clean = net(xb).argmax(1)
        ref = {k: v.clone() for k, v in taps.block_out.items()}
        h = net.blocks[0].register_forward_pre_hook(
            lambda m, inp: (inp[0] * torch.where(torch.rand(inp[0].shape, generator=g, device=inp[0].device) < p,
                                                 -1.0, 1.0),))
        noisy = net(xb).argmax(1)
        h.remove()
        for k, v in taps.block_out.items():
            changed[k] = changed.get(k, 0) + (v != ref[k]).sum().item()
            total[k] = total.get(k, 0) + v.numel()
        pred_changed += (noisy != clean).sum().item()
    taps.remove()
    return dict(blocks=[changed[k] / total[k] for k in sorted(changed)], predictions=pred_changed / len(x))


def analyze(name, seed, dev, shuffles):
    net, kind = load_net(name, seed, dev)
    train, val = tg.load_split(seed)
    xt, yt = tg._to_gpu(train, dev)
    xv, yv = tg._to_gpu(val, dev)
    sel = torch.randperm(len(xt), generator=torch.Generator().manual_seed(seed))[:N_PROBE_TRAIN].to(dev)
    xt, yt = xt[sel], yt[sel]
    ftr, fva = features(net, xt), features(net, xv)
    probes = {k: probe(ftr[k], yt, fva[k], yv, seed) for k in ftr}
    with torch.no_grad():
        val_acc = 100.0 * sum((net(xv[i:i + 500]).argmax(1) == yv[i:i + 500]).sum().item()
                              for i in range(0, len(xv), 500)) / len(xv)
    res = dict(net=name, kind=kind, seed=seed, val_acc=val_acc, probe=probes,
               convs=conv_stats(net, kind, xv, yv, shuffles, seed))
    if kind == "binary":
        res["flip"] = flip_test(net, xv, 0.01, seed)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}_s{seed}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"    -> {path} | validation {val_acc:.2f}%")
    return res


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{statistics.mean(vals):.3g}" + (f" ± {statistics.stdev(vals):.2g}" if len(vals) > 1 else "")


def summary(results):
    by = {}
    for r in results:
        by.setdefault(r["net"], []).append(r)
    for name, rs in by.items():
        print(f"\n  {name}  ({len(rs)} seed(s); validation {mean_sd([r['val_acc'] for r in rs])} %)")
        print("    linear probe, validation %: " + " | ".join(
            f"{k} {mean_sd([r['probe'][k] for r in rs])}" for k in rs[0]["probe"]))
        print(f"    {'conv':14s}{'N':>6}{'F real':>14}{'F null':>14}{'sig':>12}{'near':>12}{'z med':>12}{'OR mask':>12}")
        for c in rs[0]["convs"]:
            col = lambda key: mean_sd([r["convs"][c].get(key) for r in rs])                       # noqa: E731
            print(f"    {c:14s}{rs[0]['convs'][c]['n_bits']:>6}{col('f_real'):>14}{col('f_null'):>14}{col('sig'):>12}"
                  f"{col('near'):>12}{col('z_median'):>12}{col('or_masked'):>12}")
        if "flip" in rs[0]:
            nb = len(rs[0]["flip"]["blocks"])
            print("    1 % of the stem bits flipped, share of bits changed: " + " | ".join(
                f"block{k} {mean_sd([r['flip']['blocks'][k] for r in rs])}" for k in range(nb))
                + f" | predictions {mean_sd([r['flip']['predictions'] for r in rs])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--nets", nargs="+", required=True, help="train_gap_study.py names without _s<seed>")
    ap.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42])
    ap.add_argument("--shuffles", type=int, default=19, help="shuffled filters per channel (p < 1 / (shuffles + 1))")
    ap.add_argument("--summary", action="store_true", help="no measurement: the table from results/layer_probe")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for name in args.nets:
        for seed in args.seeds:
            path = os.path.join(OUT, f"{name}_s{seed}.json")
            if args.summary:
                if os.path.exists(path):
                    results.append(json.load(open(path)))
                continue
            print(f"\n[*] {name} seed {seed}")
            results.append(analyze(name, seed, dev, args.shuffles))
    summary(results)


if __name__ == "__main__":
    main()
