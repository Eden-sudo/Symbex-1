"""
pair_bits.py -- quick experiment: does a network whose neurons hold PAIRS of bits learn more than a flat binary one?

Everything is binary in both networks: inputs, weights and activations are bits; every product is between bits
(XNOR / AND + count on a chip). The difference is how many bits each element has:

    flat   every weight is ONE bit b (+-1), every activation ONE bit s = sign(x - t)
    pair   every weight is a PAIR of independent bits (b1, b2), every activation a pair (s1, s2) with its own
           thresholds; the network LEARNS how to combine them:  w = a1*b1 + a2*b2,  x = c1*s1 + c2*s2
           (a1, a2 per output neuron, c1, c2 and the thresholds per input feature). Not a 2-bit integer, where
           the combination is fixed (2*b1 + b0): here the two bits are two learned directions.
    One layer then costs 2 x 2 = 4 binary dot products instead of 1 (with pairs on both sides).

The fair comparison is at the SAME memory: a pair network of width H stores the same weight bits as a flat
network of width ~2H. The experiment trains, for each configuration and seed, the same MLP shape
(input -> H -> H/2 -> classes) with every layer binary, the input included (learned thresholds per pixel),
and reports accuracy, weight memory and binary operations per inference.

    python pair_bits.py                                  # Fashion-MNIST, 3 seeds, 20 epochs
    python pair_bits.py --dataset digits --epochs 100    # scikit-learn Digits (8x8)
    python pair_bits.py --width 128 --seeds 5
    python pair_bits.py --arch resnet8                   # CIFAR-10, a small binary ResNet (see below)
    python pair_bits.py --pair-on weights                # pairs only in the weights (or: activations)

--arch resnet8 (CIFAR-10): stem + 3 stages of one block with two 3x3 convolutions + classifier = 8 layers,
channels c, 2c, 4c (c = --width, default 16). The input is binary too: every RGB pixel becomes its 24 bit planes
(BPR, 8 per channel), the same for every configuration, so the comparison is only about the network. Each
convolution has its own residual shortcut (Bi-Real style: the additions are integer sums on a chip; every
product is between bits); downsampling shortcuts are a 2x2 average plus zero channels (no weights).
At the same memory a flat network has ~1.41x the channels of a pair one (convolution weights grow with c^2).
Light augmentation (random flips and shifts, on the GPU), the same for every configuration.

Run from benchmarks/experiments/ (main environment, .venv; a GPU is used if there is one). No distillation,
no tricks: plain training with the straight-through estimator, so the only difference is flat vs pair.
Writes results/pair_bits_<dataset>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------------------------------
# binary building blocks
# ---------------------------------------------------------------------------------------------------
class SignSTE(torch.autograd.Function):
    """sign with 0 -> +1 forward; straight-through gradient, clipped to |x| <= 1."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


sign = SignSTE.apply


ACT_SCALE = "feature"          # set by --act-scale: how the k bits of an activation are combined
DENSITY = False                # set while measuring: every BinaryActivation records the share of +1 per plane


class BinaryActivation(nn.Module):
    """k bits per feature: s_j = sign(x - t_j), combined as sum_j c_j * s_j.
    Thresholds are learned per feature and NESTED (t_1 < t_2 < ...: t_j = t_1 + sum of positive steps), so a feature
    above t_2 is also above t_1 and the upper planes are sparse. Scales c_j, by ACT_SCALE:
        feature  learned per feature and plane (needs a multiplication per input on a chip)
        layer    learned per plane, shared by the layer (one integer combination per neuron on a chip)
        equal    fixed, all equal: the layer is a SUM OF COUNTS, no multiplication (thermometer code)
        pow2     fixed 2^(k-1), ..., 2, 1: counts combined with shifts"""

    def __init__(self, features, k):
        super().__init__()
        self.k = k
        self.t0 = nn.Parameter(torch.full((1, features), -0.5 if k > 1 else 0.0))
        self.steps = nn.Parameter(torch.full((max(k - 1, 1), features), 0.5413))   # softplus(0.5413) = 1.0
        if ACT_SCALE == "feature":
            self.c = nn.Parameter(torch.full((k, features), 1.0 / k))
        elif ACT_SCALE == "layer":
            self.c = nn.Parameter(torch.full((k, 1), 1.0 / k))
        else:
            vals = [1.0] * k if ACT_SCALE == "equal" else [2.0 ** (k - 1 - j) for j in range(k)]
            self.register_buffer("c", torch.tensor(vals)[:, None] / k)
        self.density = []

    def thresholds(self):
        if self.k == 1:
            return self.t0
        return torch.cat([self.t0, self.t0 + torch.cumsum(F.softplus(self.steps), 0)], 0)

    def _scale(self, x, j):
        return self.c[j]

    def forward(self, x):
        t = self.thresholds()
        planes = [sign(x - self._thr(t[j], x)) for j in range(self.k)]
        if DENSITY:
            self.density = [float((p > 0).float().mean()) for p in planes]
        return [self._view(self.c[j], x) * planes[j] for j in range(self.k)]

    @staticmethod
    def _thr(t, x):
        return t

    @staticmethod
    def _view(c, x):
        return c


class BinaryLinear(nn.Module):
    """k independent binary weight sets: w = sum_i a_i * sign(W_i), a_i learned per output neuron."""

    def __init__(self, n_in, n_out, k):
        super().__init__()
        self.w = nn.ParameterList([nn.Parameter(torch.randn(n_out, n_in) * (1.0 / n_in ** 0.5)) for _ in range(k)])
        self.a = nn.Parameter(torch.full((k, n_out), 1.0 / k))

    def forward(self, planes):
        # one binary dot product per (weight set, activation plane): k_w x k_a of them
        out = 0
        for i, w in enumerate(self.w):
            bw = sign(w)
            for p in planes:
                out = out + self.a[i] * F.linear(p, bw)
        return out


def bits(k):
    """k = bits per element, the same for weights and activations (int), or (weight bits, activation bits)."""
    return (k, k) if isinstance(k, int) else tuple(k)


class BinaryMLP(nn.Module):
    """input -> H -> H/2 -> classes, every layer binary (k bits per weight and per activation, input included)."""

    def __init__(self, n_in, width, n_classes, k):
        super().__init__()
        dims = [n_in, width, width // 2, n_classes]
        kw, ka = bits(k)
        self.acts = nn.ModuleList([BinaryActivation(dims[i], ka) for i in range(3)])
        self.layers = nn.ModuleList([BinaryLinear(dims[i], dims[i + 1], kw) for i in range(3)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(3)])
        self.dims, self.kw, self.ka = dims, kw, ka

    def forward(self, x):
        x = x.flatten(1)
        for act, layer, norm in zip(self.acts, self.layers, self.norms):
            x = norm(layer(act(x)))
        return x

    def weight_bits(self):
        return self.kw * sum(self.dims[i] * self.dims[i + 1] for i in range(3))

    def binary_ops(self):
        """binary multiply-accumulates per inference (each one is an XNOR/AND + count of one bit pair)."""
        return self.kw * self.ka * sum(self.dims[i] * self.dims[i + 1] for i in range(3))


class BinaryActivation2d(BinaryActivation):
    """BinaryActivation with thresholds and scales per channel, for (N, C, H, W) maps."""

    @staticmethod
    def _thr(t, x):
        return t[None, :, None, None]

    @staticmethod
    def _view(c, x):
        return c.reshape(1, -1, 1, 1) if c.numel() > 1 else c


class BinaryConv2d(nn.Module):
    """k independent binary 3x3 (or 1x1) kernels: w = sum_i a_i * sign(W_i), a_i learned per output channel."""

    def __init__(self, c_in, c_out, k, kernel=3, stride=1):
        super().__init__()
        fan = c_in * kernel * kernel
        self.w = nn.ParameterList([nn.Parameter(torch.randn(c_out, c_in, kernel, kernel) * (1.0 / fan ** 0.5))
                                   for _ in range(k)])
        self.a = nn.Parameter(torch.full((k, c_out), 1.0 / k))
        self.stride, self.pad = stride, kernel // 2
        self.shape = (c_in, c_out, kernel)

    def forward(self, planes):
        out = 0
        for i, w in enumerate(self.w):
            bw = sign(w)
            for p in planes:
                out = out + self.a[i][None, :, None, None] * F.conv2d(p, bw, stride=self.stride, padding=self.pad)
        return out


class BiRealConv(nn.Module):
    """binary activation -> binary conv -> BN, plus its own shortcut (average + zero channels when it downsamples)."""

    def __init__(self, c_in, c_out, k, stride):
        super().__init__()
        kw, ka = bits(k)
        self.act = BinaryActivation2d(c_in, ka)
        self.conv = BinaryConv2d(c_in, c_out, kw, 3, stride)
        self.bn = nn.BatchNorm2d(c_out)
        self.stride, self.extra = stride, c_out - c_in

    def forward(self, x):
        y = self.bn(self.conv(self.act(x)))
        s = F.avg_pool2d(x, 2) if self.stride == 2 else x
        if self.extra:
            s = F.pad(s, (0, 0, 0, 0, 0, self.extra))
        return y + s


class BinaryResNet8(nn.Module):
    """BPR input (24 binary planes) -> binary stem -> 3 stages x 2 Bi-Real convs -> average -> binary classifier."""

    def __init__(self, c, n_classes, k):
        super().__init__()
        kw, ka = bits(k)
        self.stem = BinaryConv2d(24, c, kw, 3, 1)
        self.stem_bn = nn.BatchNorm2d(c)
        self.blocks = nn.Sequential(BiRealConv(c, c, k, 1), BiRealConv(c, c, k, 1),
                                    BiRealConv(c, 2 * c, k, 2), BiRealConv(2 * c, 2 * c, k, 1),
                                    BiRealConv(2 * c, 4 * c, k, 2), BiRealConv(4 * c, 4 * c, k, 1))
        self.head_act = BinaryActivation(4 * c, ka)
        self.head = BinaryLinear(4 * c, n_classes, kw)
        self.head_bn = nn.BatchNorm1d(n_classes)
        self.c, self.kw, self.ka, self.n_classes = c, kw, ka, n_classes

    def forward(self, planes):
        x = self.stem_bn(self.stem([planes]))         # the BPR planes are already bits: one input plane set
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head_bn(self.head(self.head_act(x)))

    def _convs(self):
        """(c_in, c_out, kernel, output pixels) of every binary convolution, for memory and operation counts."""
        c, out = self.c, []
        out.append((24, c, 3, 32 * 32))
        for c_in, c_out, px in ((c, c, 32), (c, c, 32), (c, 2 * c, 16), (2 * c, 2 * c, 16), (2 * c, 4 * c, 8), (4 * c, 4 * c, 8)):
            out.append((c_in, c_out, 3, px * px))
        return out

    def weight_bits(self):
        return self.kw * (sum(ci * co * kk * kk for ci, co, kk, _ in self._convs()) + 4 * self.c * self.n_classes)

    def binary_ops(self):
        stem = self.kw * 24 * self.c * 9 * 32 * 32          # input planes are single bits: k_w x 1
        rest = self.kw * self.ka * sum(ci * co * kk * kk * px for ci, co, kk, px in self._convs()[1:])
        return stem + rest + self.kw * self.ka * 4 * self.c * self.n_classes


def bpr(x_u8):
    """(N, 3, 32, 32) uint8 -> (N, 24, 32, 32) of +-1: the 8 bit planes of every channel, most significant first."""
    planes = [((x_u8 >> b) & 1) for b in range(7, -1, -1)]
    return torch.stack(planes, 2).flatten(1, 2).float() * 2 - 1


def thermometer(x_u8):
    """(N, 3, 32, 32) uint8 -> (N, 24, 32, 32) of +-1: 8 thermometer levels per channel (pixel >= 256*j/9, j = 1..8),
    the same number of planes as BPR."""
    levels = torch.arange(1, 9, device=x_u8.device) * 256 // 9
    return (x_u8[:, :, None] >= levels[None, None, :, None, None]).flatten(1, 2).float() * 2 - 1


ENCODE = bpr                   # set by --input


def augment(x_u8, gen):
    """random horizontal flip per image and one random shift (up to 4 pixels) per batch, uint8 on the GPU."""
    flip = torch.rand(x_u8.shape[0], generator=gen, device=x_u8.device) < 0.5
    x = torch.where(flip[:, None, None, None], x_u8.flip(3), x_u8)
    dx, dy = (int(v) for v in torch.randint(-4, 5, (2,), generator=gen, device=x_u8.device))
    x = torch.roll(x, (dy, dx), (2, 3))
    return x


# ---------------------------------------------------------------------------------------------------
# data (whole sets on the device) and training
# ---------------------------------------------------------------------------------------------------
def load(dataset, dev):
    if dataset == "cifar10":
        import torchvision
        root = os.path.join(HERE, "..", "data", "cifar10")
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        to = lambda d: torch.tensor(d.data).permute(0, 3, 1, 2).contiguous()          # uint8 (N, 3, 32, 32)
        return to(tr).to(dev), torch.tensor(tr.targets).to(dev), to(te).to(dev), torch.tensor(te.targets).to(dev)
    if dataset == "digits":
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        x, y = load_digits(return_X_y=True)
        x = (x / 16.0).astype(np.float32)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
        xs = torch.tensor(x)
        ys = torch.tensor(y, dtype=torch.long)
        xtr, ytr, xte, yte = xs[tr], ys[tr], xs[te], ys[te]
    else:
        import torchvision
        root = os.path.join(HERE, "..", "data", "fashionmnist")
        tr = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        xtr, ytr = tr.data.float().div(255).flatten(1), tr.targets
        xte, yte = te.data.float().div(255).flatten(1), te.targets
    mean, std = xtr.mean(), xtr.std()
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    return xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)


@torch.no_grad()
def accuracy(model, x, y, images=False):
    model.eval()
    correct = 0
    for i in range(0, len(y), 1000):
        xb = ENCODE(x[i:i + 1000]) if images else x[i:i + 1000]
        correct += int((model(xb).argmax(1) == y[i:i + 1000]).sum())
    model.train()
    return 100.0 * correct / len(y)


@torch.no_grad()
def plane_density(model, x, images):
    """share of +1 in every activation plane, averaged over the binary activations of the model (test samples)."""
    global DENSITY
    DENSITY = True
    model.eval()
    model(ENCODE(x[:1000]) if images else x[:1000])
    model.train()
    DENSITY = False
    acts = [m.density for m in model.modules() if isinstance(m, BinaryActivation) and m.density]
    k = max(len(d) for d in acts)
    return [float(np.mean([d[j] for d in acts if len(d) > j])) for j in range(k)]


def train(model, data, epochs, batch, seed, images=False):
    xtr, ytr, xte, yte = data
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=xtr.device).manual_seed(seed)
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(ytr), generator=gen, device=xtr.device)
        for i in range(0, len(ytr), batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            xb = ENCODE(augment(xtr[idx], gen)) if images else xtr[idx]
            F.cross_entropy(model(xb), ytr[idx]).backward()
            opt.step()
        sched.step()
        history.append(accuracy(model, xte, yte, images))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "resnet8"], default="mlp", help="resnet8 uses CIFAR-10")
    ap.add_argument("--dataset", choices=["fashion", "digits"], default="fashion", help="for --arch mlp")
    ap.add_argument("--width", type=int, default=0,
                    help="mlp: H of the pair network (flat also at 2H; default 256) | resnet8: base channels c (default 16)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--act-scale", choices=["feature", "layer", "equal", "pow2"], default="feature",
                    help="how the bits of an activation are combined: learned per feature (default), per layer, "
                         "equal (sum of counts, no multiplication) or powers of two (counts and shifts)")
    ap.add_argument("--input", choices=["bpr", "thermometer"], default="bpr", help="resnet8: input encoding (24 planes)")
    ap.add_argument("--widths", default="", help="sweep: comma-separated multipliers of the base width, e.g. 0.5,1,1.5,2 "
                                                 "(flat and pair at each one: accuracy vs binary ops)")
    ap.add_argument("--pair-on", choices=["both", "weights", "activations"], default="both",
                    help="where the pair network has two bits: weights and activations (default), or only one of them")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global ACT_SCALE, ENCODE
    ACT_SCALE = args.act_scale
    ENCODE = thermometer if args.input == "thermometer" else bpr
    images = args.arch == "resnet8"
    pair_k = {"both": (2, 2), "weights": (2, 1), "activations": (1, 2)}[args.pair_on]
    if images:
        args.dataset = "cifar10"
    data = load(args.dataset, dev)
    n_classes = 10
    if images:
        h = args.width or 16
        wide = int(round(h * 2 ** 0.5))
        configs = [("flat, channels c", h, 1), (f"flat, channels {wide} (~same memory as pair)", wide, 1),
                   (f"pair ({args.pair_on}), channels c", h, pair_k)]
        build = lambda width, k: BinaryResNet8(width, n_classes, k)
        shape = lambda width: f"binary ResNet-8, channels {width}/{2 * width}/{4 * width}, BPR input"
    else:
        h = args.width or 256
        n_in = data[0].shape[1]
        configs = [("flat, width H", h, 1), ("flat, width 2H (same memory as pair)", 2 * h, 1),
                   (f"pair ({args.pair_on}), width H", h, pair_k)]
        build = lambda width, k: BinaryMLP(n_in, width, n_classes, k)
        shape = lambda width: f"{n_in} -> {width} -> {width // 2} -> {n_classes}"
    if args.widths:
        mults = [float(m) for m in args.widths.split(",")]
        configs = []
        for m in mults:
            w = max(2, int(round(h * m)))
            configs += [(f"flat, x{m:g}", w, 1), (f"pair ({args.pair_on}), x{m:g}", w, pair_k)]
    # a 4th network: flat, as wide as it takes to do the SAME binary operations as the pair one (fair in compute)
    if not args.widths:
        pair_ops = build(h, pair_k).binary_ops()
        same_ops = min(range(h, 8 * h + 1), key=lambda w: abs(build(w, 1).binary_ops() - pair_ops))
        configs.append((f"flat, {'channels' if images else 'width'} {same_ops} (~same binary ops as pair)", same_ops, 1))
    print(f"[*] activations: --act-scale {args.act_scale}" + (f" | input: {args.input}" if images else ""))
    print(f"[*] {args.dataset}: {len(data[1])} training / {len(data[3])} test samples | {dev} | "
          f"{args.epochs} epochs | seeds {args.seeds} | {'c' if images else 'H'} = {h}")
    results = []
    for name, width, k in configs:
        finals, last5 = [], []
        t0 = time.time()
        for s in range(args.seeds):
            torch.manual_seed(42 + s)
            model = build(width, k).to(dev)
            hist = train(model, data, args.epochs, args.batch, 42 + s, images)
            finals.append(hist[-1])
            dens = plane_density(model, data[2], images)
            last5.append(float(np.mean(hist[-5:])))
        r = dict(config=name, width=width, weight_bits=bits(k)[0], activation_bits=bits(k)[1],
                 accuracy_last=float(np.mean(finals)),
                 accuracy_last_std=float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
                 accuracy_last5=float(np.mean(last5)), weight_kb=model.weight_bits() / 8 / 1024,
                 binary_ops=model.binary_ops(), plane_density=dens, seconds=time.time() - t0)
        results.append(r)
        print(f"\n    {name}   ({shape(width)}, {bits(k)[0]} bit(s) per weight, {bits(k)[1]} per activation)")
        print(f"      accuracy      : {r['accuracy_last']:.2f} +- {r['accuracy_last_std']:.2f} %  (last epoch)  |  "
              f"{r['accuracy_last5']:.2f} %  (mean of the last 5)")
        print(f"      weight memory : {r['weight_kb']:.1f} KB")
        print(f"      binary ops    : {r['binary_ops']:,} per inference")
        print("      plane density : " + "  ".join(f"plane {j + 1}: {100 * d:.0f}% ones" for j, d in enumerate(dens))
              + ("   (an upper plane at <= 50% can be skipped where it is empty)" if len(dens) > 1 else ""))
    if args.widths:
        print("\n    ACCURACY vs BINARY OPS (each pair next to the flat network of the same width)")
        print(f"    {'network':<26}{'accuracy':>10}{'last 5':>9}{'binary ops':>14}{'memory KB':>11}")
        for r in sorted(results, key=lambda r: r["binary_ops"]):
            print(f"    {r['config']:<26}{r['accuracy_last']:>9.2f}%{r['accuracy_last5']:>8.2f}%{r['binary_ops']:>14,}{r['weight_kb']:>11.1f}")
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        path = os.path.join(HERE, "results", f"pair_bits_{args.arch}_{args.dataset}_{args.pair_on}_{args.act_scale}_sweep.json")
        json.dump(dict(args=vars(args), results=results), open(path, "w"), indent=1)
        print(f"\n    -> {path}")
        return
    base = results[1] if bits(pair_k)[0] > 1 else results[0]      # same memory: pairs in weights double it
    print(f"\n    SAME MEMORY (approximately): pair ({results[2]['weight_kb']:.1f} KB) vs flat ({base['weight_kb']:.1f} KB)")
    print(f"      accuracy  : {results[2]['accuracy_last']:.2f}% vs {base['accuracy_last']:.2f}%  "
          f"({results[2]['accuracy_last'] - base['accuracy_last']:+.2f} points)")
    print(f"      binary ops: {results[2]['binary_ops'] / base['binary_ops']:.2f}x the flat network's")
    ops = results[3]
    print(f"\n    SAME BINARY OPS (approximately): pair ({results[2]['binary_ops']:,}) vs flat ({ops['binary_ops']:,})")
    print(f"      accuracy  : {results[2]['accuracy_last']:.2f}% vs {ops['accuracy_last']:.2f}%  "
          f"({results[2]['accuracy_last'] - ops['accuracy_last']:+.2f} points)  |  mean of the last 5: "
          f"{results[2]['accuracy_last5']:.2f}% vs {ops['accuracy_last5']:.2f}%")
    print(f"      memory    : {results[2]['weight_kb']:.1f} KB vs {ops['weight_kb']:.1f} KB")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"pair_bits_{args.arch}_{args.dataset}_{args.pair_on}_{args.act_scale}"
                                         + ("_thermo" if images and args.input == "thermometer" else "") + ".json")
    json.dump(dict(args=vars(args), results=results), open(path, "w"), indent=1)
    print(f"\n    -> {path}")


if __name__ == "__main__":
    main()
