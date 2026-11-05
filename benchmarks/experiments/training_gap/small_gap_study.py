"""
small_gap_study.py -- the training-gap and layer-by-layer study of the binary ResNet-8 (train_gap_study.py,
layer_probe_study.py) on the two other kinds of network: the 1D convolutions of conv1d/har_ultracompact (UniMiB-SHAR)
and the MLP of mlp/mlp_fashion (Fashion-MNIST). Do their binary networks lose accuracy in the same place and for
the same reasons?

Three networks of each shape, one recipe for the three:
    binary   the benchmark's network: 1-bit weights and activations (BatchNorm + sign); the first layer takes the
             benchmark's 8-bit input (accelerometer samples / pixels) with 1-bit weights (--first int8: int8 weights,
             har only); max pooling of bits = OR; the last layer 1-bit weights + BatchNorm, the class the largest score
    twin     the same graph, real weights, hardtanh activations (what the straight-through estimator pretends to train)
    relu     the same graph, real weights, ReLU activations

    python small_gap_study.py --bench har --arch max --seeds 42 43 44
    python small_gap_study.py --bench har --arch min --seeds 42 43 44
    python small_gap_study.py --bench mlp --hidden 256 128 --seeds 42 43 44

Recipe: Adam, --lr with cosine decay, --epochs (har 100, mlp 50), batches of 128 / 256, cross-entropy, labels only,
no augmentation, the data on the GPU, the LAST epoch (no checkpoint choice). Data: har, the benchmark's subject split
(har_data.py: validation subjects apart from the training ones); mlp, the CDT's split of the 60,000 training images
(model.split_indices, 54,000 / 6,000). The test sets are never read.

Then, on the validation data, as layer_probe_study.py does for ResNet-8:
    probe        linear probe on each layer's output (flattened), trained on up to 10,000 training samples
    margin       binary layers with binary inputs: share of outputs within one noise unit (sqrt(N), N = fan-in) of
                 the neuron's threshold (near), median distance in noise units (z_med)
    null         each neuron's weights shuffled within the neuron (--shuffles times), its output moved to the real
                 one's mean and standard deviation, the layer's BatchNorm + activation; sig = share of neurons more
                 class-selective (F ratio of the mean activation per sample) than every shuffle
    flip         binary only: 1 % of the first layer's output bits flipped; share of bits changed in every later
                 layer, share of predictions changed
--head int8: the output layer with int8 weights (one scale per class, straight-through rounding), its input bits
unchanged; the shape gets _hi8. Results: results/small_gap/<bench>_<shape>_<kind>_s<seed>.json; weights: runs/small_gap/ (gitignored). A shape
already trained is measured again from its weights (--retrain to train it again).
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layer_probe_study import activation, f_ratio, probe                 # noqa: E402

BENCH = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "small_gap")
RUNS = os.path.join(HERE, "runs", "small_gap")
KINDS = ("binary", "twin", "relu")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


class _Sign(torch.autograd.Function):
    """sign with 0 -> +1; straight-through gradient where |x| <= 1 (as the benchmarks' model.py)"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


ste_sign = _Sign.apply


def int8_weights(w):
    """har's int8 first layer: symmetric, one scale per output channel, straight-through rounding (har model.py)"""
    scale = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-8) / 127
    q = torch.clamp(torch.round(w / scale), -127, 127)
    return w / scale + (q - w / scale).detach()


class Stack(nn.Module):
    """a feed-forward stack: layers of (conv1d | linear, BatchNorm, activation, optional max pool), then a linear
    output with its BatchNorm. kind: binary / twin / relu (see the docstring)"""
    def __init__(self, kind, layers, n_in, classes, first="binary", head="binary"):
        super().__init__()
        self.kind, self.first, self.head = kind, first, head
        self.ops, self.bns, self.pools, self.conv = nn.ModuleList(), nn.ModuleList(), [], []
        for op in layers:
            if op[0] == "pool":
                self.pools[-1] = op[1]
                continue
            if op[0] == "conv":
                _, cin, cout, k = op
                self.ops.append(nn.Conv1d(cin, cout, k, bias=False))
            else:
                _, cin, cout = op
                self.ops.append(nn.Linear(cin, cout, bias=False))
            self.bns.append(nn.BatchNorm1d(cout))
            self.pools.append(0)
            self.conv.append(op[0] == "conv")
        self.out = nn.Linear(n_in, classes, bias=False)
        self.out_bn = nn.BatchNorm1d(classes)

    def weight(self, i):
        """the weights layer i uses (i = len(ops): the output layer)"""
        w = self.out.weight if i == len(self.ops) else self.ops[i].weight
        if (i == 0 and self.first == "int8") or (i == len(self.ops) and self.head == "int8"):
            return int8_weights(w)
        return ste_sign(w) if self.kind == "binary" else w

    def act(self, t):
        if self.kind == "binary":
            return ste_sign(t)
        return F.hardtanh(t) if self.kind == "twin" else F.relu(t)

    def apply(self, i, x, w):
        return F.conv1d(x, w) if self.conv[i] else F.linear(x, w)

    def forward(self, x, record=None, perturb=None):
        """record: a list that receives (layer input, pre-BatchNorm output, layer output) per layer; perturb(x): applied
        to the first layer's output"""
        for i in range(len(self.ops)):
            d = self.apply(i, x, self.weight(i))
            y = self.act(self.bns[i](d))
            if self.pools[i]:
                y = F.max_pool1d(y, self.pools[i])          # max of +-1 = OR of the bits
            if i == 0 and perturb is not None:
                y = perturb(y)
            if record is not None:
                record.append((x, d, y))
            x = y
        return self.out_bn(F.linear(x.flatten(1), self.weight(len(self.ops))))


def har_setup(args):
    har = _load("har_model_for_gap", os.path.join(BENCH, "conv1d", "har_ultracompact", "model.py"))
    data = _load("har_data_for_gap", os.path.join(BENCH, "conv1d", "har_ultracompact", "har_data.py"))
    layers, cin, length = [], har.AXES, har.LENGTH
    for op in har.ARCHS[args.arch]:
        if op[0] == "conv":
            layers.append(("conv", cin, op[1], op[2]))
            cin, length = op[1], length - op[2] + 1
        else:
            layers.append(("pool", op[1]))
            length //= op[1]

    def prep(x):                                            # N x 151 x 3 int8 -> N x 3 x 151, as har model.py
        return torch.as_tensor(x).float().transpose(1, 2) / 127
    (xt, yt), (xv, yv) = data.load("train"), data.load("validation")
    split = (prep(xt), torch.as_tensor(yt).long(), prep(xv), torch.as_tensor(yv).long())
    return layers, cin * length, har.CLASSES, split, f"har_{args.arch}" + ("_i8" if args.first == "int8" else "")


def mlp_setup(args):
    mlp = _load("mlp_model_for_gap", os.path.join(BENCH, "mlp", "mlp_fashion", "model.py"))
    import torchvision
    ds = torchvision.datasets.FashionMNIST(os.path.join(BENCH, "data", "fashionmnist"), train=True, download=True)
    x = (ds.data.float().view(-1, 784) / 255 - mlp.MEAN) / mlp.STD          # the benchmark's normalization
    y = ds.targets.long()
    keep, held = (torch.as_tensor(i) for i in mlp.split_indices(len(ds)))
    h1, h2 = args.hidden
    layers = [("linear", 784, h1), ("linear", h1, h2)]
    return layers, h2, 10, (x[keep], y[keep], x[held], y[held]), f"mlp_{h1}_{h2}"


@torch.no_grad()
def accuracy(net, x, y, bs=2000):
    net.eval()
    return 100.0 * sum((net(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs)) / len(x)


def train(net, split, seed, epochs, lr, bs, dev):
    xt, yt, xv, yv = split
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    sub = torch.randperm(len(xt), generator=torch.Generator().manual_seed(seed))[:5000].to(dev)
    history, t0 = [], time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(net(xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            history.append(dict(epoch=ep + 1, train_acc=accuracy(net, xt[sub], yt[sub]), val_acc=accuracy(net, xv, yv)))
    return history, round((time.time() - t0) / 60, 2)


@torch.no_grad()
def trace(net, x, bs=2000, perturb=None):
    """per layer: (input, pre-BatchNorm, output) for all of x; and the predictions"""
    net.eval()
    parts, preds = [], []
    for i in range(0, len(x), bs):
        rec = []
        preds.append(net(x[i:i + bs], rec, perturb).argmax(1))
        parts.append(rec)
    layers = [tuple(torch.cat([p[l][j] for p in parts]) for j in range(3)) for l in range(len(parts[0]))]
    return layers, torch.cat(preds)


def threshold(bn):
    sd = torch.sqrt(bn.running_var + bn.eps)
    return bn.running_mean - bn.bias * sd / bn.weight


def per_sample(t):
    return t.mean(2) if t.dim() == 3 else t


@torch.no_grad()
def measure(net, split, seed, shuffles, dev):
    xt, yt, xv, yv = split
    n = min(10000, len(xt))
    sel = torch.randperm(len(xt), generator=torch.Generator().manual_seed(seed))[:n].to(dev)
    tr_layers, _ = trace(net, xt[sel])
    va_layers, preds = trace(net, xv)
    classes = net.out.out_features
    probes = {"input": probe(xt[sel].flatten(1), yt[sel], xv.flatten(1), yv, seed, classes=classes)}
    for i, ((_, _, a), (_, _, b)) in enumerate(zip(tr_layers, va_layers)):
        probes[f"layer{i}"] = probe(a.flatten(1), yt[sel], b.flatten(1), yv, seed, classes=classes)
    g = torch.Generator().manual_seed(seed)
    layers = {}
    for i, (inp, d, _) in enumerate(va_layers):
        w = net.weight(i).detach()
        bn = net.bns[i]
        n_in = w[0].numel()
        real = f_ratio(per_sample(activation(bn(d), net.kind)), yv, classes)
        nulls = []
        for _ in range(shuffles):
            p = torch.stack([torch.randperm(n_in, generator=g) for _ in range(w.shape[0])]).to(dev)
            ws = torch.gather(w.flatten(1), 1, p).view_as(w)
            ds = net.apply(i, inp, ws)
            dims = (0, 2) if ds.dim() == 3 else (0,)
            shape = (1, -1, 1) if ds.dim() == 3 else (1, -1)
            ds = (ds - ds.mean(dims).view(shape)) / ds.std(dims).clamp_min(1e-6).view(shape) * d.std(dims).view(shape) \
                + d.mean(dims).view(shape)
            nulls.append(f_ratio(per_sample(activation(bn(ds), net.kind)), yv, classes))
        nulls = torch.stack(nulls)
        row = dict(n_in=n_in, f_real=real.median().item(), f_null=nulls.median().item(),
                   sig=(real > nulls.max(0).values).float().mean().item())
        if net.kind == "binary" and i > 0:
            shape = (1, -1, 1) if d.dim() == 3 else (1, -1)
            z = (d - threshold(bn).view(shape)).abs() / math.sqrt(n_in)
            row.update(near=(z < 1).float().mean().item(), z_median=z.flatten()[::7].median().item())
        layers[f"layer{i}"] = row
    res = dict(probe=probes, layers=layers)
    if net.kind == "binary":
        gen = torch.Generator(device=dev).manual_seed(seed)
        flipped, fpreds = trace(net, xv, perturb=lambda y: y * torch.where(
            torch.rand(y.shape, generator=gen, device=y.device) < 0.01, -1.0, 1.0))
        res["flip"] = dict(layers=[(b[2] != a[2]).float().mean().item() for a, b in zip(va_layers, flipped)],
                           predictions=(fpreds != preds).float().mean().item())
    return res


def run(kind, args, setup, dev):
    layers, n_in, classes, split, shape = setup
    split = tuple(t.to(dev) for t in split)
    shape += "_hi8" if args.head == "int8" else ""
    name = f"{shape}_{kind}_s{args.seed}"
    torch.manual_seed(args.seed)                           # the starting weights depend on the seed only
    net = Stack(kind, layers, n_in, classes, args.first, args.head).to(dev)
    path = os.path.join(RUNS, name + ".pth")
    out = os.path.join(OUT, name + ".json")
    if os.path.exists(path) and os.path.exists(out) and not args.retrain:
        net.load_state_dict(torch.load(path, map_location=dev))
        info = json.load(open(out))
    else:
        history, minutes = train(net, split, args.seed, args.epochs, args.lr, args.batch_size, dev)
        os.makedirs(RUNS, exist_ok=True)
        torch.save(net.state_dict(), path)
        info = dict(bench=args.bench, shape=shape, head=args.head, kind=kind, seed=args.seed, epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, n_train=len(split[0]), n_validation=len(split[2]),
                    train_acc=history[-1]["train_acc"], val_acc=history[-1]["val_acc"], history=history, minutes=minutes)
    info.update(measure(net, split, args.seed, args.shuffles, dev))
    os.makedirs(OUT, exist_ok=True)
    json.dump(info, open(out, "w"), indent=1)
    print(f"    -> {out} | train {info['train_acc']:.2f}% | validation {info['val_acc']:.2f}%")
    return info


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{statistics.mean(vals):.3g}" + (f"±{statistics.stdev(vals):.2g}" if len(vals) > 1 else "")


def summary(results):
    by = {}
    for r in results:
        by.setdefault((r["shape"], r["kind"]), []).append(r)
    for (shape, kind), rs in by.items():
        print(f"\n  {shape} {kind} ({len(rs)} seed(s)): validation {ms([r['val_acc'] for r in rs])} %, "
              f"train {ms([r['train_acc'] for r in rs])} %")
        print("    probe: " + " | ".join(f"{k} {ms([r['probe'][k] for r in rs])}" for k in rs[0]["probe"]))
        for l in rs[0]["layers"]:
            c = lambda key: ms([r["layers"][l].get(key) for r in rs])                  # noqa: E731
            print(f"    {l}: N {rs[0]['layers'][l]['n_in']} | F real {c('f_real')} | F null {c('f_null')} | "
                  f"sig {c('sig')} | near {c('near')} | z med {c('z_median')}")
        if "flip" in rs[0]:
            nl = len(rs[0]["flip"]["layers"])
            print("    flip 1 % of layer0: " + " | ".join(f"layer{k} {ms([r['flip']['layers'][k] for r in rs])}"
                                                      for k in range(nl))
                  + f" | predictions {ms([r['flip']['predictions'] for r in rs])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bench", choices=("har", "mlp"), required=True)
    ap.add_argument("--arch", default="max", help="har: model.py's ARCHS (min, max10, max5, max)")
    ap.add_argument("--first", choices=("binary", "int8"), default="binary", help="har: the first layer's weights")
    ap.add_argument("--head", choices=("binary", "int8"), default="binary",
                    help="the output layer's weights: binary (the benchmarks') or int8 (its input stays binary)")
    ap.add_argument("--hidden", type=int, nargs=2, default=[256, 128], help="mlp: the two hidden widths")
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    ap.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=0, help="default: har 100, mlp 50")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=0, help="default: har 128, mlp 256")
    ap.add_argument("--shuffles", type=int, default=19)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    if args.first == "int8" and args.bench != "har":
        sys.exit("[X] --first int8: har only")
    args.epochs = args.epochs or (100 if args.bench == "har" else 50)
    args.batch_size = args.batch_size or (128 if args.bench == "har" else 256)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    setup = har_setup(args) if args.bench == "har" else mlp_setup(args)
    results = []
    for seed in args.seeds:
        args.seed = seed
        for kind in args.kinds:
            print(f"\n[*] {setup[4]} {kind} seed {seed}")
            results.append(run(kind, args, setup, dev))
    summary(results)


if __name__ == "__main__":
    main()
