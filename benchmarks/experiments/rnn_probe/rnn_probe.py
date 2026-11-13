"""
rnn_probe.py -- does an error in a binary recurrent network grow step after step? (see README.md and LOGBOOK.md)

A minimal Elman RNN on UniMiB-SHAR (the windows of conv1d/har_ultracompact read as sequences: 3 axes per step,
151 steps, 17 classes), the same graph in three kinds, one recipe for the three:
    binary   1-bit recurrent and input weights, 1-bit state: h_t = sign(BN(W_x x_t + W_h h_{t-1}))
    twin     real weights, hardtanh state (what the straight-through estimator pretends to train)
    tanh     real weights, tanh state (the textbook RNN)
    int8     the INT8 reference: weights quantized to 8 bits (symmetric, per tensor) and an 8-bit hardtanh state
             (-127..127), trained with quantization in the loop (QAT, straight-through rounding); the BatchNorm and
             the input gain stay real (folded into the requantization scale on a chip)
The BatchNorm is shared by every step (one threshold per neuron: what a binary chip would run). The class: a linear
layer on the last state (1-bit weights + BatchNorm for binary).

Measured on the validation subjects (the test set is never read), per seed:
    acc        validation and training accuracy (last epoch, no checkpoint choice)
    probe      a linear probe on the state h_t at several steps: how much the state "knows" as the sequence goes on
    margin     binary: share of pre-activations within one noise unit (sqrt(N), N = fan-in) of the threshold
    flip       at step t0, the sign of 1 % of the state's units is flipped; share of units whose sign differs from
               the clean run at every later step, and share of predictions changed. This is the recurrent version
               of small_gap_study.py's flip test (feed-forward HAR max: 1 % -> 8 % of the last layer's bits)
    derrida    the damage map at t0 = T/4 for d0 = 0.5-20 % of the state flipped, with the input on ("driven") and
               with the input set to zero from t0 on ("autonomous": only the recurrence), for the trained network
               and for the same network before training ("init"); d after 1, 5, 20 steps and at the end
    margin2    binary: share of neurons whose count lies within 2 (one flipped input bit) and 4 of the threshold
Weights: results/rnn_h<H>_<kind>_s<seed>.pth.
Options for the binary kind (each one alone, against the same recipe without it):
    --ste rect|tri|tanh   surrogate gradient of the state's sign: 1{|x|<=1} (default), max(0, 1-|x|), sech^2(x)
    --margin-loss M       adds lambda * mean(max(0, M - |count - threshold|)) over every step and neuron
    --sparse K            each neuron hears only K fixed random neurons of the state (mask on W_h)

    python rnn_probe.py --seeds 42 43 44                 # hidden 128, 60 epochs
    python rnn_probe.py --hidden 64 --seeds 42 --epochs 20
Results: results/rnn_h<H>_<kind>_s<seed>.json next to this script.
"""
import argparse
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
GAP = os.path.join(os.path.dirname(HERE), "training_gap")                # reuses small_gap_study's HAR setup
sys.path.insert(0, GAP)
import small_gap_study as sg                                              # noqa: E402

KINDS = ("binary", "twin", "tanh", "int8")


def fake_quant(x, scale):
    """round to the 8-bit grid of `scale`, gradient straight through"""
    return x + ((x / scale).round().clamp(-127, 127) * scale - x).detach()


class _SignTri(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (1 - x.abs()).clamp(min=0)


class _SignTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (1 - torch.tanh(x) ** 2)


STE = dict(rect=sg.ste_sign, tri=_SignTri.apply, tanh=_SignTanh.apply)
OUT = os.path.join(HERE, "results")


class RNN(nn.Module):
    def __init__(self, kind, n_in, hidden, classes, in_gain=False, ste="rect", margin=0.0, sparse=0):
        super().__init__()
        self.kind, self.hidden = kind, hidden
        self.ste, self.margin, self.aux = STE[ste], margin, None
        mask = torch.ones(hidden, hidden)
        if sparse:                                   # K ones per row, fixed (the seed set by the caller)
            mask = torch.zeros(hidden, hidden).scatter_(1, torch.rand(hidden, hidden).argsort(1)[:, :sparse], 1.0)
        self.register_buffer("mask", mask)
        self.fan_in = sparse or hidden
        # in_gain: a learned per-neuron scale on the input term (an int8 input layer on a chip), so 3 input values
        # are not drowned by the H-bit recurrent count; started at sqrt(H)
        self.gain = nn.Parameter(torch.full((hidden,), math.sqrt(hidden))) if in_gain else None
        self.wx = nn.Linear(n_in, hidden, bias=False)
        self.wh = nn.Linear(hidden, hidden, bias=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, classes, bias=False)
        self.out_bn = nn.BatchNorm1d(classes)
        with torch.no_grad():
            self.wh.weight.mul_(0.5)                # the real kinds start contractive (binary ignores the scale)

    def w(self, lin):
        if self.kind == "int8":
            return fake_quant(lin.weight, lin.weight.detach().abs().max().clamp(min=1e-8) / 127)
        return sg.ste_sign(lin.weight) if self.kind == "binary" else lin.weight

    def act(self, t):
        if self.kind == "binary":
            return self.ste(t)
        if self.kind == "int8":
            return fake_quant(F.hardtanh(t), 1 / 127)
        return F.hardtanh(t) if self.kind == "twin" else torch.tanh(t)

    def forward(self, x, record=None, flip=None, zero_from=None):
        """x: N x T x n_in. record: list receiving (pre-BN, state) per step. flip: (t0, mask N x H of +-1).
        zero_from: the input term is 0 from that step on (the network runs on its recurrence alone)"""
        wx, wh = self.w(self.wx), self.w(self.wh) * self.mask
        h = torch.zeros(x.shape[0], self.hidden, device=x.device)
        xs = F.linear(x, wx)                         # every step's input term at once
        if self.gain is not None:
            xs = xs * self.gain
        if zero_from is not None:
            xs = xs.clone()
            xs[:, zero_from:] = 0
        pen = []
        for t in range(x.shape[1]):
            d = xs[:, t] + F.linear(h, wh)
            z = self.bn(d)
            if self.margin and self.training:        # |count - threshold| = |z| * sd / |gamma|, in count units
                dist = z.abs() * torch.sqrt(self.bn.running_var + self.bn.eps) / self.bn.weight.abs()
                pen.append((self.margin - dist).clamp(min=0).mean())
            h = self.act(z)
            if flip is not None and t == flip[0]:
                h = h * flip[1]
            if record is not None:
                record.append((d, h))
        self.aux = torch.stack(pen).mean() if pen else None
        return self.out_bn(F.linear(h, self.w(self.out)))


def train(net, split, seed, epochs, lr, bs, dev, lam=0.0):
    """small_gap_study.train with the margin term added when the network has one (the same recipe otherwise)"""
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
            if net.aux is not None:
                loss = loss + lam * net.aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            history.append(dict(epoch=ep + 1, train_acc=sg.accuracy(net, xt[sub], yt[sub]),
                                val_acc=sg.accuracy(net, xv, yv)))
    return history, round((time.time() - t0) / 60, 2)


def setup(stride=1):
    """stride > 1 keeps one step in `stride` (shorter sequences, same windows)"""
    args = argparse.Namespace(arch="max", first="binary")
    _, _, classes, split, _ = sg.har_setup(args)
    xt, yt, xv, yv = split                           # N x 3 x 151 -> N x 151 x 3
    return classes, (xt[:, :, ::stride].transpose(1, 2).contiguous(), yt,
                     xv[:, :, ::stride].transpose(1, 2).contiguous(), yv)


@torch.no_grad()
def run_all(net, x, bs=2000, record_steps=None, flip=None):
    """predictions, and the states at record_steps (dict step -> N x H), and the pre-activations of those steps"""
    net.eval()
    preds, states, pres = [], {}, {}
    for i in range(0, len(x), bs):
        rec = []
        f = None if flip is None else (flip[0], flip[1][i:i + bs])
        preds.append(net(x[i:i + bs], rec, f).argmax(1))
        for s in (record_steps or []):
            states.setdefault(s, []).append(rec[s][1])
            pres.setdefault(s, []).append(rec[s][0])
    return (torch.cat(preds), {s: torch.cat(v) for s, v in states.items()},
            {s: torch.cat(v) for s, v in pres.items()})


@torch.no_grad()
def derrida(net, x, seed, dev, t0, bs=2000):
    """d_t (% of state signs that differ) after flipping a share d0 of the state at t0, input on / off"""
    net.eval()
    T, out = x.shape[1], {}
    lags = [l for l in (1, 5, 20) if t0 + l < T] + [T - 1 - t0]
    for mode, zf in (("driven", None), ("autonomous", t0 + 1)):
        rows = {}
        for d0 in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            k = max(1, round(d0 * net.hidden))
            g = torch.Generator().manual_seed(seed + int(1000 * d0))
            idx = torch.rand(len(x), net.hidden, generator=g).argsort(1)[:, :k].to(dev)
            mask = torch.ones(len(x), net.hidden, device=dev).scatter_(1, idx, -1.0)
            diff = {l: 0.0 for l in lags}
            for i in range(0, len(x), bs):
                ra, rb = [], []
                net(x[i:i + bs], ra, None, zf)
                net(x[i:i + bs], rb, (t0, mask[i:i + bs]), zf)
                for l in lags:
                    diff[l] += ((ra[t0 + l][1] >= 0) != (rb[t0 + l][1] >= 0)).float().sum().item()
            n = len(x) * net.hidden
            rows[f"{100 * k / net.hidden:.2f}"] = {str(l): round(100 * v / n, 2) for l, v in diff.items()}
        out[mode] = rows
    return out


@torch.no_grad()
def measure(net, split, seed, dev):
    xt, yt, xv, yv = split
    T = xv.shape[1]
    steps = sorted(set([0, 4, 9] + list(range(19, T, 20)) + [T - 1]))
    out = dict(val_acc=sg.accuracy(net, xv, yv), train_acc=sg.accuracy(net, xt, yt))

    g = torch.Generator().manual_seed(seed)
    sub = torch.randperm(len(xt), generator=g)[:5000].to(dev)
    _, st_tr, _ = run_all(net, xt[sub], record_steps=steps)
    clean, st_va, pre_va = run_all(net, xv, record_steps=list(range(T)))
    out["probe"] = {str(s): round(sg.probe(st_tr[s], yt[sub], st_va[s], yv, seed, classes=int(yt.max()) + 1), 2)
                    for s in steps}

    if net.kind == "binary":                         # threshold of each neuron, in the units of the count
        bn = net.bn
        thr = sg.threshold(bn)
        noise = math.sqrt(net.wx.in_features + net.fan_in)
        near = [((pre_va[s] - thr).abs() <= noise).float().mean().item() for s in steps if s > 0]
        out["margin_near"] = round(100 * statistics.mean(near), 1)
        # the count in its own units: (pre - threshold) as an integer distance; one flipped input bit moves it by 2
        within = {w: statistics.mean(((pre_va[s] - thr).abs() <= w).float().mean().item() for s in steps if s > 0)
                  for w in (2, 4)}
        out["margin2"] = {str(w): round(100 * v, 1) for w, v in within.items()}

    flips = {}
    for t0 in (9, T // 4, T // 2):
        mask = torch.ones(len(xv), net.hidden, device=dev)
        k = max(1, round(0.01 * net.hidden))
        idx = torch.rand(len(xv), net.hidden, generator=torch.Generator().manual_seed(seed + t0)).argsort(1)[:, :k].to(dev)
        mask.scatter_(1, idx, -1.0)
        pert, st_p, _ = run_all(net, xv, record_steps=list(range(t0, T)), flip=(t0, mask))
        diff = {}
        for s in range(t0, T):
            a, b = st_va[s], st_p[s]
            diff[s - t0] = round(100 * ((a >= 0) != (b >= 0)).float().mean().item(), 3)
        flips[str(t0)] = dict(k_flipped=k, sign_changed_by_lag={str(l): diff[l] for l in sorted(diff) if
                                                                 l in (0, 1, 2, 5, 10, 20, 50, 100) or l == T - 1 - t0},
                              preds_changed=round(100 * (pert != clean).float().mean().item(), 2))
    out["flip"] = flips
    out["derrida"] = derrida(net, xv, seed, dev, T // 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1, help="keep one time step in STRIDE")
    ap.add_argument("--in-gain", action="store_true", help="learned per-neuron scale on the input term")
    ap.add_argument("--ste", choices=sorted(STE), default="rect")
    ap.add_argument("--margin-loss", type=float, default=0.0, metavar="M")
    ap.add_argument("--margin-lambda", type=float, default=3e-3)
    ap.add_argument("--sparse", type=int, default=0, metavar="K")
    ap.add_argument("--tag", default="", help="suffix of the result files (one per option set)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes, split = setup(args.stride)
    split = tuple(t.to(dev) for t in split)
    os.makedirs(OUT, exist_ok=True)
    for seed in args.seeds:
        for kind in args.kinds:
            torch.manual_seed(seed)
            net = RNN(kind, split[0].shape[2], args.hidden, classes, args.in_gain,
                      args.ste, args.margin_loss, args.sparse).to(dev)
            with torch.no_grad():                    # the untrained network: BatchNorm statistics of a few batches
                net.train()
                for i in range(0, 2048, 256):
                    net(split[0][i:i + 256])
                init = derrida(net, split[2], seed, dev, split[2].shape[1] // 4)
            hist, minutes = train(net, split, seed, args.epochs, args.lr, args.bs, dev, args.margin_lambda)
            res = dict(kind=kind, seed=seed, hidden=args.hidden, epochs=args.epochs, lr=args.lr, stride=args.stride, in_gain=args.in_gain, ste=args.ste,
                       margin_loss=args.margin_loss, margin_lambda=args.margin_lambda, sparse=args.sparse,
                       minutes=minutes, history=hist,
                       init_derrida=init, **measure(net, split, seed, dev))
            torch.save(net.state_dict(), os.path.join(OUT, f"rnn_h{args.hidden}{args.tag}_{kind}_s{seed}.pth"))
            path = os.path.join(OUT, f"rnn_h{args.hidden}{args.tag}_{kind}_s{seed}.json")
            with open(path, "w") as f:
                json.dump(res, f, indent=1)
            fl = res["flip"][str(split[2].shape[1] // 4)]
            print(f"[{kind:>6} s{seed}] val {res['val_acc']:.2f}  train {res['train_acc']:.2f}  "
                  f"probe {list(res['probe'].values())[:3]}..{list(res['probe'].values())[-1]}  "
                  f"flip@T/4 {fl['sign_changed_by_lag']}  preds {fl['preds_changed']} %  ({minutes} min)", flush=True)
            for name, dd in (("init", init), ("trained", res["derrida"])):
                for mode, rows in dd.items():
                    print(f"      derrida {name:>7} {mode:>10}: " + "  ".join(
                        f"{d0}%->{'/'.join(str(v) for v in r.values())}" for d0, r in rows.items()), flush=True)


if __name__ == "__main__":
    main()
