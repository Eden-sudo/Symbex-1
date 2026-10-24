"""
conv.py -- the exporter of convolutional networks: a binarized standard ResNet (the CDT's "plain" method on a
torchvision-style CIFAR ResNet) in integers. (plain.py exports the SYMBEX v3 ReActNet-style ResNet instead.)

The trained model (its state dict) is FOLDED into integers once, and SIMULATED with integer arithmetic only. The
simulation is the ground truth the C library must reproduce bit for bit; its agreement with the PyTorch model is the
fidelity of the folding.

The model (keys of the state dict, "net." prefix optional):
    conv1 + bn1 + ReLU                           stem; its input is the normalized image (8-bit pixels, not binarized)
    layer<s>.<i>: conv1 (RSign) + bn1 + ReLU, conv2 (RSign) + bn2, + shortcut, ReLU
        shortcut = the block input, or downsample.0 (RSign, 1x1 conv, stride 2) + downsample.1 (BatchNorm)
    average pool + fc (RSign, binary weights, bias)
Every binary conv / linear computes sign(x - beta) (0 -> +1) times sign(w) (>= 0 -> +1), scaled by alpha = mean |w| per
output channel; zero padding adds nothing.

In integers (values: Q_ACT fractional bits; scales: Q_MULT):
    stem      per color channel c: U_c = sum of pixel * sign over the window, V_c = sum of the signs over the in-image part
              of the window (the normalized padding is 0). value = ReLU(((sum_c A_c U_c + B_c V_c) + half) >> (Q_MULT - Q_ACT) + H)
    conv1     input bits = value >= ceil(beta * 2^Q_ACT); raw = sum of +-1 over the in-image window.
              Its BatchNorm, ReLU and the RSign of conv2 fold into ONE integer threshold per channel on raw
              (weights negated where the folded scale is negative): out bit = raw >= T. No fixed point at all.
    conv2     value = ReLU(((S raw + half) >> (Q_MULT - Q_ACT)) + H + shortcut)
    shortcut  the block input value, or (S raw + half) >> (Q_MULT - Q_ACT) + H of the 1x1 stride-2 conv of its bits
    head      bit = sum of the final map over its positions >= ceil(n_positions * beta * 2^Q_ACT) (no division);
              logit = ((S raw + half) >> (Q_MULT - Q_ACT)) + bias; class = the first largest logit
The simulation runs with torch in float64: every value is an integer below 2^53, so it is exact (and fast on a GPU).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1


def _clamp_threshold(v):
    """an integer threshold for raw >= T; beyond the int32 range (a BatchNorm scale near 0) it means always / never"""
    if math.isnan(v):
        return INT_MAX
    if v <= INT_MIN:
        return INT_MIN                                   # raw >= T always holds
    if v > INT_MAX:
        return INT_MAX                                   # never (raw can not get there)
    return v


def _strip(sd):
    return {(k[4:] if k.startswith("net.") else k): v.detach().double().cpu() for k, v in sd.items()}


def _bn(sd, p, eps=1e-5):
    g = sd[f"{p}.weight"] / torch.sqrt(sd[f"{p}.running_var"] + eps)
    return g, sd[f"{p}.bias"] - sd[f"{p}.running_mean"] * g


def _sign(w):
    return torch.where(w >= 0, 1.0, -1.0).double()


def _alpha(w):
    return w.abs().mean(dim=tuple(range(1, w.dim())))


def _blocks(sd):
    names = sorted({".".join(k.split(".")[:2]) for k in sd if k.startswith("layer")},
                   key=lambda n: (int(n[5:].split(".")[0]), int(n.split(".")[1])))
    return names


def fold(state_dict, mean, std, q_act=16, q_mult=24, input_size=32):
    """state dict of the trained binary ResNet -> integer parameters (python ints in torch float64 tensors)."""
    sd = _strip(state_dict)
    one, half_q = 2.0 ** q_mult, 2.0 ** q_act
    rnd = lambda t: torch.round(t)
    P = dict(q_act=q_act, q_mult=q_mult, blocks=[])
    # stem: x_norm = a_c * pixel + b_c
    w = sd["conv1.weight"]
    g, h = _bn(sd, "bn1")
    k = g * _alpha(w)
    a = torch.tensor([1 / (255 * s) for s in std], dtype=torch.float64)
    b = torch.tensor([-m / s for m, s in zip(mean, std)], dtype=torch.float64)
    P["stem"] = dict(sign=_sign(w), A=rnd(one * k[:, None] * a[None, :]), B=rnd(one * k[:, None] * b[None, :]),
                     H=rnd(half_q * h))
    res = input_size
    for name in _blocks(sd):
        ds = f"{name}.downsample.0.weight" in sd
        stride = 2 if ds else 1
        res //= stride
        w1, w2 = sd[f"{name}.conv1.weight"], sd[f"{name}.conv2.weight"]
        g1, h1 = _bn(sd, f"{name}.bn1")
        g2, h2 = _bn(sd, f"{name}.bn2")
        beta1 = sd[f"{name}.conv1.rsign.beta"].flatten()
        beta2 = sd[f"{name}.conv2.rsign.beta"].flatten()
        # conv1 -> bn1 -> ReLU -> RSign(beta2) as one threshold on the integer raw count
        kk, c = g1 * _alpha(w1), beta2 - h1
        T = torch.empty(len(kk), dtype=torch.float64)
        flip = torch.zeros(len(kk), dtype=torch.float64)
        for o in range(len(kk)):
            ko, co = float(kk[o]), float(c[o])
            if float(beta2[o]) <= 0:                           # max(z, 0) >= beta2 always holds
                T[o] = INT_MIN
            elif ko > 0:
                r = co / ko
                T[o] = _clamp_threshold(math.ceil(r) if math.isfinite(r) else r)
            elif ko < 0:                                       # raw <= c / k  <=>  (-raw) >= -floor(c / k)
                r = co / ko
                flip[o], T[o] = 1, _clamp_threshold(-math.floor(r) if math.isfinite(r) else -r)
            else:
                T[o] = INT_MIN if float(h1[o]) >= float(beta2[o]) else INT_MAX
        s1 = _sign(w1) * torch.where(flip > 0, -1.0, 1.0)[:, None, None, None]
        blk = dict(name=name, stride=stride, res_out=res,
                   t_in1=torch.ceil(half_q * beta1), sign1=s1, T=T,
                   sign2=_sign(w2), S2=rnd(one * g2 * _alpha(w2)), H2=rnd(half_q * h2))
        if ds:
            wd = sd[f"{name}.downsample.0.weight"]
            gd, hd = _bn(sd, f"{name}.downsample.1")
            blk.update(t_ind=torch.ceil(half_q * sd[f"{name}.downsample.0.rsign.beta"].flatten()), signd=_sign(wd),
                       Sd=rnd(one * gd * _alpha(wd)), Hd=rnd(half_q * hd))
        P["blocks"].append(blk)
    wf = sd["fc.weight"]
    n_pos = res * res
    P["head"] = dict(n_pos=n_pos, t_sum=torch.ceil(n_pos * half_q * sd["fc.rsign.beta"].flatten()), sign=_sign(wf),
                     S=rnd(one * _alpha(wf)), bias=rnd(half_q * sd["fc.bias"]) if "fc.bias" in sd else torch.zeros(len(wf)))
    return P


def _pm(bits):
    return 2 * bits - 1


def _shift(t, P):
    """(t + half) >> (Q_MULT - Q_ACT), arithmetic (floor), on exact integers"""
    sh = P["q_mult"] - P["q_act"]
    return torch.floor((t + 2.0 ** (sh - 1)) / 2.0 ** sh)


@torch.no_grad()
def simulate(P, images_u8, device="cpu", batch=500, return_logits=False):
    """images_u8: (N, H, W, 3) uint8. Returns the predicted classes (and the integer logits)."""
    dev = torch.device(device)
    to = lambda t: t.to(dev)
    st, hd = P["stem"], P["head"]
    preds, logits = [], []
    for i in range(0, len(images_u8), batch):
        p = to(torch.as_tensor(np.ascontiguousarray(images_u8[i:i + batch])).permute(0, 3, 1, 2).double())
        ones = torch.ones_like(p[:1, :1])
        acc = 0
        for c in range(3):
            U = F.conv2d(p[:, c:c + 1], to(st["sign"][:, c:c + 1]), padding=1)
            V = F.conv2d(ones, to(st["sign"][:, c:c + 1]), padding=1)
            acc = acc + to(st["A"][:, c])[None, :, None, None] * U + to(st["B"][:, c])[None, :, None, None] * V
        v = torch.relu(_shift(acc, P) + to(st["H"])[None, :, None, None])
        for b in P["blocks"]:
            col = lambda t: to(t)[None, :, None, None]
            bits1 = (v >= col(b["t_in1"])).double()
            r1 = F.conv2d(_pm(bits1), to(b["sign1"]), stride=b["stride"], padding=1)
            bits2 = (r1 >= col(b["T"])).double()
            r2 = F.conv2d(_pm(bits2), to(b["sign2"]), padding=1)
            val = _shift(col(b["S2"]) * r2, P) + col(b["H2"])
            if "signd" in b:
                bd = (v >= col(b["t_ind"])).double()
                rd = F.conv2d(_pm(bd), to(b["signd"]), stride=2)
                ident = _shift(col(b["Sd"]) * rd, P) + col(b["Hd"])
            else:
                ident = v
            v = torch.relu(val + ident)
        bits = (v.sum(dim=(2, 3)) >= to(hd["t_sum"])[None, :]).double()
        raw = _pm(bits) @ to(hd["sign"]).T
        lg = _shift(to(hd["S"])[None, :] * raw, P) + to(hd["bias"])[None, :]
        lg = lg.cpu().numpy().astype(np.int64)
        logits.append(lg)
        preds.append(np.argmax(lg, axis=1))                    # numpy: the FIRST largest, as the C classifier
    preds = np.concatenate(preds)
    return (preds, np.concatenate(logits)) if return_logits else preds


@torch.no_grad()
def forward_float(state_dict, images_u8, mean, std, device="cpu", batch=500):
    """The binary model in floating point, as the CDT trains it (to check the folding against)."""
    sd = _strip(state_dict)
    dev = torch.device(device)
    to = lambda t: t.to(dev)
    m = to(torch.tensor(mean, dtype=torch.float64))[None, :, None, None]
    s = to(torch.tensor(std, dtype=torch.float64))[None, :, None, None]
    bn = lambda x, p: x * to(_bn(sd, p)[0])[None, :, None, None] + to(_bn(sd, p)[1])[None, :, None, None]
    rsign = lambda x, p: _pm((x - to(sd[p].flatten())[None, :, None, None] >= 0).double())
    wb = lambda p: to(_sign(sd[p]) * _alpha(sd[p])[:, None, None, None])
    preds = []
    for i in range(0, len(images_u8), batch):
        x = to(torch.as_tensor(np.ascontiguousarray(images_u8[i:i + batch])).permute(0, 3, 1, 2).double() / 255)
        x = (x - m) / s
        v = torch.relu(bn(F.conv2d(x, wb("conv1.weight"), padding=1), "bn1"))
        for name in _blocks(sd):
            ds = f"{name}.downsample.0.weight" in sd
            stride = 2 if ds else 1
            y = torch.relu(bn(F.conv2d(rsign(v, f"{name}.conv1.rsign.beta"), wb(f"{name}.conv1.weight"),
                                       stride=stride, padding=1), f"{name}.bn1"))
            y = bn(F.conv2d(rsign(y, f"{name}.conv2.rsign.beta"), wb(f"{name}.conv2.weight"), padding=1), f"{name}.bn2")
            if ds:
                ident = bn(F.conv2d(rsign(v, f"{name}.downsample.0.rsign.beta"), wb(f"{name}.downsample.0.weight"),
                                    stride=2), f"{name}.downsample.1")
            else:
                ident = v
            v = torch.relu(y + ident)
        f = v.mean(dim=(2, 3))
        fb = _pm((f - to(sd["fc.rsign.beta"].flatten())[None, :] >= 0).double())
        wf = to(_sign(sd["fc.weight"]) * _alpha(sd["fc.weight"])[:, None])
        out = fb @ wf.T + (to(sd["fc.bias"])[None, :] if "fc.bias" in sd else 0)
        preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)
