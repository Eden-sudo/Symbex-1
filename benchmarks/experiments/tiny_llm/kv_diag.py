"""
kv_diag.py -- why low-bit keys break: a measurement of the integer model's attention, no new model.

    python kv_diag.py runs/binary_d384l8_s42 --windows 8

The int8 model runs on validation windows and every layer's q, k, v are kept. For each layer and head:
    D1  outlier channels: max channel RMS / median channel RMS of K before and after RoPE, Q and V; whether the top-4
        key channels are the same in the two halves of the windows (KIVI 2402.02750, KVQuant 2401.18079)
    D2  score error delta = s_quant - s_int8 (logits = q . k / sqrt(64)), weighted by the int8 attention p8: the
        per-query shift (softmax ignores it, HeadQ 2605.03562), the slope against the int8 score (a temperature bias)
        and the residual noise; the share of delta from the 4 largest key channels
    D3  KL(p8 || pq) against its small-noise value 1/2 Var_p8(delta): equal = the error enters quadratically
    D4  where the attention moves: total variation share at position 0, at the int8 top-1 key and in the recent 32;
        Spearman of a key's error size with its weight gain (the "Jensen bias", 2605.26266)
The low-bit keys come from intsim.kv_store (the same levels as intsim / the chip); "pre" configs quantize K per channel
BEFORE RoPE (KVQuant), then rotate (float, diagnosis only). Floats here are for measuring only.
"""
import argparse
import json
import math
import os

import torch

import intsim as I
import train as T

CONFIGS = [  # name, bits, mode, key scale factor
    ("K4", 4, "token", 1.0), ("K3", 3, "token", 1.0), ("K3 channel", 3, "channel", 1.0), ("K3 pre", 3, "pre", 1.0),
    ("K3 rot", 3, "rot", 1.0), ("K2", 2, "token", 1.0), ("K2 channel", 2, "channel", 1.0), ("K2 pre", 2, "pre", 1.0),
    ("K2 rot", 2, "rot", 1.0), ("K1 x pi/2", 1, "token", math.pi / 2), ("K1", 1, "token", 1.0)]


def rms_ratio(x):
    """x: b, h, t, d -> per head: max channel RMS / median channel RMS, and the RMS per channel (h, d)"""
    r = x.double().pow(2).mean((0, 2)).sqrt()
    return (r.amax(-1) / r.median(-1).values), r


def top4_same(x):
    """share of the top-4 channels (by RMS) shared by the two halves of the windows, per head"""
    b = x.shape[0] // 2
    a, c = (rms_ratio(z)[1].topk(4, -1).indices for z in (x[:b], x[b:]))
    return torch.tensor([len(set(a[i].tolist()) & set(c[i].tolist())) / 4 for i in range(a.shape[0])])


def dequant(x, bits, mode, scale, C, S):
    """the key the attention reads (float, Q12 units): kv_store's codes x scale / 127"""
    if mode == "pre":       # per channel before RoPE, then rotate (rounded back to integers for rope)
        q, m = I.kv_store(x, bits, "channel")
        return I.rope(I.rdiv(q * m[..., None], torch.full_like(q, 127)), C, S).double()
    q, m = I.kv_store(I.rope(x, C, S), bits, mode)
    return q.double() * (m.double() * scale)[..., None] / 127


@torch.no_grad()
def layer_stats(q, k, v, heads, C, S):
    b, t, d = q.shape
    q, k, v = (z.view(b, t, heads, -1).transpose(1, 2) for z in (q, k, v))
    kr = I.rope(k, C[:t], S[:t])
    out = {"D1": {}}
    for name, x in (("K pre-RoPE", k), ("K post-RoPE", kr), ("Q post-RoPE", I.rope(q, C[:t], S[:t])), ("V", v)):
        ratio, _ = rms_ratio(x)
        out["D1"][name] = {"max/median RMS": ratio.tolist(), "top4 same": top4_same(x).tolist()}
    _, krms = rms_ratio(kr)
    top4 = krms.topk(4, -1).indices                                                     # h, 4
    qq, qm = I.quant8(I.rope(q, C[:t], S[:t]))
    qh = qq.double() * qm.double()[..., None] / 127
    norm = 4096.0 ** 2 * 8
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    k8 = dequant(k, 8, "token", 1.0, C[:t], S[:t])
    s8 = (qh @ k8.transpose(-1, -2)) / norm
    s8 = s8.masked_fill(~causal, -math.inf)
    p8 = s8.softmax(-1)
    lp8 = s8.log_softmax(-1)
    s8z = s8.masked_fill(~causal, 0)
    mu = (p8 * s8z).sum(-1, keepdim=True)
    var8 = (p8 * (s8z - mu) ** 2).sum(-1, keepdim=True)
    pos = torch.arange(t, device=q.device)
    recent = causal & (pos[:, None] - pos[None, :] < 32)
    top1 = torch.zeros_like(causal).expand_as(p8).clone()
    top1.scatter_(-1, p8.argmax(-1, keepdim=True), True)
    sel = torch.zeros(heads, d // heads, dtype=torch.double, device=q.device).scatter_(1, top4, 1.0)
    for name, bits, mode, scale in CONFIGS:
        e = dequant(k, bits, mode, scale, C[:t], S[:t]) - k8                              # key error
        delta = ((qh @ e.transpose(-1, -2)) / norm).masked_fill(~causal, 0)
        dtop = ((qh * sel[None, :, None, :]) @ e.transpose(-1, -2) / norm).masked_fill(~causal, 0)
        sq = (s8z + delta).masked_fill(~causal, -math.inf)
        pq, lpq = sq.softmax(-1), sq.log_softmax(-1)
        c = (p8 * delta).sum(-1, keepdim=True)                                           # per-query shift
        a = (p8 * (delta - c) * (s8z - mu)).sum(-1, keepdim=True) / var8.clamp(min=1e-12)  # slope (temperature)
        r = delta - c - a * (s8z - mu)
        vd = (p8 * (delta - c) ** 2).sum(-1)
        vr = (p8 * r ** 2).sum(-1)
        vtop = (p8 * (dtop - (p8 * dtop).sum(-1, keepdim=True)) ** 2).sum(-1)
        kl = (p8 * (lp8 - lpq).masked_fill(~causal, 0)).sum(-1)
        tv = (pq - p8).abs()
        tvs = tv.sum(-1)
        gain = ((p8 * (lpq - lp8).masked_fill(~causal, 0)).sum(2) / p8.sum(2).clamp(min=1e-12))  # b, h, key
        esize = e.pow(2).sum(-1).sqrt()                                                  # b, h, key
        rho = spearman(esize.flatten(), gain.flatten())
        ent8 = -(p8 * lp8.masked_fill(~causal, 0)).sum(-1)
        entq = -(pq * lpq.masked_fill(~causal, 0)).sum(-1)
        out[name] = {
            "slope (temperature bias)": a.mean().item(),
            "Var_p8 delta (shift removed)": vd.mean().item(), "share of it explained by the slope":
                1 - vr.mean().item() / max(vd.mean().item(), 1e-30),
            "share of Var from top-4 key channels": vtop.mean().item() / max(vd.mean().item(), 1e-30),
            "KL": kl.mean().item(), "KL predicted 1/2 Var": 0.5 * vd.mean().item(),
            "TV": 0.5 * tvs.mean().item(),
            "TV share pos 0": (tv[..., 0].sum() / tvs.sum()).item(),
            "TV share int8 top-1": ((tv * top1).sum() / tvs.sum()).item(),
            "TV share recent 32": ((tv * recent).sum() / tvs.sum()).item(),
            "Spearman key error size vs weight gain": rho,
            "entropy int8 -> quant": [ent8.mean().item(), entq.mean().item()]}
    return out


def spearman(x, y):
    rx, ry = (z.argsort().argsort().double() for z in (x, y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return ((rx * ry).sum() / (rx.norm() * ry.norm())).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="runs/<weights>_<name>_s<seed>")
    ap.add_argument("--windows", type=int, default=8, help="validation windows of 256 tokens (an even number)")
    args = ap.parse_args()
    import sentencepiece as spm
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = T.pick_device(allow_mps=False)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(T.DATA, "tok4096.model"))
    model, a = I.load_model(args.run, sp.get_piece_size(), dev)
    P = I.to_device(I.integerize(model, a["weights"]), dev)
    val = T.load_split("val", dev)
    w = val[torch.arange(args.windows, device=dev)[:, None] * a["seq"] + torch.arange(a["seq"], device=dev)].long()
    cap, orig = [], I.attention

    def keep(q, k, v, heads, C, S, exp_t, kv=None, tq=None):
        cap.append((q, k, v))
        return orig(q, k, v, heads, C, S, exp_t, kv, tq)
    I.attention = keep
    I.forward(P, w)
    I.attention = orig
    C, S = P["cos"].to(dev), P["sin"].to(dev)
    res = {"run": os.path.basename(os.path.normpath(args.run)), "windows": args.windows, "layers": []}
    for li, (q, k, v) in enumerate(cap):
        st = layer_stats(q, k, v, P["heads"], C, S)
        res["layers"].append(st)
        d1 = st["D1"]
        print(f"layer {li}: max/median channel RMS (mean over heads) " + ", ".join(
            f"{n} {sum(x['max/median RMS']) / len(x['max/median RMS']):.1f}" for n, x in d1.items())
            + f"; top4 K pre same {sum(d1['K pre-RoPE']['top4 same']) / heads_of(d1):.2f}")
    print(f"\n{'config':12s} {'slope':>7s} {'Var d':>8s} {'slope%':>7s} {'top4%':>6s} {'KL':>8s} {'1/2Var':>8s}"
          f" {'TV':>6s} {'pos0':>5s} {'top1':>5s} {'rec32':>5s} {'rho':>6s} {'H8->Hq':>12s}")
    for name, *_ in CONFIGS:
        rows = [L[name] for L in res["layers"]]
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if not isinstance(rows[0][k], list)}
        h = [sum(r["entropy int8 -> quant"][i] for r in rows) / len(rows) for i in (0, 1)]
        res.setdefault("mean over layers", {})[name] = dict(mean, **{"entropy int8 -> quant": h})
        print(f"{name:12s} {mean['slope (temperature bias)']:7.3f} {mean['Var_p8 delta (shift removed)']:8.4f}"
              f" {100 * mean['share of it explained by the slope']:6.1f}% {100 * mean['share of Var from top-4 key channels']:5.1f}%"
              f" {mean['KL']:8.4f} {mean['KL predicted 1/2 Var']:8.4f} {mean['TV']:6.3f}"
              f" {mean['TV share pos 0']:5.2f} {mean['TV share int8 top-1']:5.2f} {mean['TV share recent 32']:5.2f}"
              f" {mean['Spearman key error size vs weight gain']:6.2f} {h[0]:5.2f}->{h[1]:5.2f}")
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"kv_diag_{res['run']}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")


def heads_of(d1):
    return len(d1["K pre-RoPE"]["top4 same"])


if __name__ == "__main__":
    main()
