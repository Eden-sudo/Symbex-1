"""k_anatomy.py -- the key K taken apart: what makes it hard to store in few bits, and which change fixes it.

    python k_anatomy.py --runs fp32_v512d64l5kv4_lr3e-3_20k binary_v512d64l5kv4_lr3e-3_20k --seeds 42 43 44

No training. For each trained model, layer and key head (head width 8 at 260K):
    stats   outlier channels (max channel RMS / median, after RoPE), share of the key's energy in its mean over tokens
            before RoPE (||mean||^2 / mean ||k||^2), peak of a token's key (max |k_c| / RMS over its channels: what a
            per-token scale must cover), attention entropy (nats, mean over queries; flat attention = high), and the
            score split q . k = q . RoPE(mean) + q . (k - mean): variance over the visible keys of each part / of the
            whole score, pooled over queries (sum > 1: the two parts partly cancel; the mean part varies with the key's position only: a learned position preference)
    cache   the key cache stored at 4 / 3 / 2 bits, the rest of the model unchanged; validation loss rise for:
              token       per token and head, scale from that key (as train.py / intsim.py: max for 4 bits)
              smooth      each channel divided by a fixed scale first (its RMS on TRAINING windows, one per RoPE pair,
                          multiplied back after): equal to scaling the query by it (SmoothQuant), cheap on a chip
              hadamard    the 8 channels rotated by an 8 x 8 Hadamard first (rotated back after), cheap
              smooth+had  both
              pre         quantized BEFORE RoPE (KVQuant), RoPE applied when read: costly on a chip
              pre-center  before RoPE, minus the fixed channel mean (TRAINING windows), added back when read
              center      after RoPE, minus that mean rotated to the key's position (a fixed table), added back when
                          read: no RoPE on read, but q . RoPE(mean) is one more dot product per key (estimate)
              center+had  center, then the Hadamard rotation
              sage        after RoPE, minus a fixed vector (the mean of the keys AFTER RoPE on training windows),
                          NOT added back: q . c is the same for every key of a query, softmax ignores it
                          (SageAttention's smooth K with a fixed mean): free on a chip, exact without quantization
    stats also split the key mean by RoPE pair: mean_slow = share of the mean's energy in the pairs that turn less
    than once over the context (theta x seq < 2 pi, the "rotary offset features" of Jonasson 2025)
    The value cache stays as trained. Fixed scales and means come from training windows; losses from validation.
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

import intsim as I
import train as T

BITS = (4, 3, 2)
MODES = ("token", "smooth", "hadamard", "smooth+had", "pre", "pre-center", "center", "center+had", "sage")
KFUN = None          # (block index, k before RoPE, cos, sin) -> key after RoPE as the attention reads it
STATS = None         # list collecting (q after RoPE, k before RoPE, k after RoPE) per block when not None


def forward(self, x, cos, sin):
    """train.Block.forward (SDPA path, as-trained 8-bit cache) with the key replaced by KFUN"""
    b, t, d = x.shape
    h = self.n1(x)
    q = self.q(h).view(b, t, self.heads, -1).transpose(1, 2)
    k, v = (m(h).view(b, t, self.kv_heads, -1).transpose(1, 2) for m in (self.k, self.v))
    qr = T.rope(q, cos, sin)
    kr = T.rope(k, cos, sin) if KFUN is None else KFUN(self.index, k, cos, sin)
    if STATS is not None:
        STATS.append((qr, k, kr))
    if self.kv_heads != self.heads:
        kr, v = (z.repeat_interleave(self.heads // self.kv_heads, dim=1) for z in (kr, v))
    a = F.scaled_dot_product_attention(qr, kr, v, is_causal=True)
    x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
    h = self.n2(x)
    return x + self.w2(self.gate(self.w1(h)) * self.w3(h))


def qtok(x, bits):
    """per token (last axis): 4 bits = train.quant_kv's max / 7 levels, 3 bits = max / 3, 2 bits = quant_kv's
    (c + 0.5) x mean|x| levels"""
    if bits == 3:
        s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 3.0
        return (x / s).round().clamp(-3, 3) * s
    return T.quant_kv(x, bits)


def make_kfun(mode, bits, cal, had):
    def f(i, k, cos, sin):
        if mode in ("pre", "pre-center"):
            mu = cal["mean"][i] if mode == "pre-center" else 0.0
            return T.rope(qtok(k - mu, bits) + mu, cos, sin)
        kr = T.rope(k, cos, sin)
        if mode == "sage":                   # a constant per key head: shifts each query's scores by one number
            return qtok(kr - cal["post"][i], bits)
        if mode.startswith("center"):        # minus the fixed mean, rotated to the key's position (no RoPE on read)
            m = T.rope(cal["mean"][i].expand_as(k), cos, sin)
            z = kr - m
            return (qtok(z @ had, bits) @ had if "had" in mode else qtok(z, bits)) + m
        s = cal["rms"][i] if "smooth" in mode else 1.0
        z = kr / s
        if "had" in mode:
            return (qtok(z @ had, bits) @ had) * s
        return qtok(z, bits) * s
    return f


def windows(data, n, seq, bsz, dev):
    rows = torch.arange(n, device=dev)
    idx = rows[:, None] * seq + torch.arange(seq + 1, device=dev)
    return list(data[idx].long().split(bsz))


@torch.no_grad()
def loss(model, ws):
    total, count = 0.0, 0
    for w in ws:
        logits = model(w[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1),
                                 reduction="sum").item()
        count += w[:, 1:].numel()
    return total / count


@torch.no_grad()
def collect(model, ws):
    """per block: q after RoPE, k before and after RoPE, concatenated over the windows (CPU)"""
    global STATS
    per = None
    for w in ws:
        STATS = []
        model(w[:, :-1])
        got = [tuple(z.float().cpu() for z in s) for s in STATS]
        per = got if per is None else [tuple(torch.cat([a, b]) for a, b in zip(p, g)) for p, g in zip(per, got)]
    STATS = None
    return per


def head_stats(qr, k, kr, heads, cos, sin):
    """per key head: outlier ratio, mean share, token peak; attention entropy per query head"""
    rms = kr.pow(2).mean((0, 2)).sqrt()                          # kv heads x hd
    outlier = (rms.amax(-1) / rms.median(-1).values).tolist()
    mu = k.mean((0, 2))
    mean_share = (mu.pow(2).sum(-1) / k.pow(2).sum(-1).mean((0, 2))).tolist()
    t_all = cos.shape[0]
    slow = torch.acos(cos[1].clamp(-1, 1)) * t_all < 2 * math.pi   # angle per position step = the pair's frequency
    pe = mu.pow(2).view(mu.shape[0], -1, 2).sum(-1)              # kv heads x pairs: the mean's energy per pair
    mean_slow = (pe[:, slow].sum(-1) / pe.sum(-1).clamp(min=1e-12)).tolist()
    peak = (kr.abs().amax(-1) / kr.pow(2).mean(-1).sqrt().clamp(min=1e-8)).mean((0, 2)).tolist()
    kv = kr.shape[1]
    kk = kr.repeat_interleave(heads // kv, dim=1)
    t, hd = qr.shape[2], qr.shape[3]
    mm = T.rope(mu[None, :, None, :].expand(1, kv, t, hd), cos[:t], sin[:t]).repeat_interleave(heads // kv, dim=1)
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    n = causal.sum(-1).clamp(min=2)

    def kvar(z):                     # variance over the visible keys, per query
        z = z.masked_fill(~causal, 0.0)
        m1 = z.sum(-1) / n
        return (z.pow(2).sum(-1) / n - m1.pow(2)).clamp(min=0)
    ent, mshare, cshare = [], [], []
    for b0 in range(0, qr.shape[0], 8):
        s = (qr[b0:b0 + 8] @ kk[b0:b0 + 8].transpose(-1, -2)) / math.sqrt(hd)
        sm = (qr[b0:b0 + 8] @ mm.transpose(-1, -2)) / math.sqrt(hd)
        vs = kvar(s).clamp(min=1e-8)
        vt = vs[..., 1:].mean(-1)                    # pooled over the queries (a per-query ratio blows up on
        mshare.append(kvar(sm)[..., 1:].mean(-1) / vt)          # queries whose scores barely vary)
        cshare.append(kvar(s - sm)[..., 1:].mean(-1) / vt)
        s = s.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), float("-inf"))
        p = s.softmax(-1)
        ent.append(-(p * p.clamp(min=1e-12).log()).sum(-1).mean(-1))   # batch x heads
    ent = torch.cat(ent).mean(0).tolist()
    mshare, cshare = (torch.cat(z).mean(0).tolist() for z in (mshare, cshare))
    return {"outlier": [round(v, 2) for v in outlier], "mean_share": [round(v, 3) for v in mean_share],
            "token_peak": [round(v, 2) for v in peak], "attn_entropy": [round(v, 2) for v in ent],
            "mean_slow": [round(v, 3) for v in mean_slow], "slow_pairs": int(slow.sum()),
            "score_var_mean": [round(v, 3) for v in mshare], "score_var_content": [round(v, 3) for v in cshare]}


def one(run, seed, args, dev):
    global KFUN
    model, a = I.load_model(os.path.join(T.HERE, "runs", f"{run}_s{seed}"), 512 if "v512" in run else 4096, dev)
    assert not a.get("attn_temp") and a.get("k_bits", 8) == 8 and not a.get("kv_recent"), \
        "k_anatomy.py reads models with the plain as-trained cache"
    for i, blk in enumerate(model.blocks):
        blk.index = i
        blk.forward = forward.__get__(blk)
    seq = a["seq"]
    val = windows(T.load_split("val", dev, a.get("vocab", 4096)), args.windows, seq, args.batch, dev)
    cal_w = windows(T.load_split("train", dev, a.get("vocab", 4096)), args.cal_windows, seq, args.batch, dev)
    per = collect(model, cal_w)
    hd = per[0][1].shape[-1]
    cal = {"mean": [], "rms": [], "post": []}
    for _, k, kr in per:
        cal["mean"].append(k.mean((0, 2))[None, :, None, :].to(dev))
        cal["post"].append(kr.mean((0, 2))[None, :, None, :].to(dev))
        r = kr.pow(2).mean((0, 2))                                 # kv heads x hd
        r = r.view(r.shape[0], -1, 2).mean(-1).repeat_interleave(2, -1).sqrt()   # one scale per RoPE pair
        cal["rms"].append(r[None, :, None, :].to(dev))
    had = T.hadamard(hd).to(dev)
    cs = (model.cos.cpu(), model.sin.cpu())
    stats = [head_stats(qr, k, kr, model.blocks[0].heads, *cs) for qr, k, kr in collect(model, val[:2])]
    base = loss(model, val)
    cache = {}
    for mode in MODES:
        for bits in BITS:
            KFUN = make_kfun(mode, bits, cal, had)
            cache[f"{mode} {bits}b"] = round(loss(model, val) - base, 4)
            KFUN = None
    return {"run": f"{run}_s{seed}", "weights": a["weights"], "loss": round(base, 4), "stats": stats,
            "cache_rise": cache}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["fp32_v512d64l5kv4_lr3e-3_20k", "binary_v512d64l5kv4_lr3e-3_20k",
                                                   "ternary_v512d64l5kv4_lr3e-3_20k"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--windows", type=int, default=256, help="validation windows")
    ap.add_argument("--cal-windows", type=int, default=64, help="training windows for the fixed scales / means")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(T.HERE, "results", "v512d64l5kv4_k_anatomy.json"))
    args = ap.parse_args()
    dev = "cuda"
    res = {run: [one(run, s, args, dev) for s in args.seeds] for run in args.runs}

    def ms(v):
        m = sum(v) / len(v)
        return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0)
    summary = {}
    for run, rs in res.items():
        print(f"\n{run} ({len(rs)} seeds), loss {ms([r['loss'] for r in rs])[0]:.4f}")
        L = len(rs[0]["stats"])
        for key in ("outlier", "mean_share", "mean_slow", "token_peak", "attn_entropy", "score_var_mean",
                    "score_var_content"):
            row = [round(ms([sum(r["stats"][l][key]) / len(r["stats"][l][key]) for r in rs])[0], 3)
                   for l in range(L)]
            summary.setdefault(run, {})[key] = row
            print(f"  {key:13s} by layer (mean over heads): {row}")
        print("  cache rise   " + "  ".join(f"{b}b" .rjust(15) for b in BITS))
        for mode in MODES:
            cells = [ms([r["cache_rise"][f"{mode} {b}b"] for r in rs]) for b in BITS]
            summary[run][mode] = [[round(m, 4), round(sd, 4)] for m, sd in cells]
            print(f"  {mode:11s}  " + "  ".join(f"{m:7.4f} ± {sd:.4f}" for m, sd in cells))
    with open(args.out, "w") as f:
        json.dump({"what": __doc__.strip().split("\n")[0], "args": vars(args), "summary": summary, "seeds": res},
                  f, indent=2)
    print("->", args.out)


if __name__ == "__main__":
    main()
