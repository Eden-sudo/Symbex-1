"""
intsim.py -- the integer version of a model trained by train.py: what the chip will compute, integer by integer.

    python intsim.py runs/binary_d384l8_s42                 # validation loss of the integer model vs the float one
    python intsim.py runs/binary_d384l8_s42 --export        # also writes model.bin + check.bin for the C check

Everything is integer arithmetic (int64), no floating point after the export:
    residual stream     int32, 12 fractional bits (Q12)
    norm + linear       RMSNorm through an integer square root, the input quantized to int8 (absmax per token), 1-bit
                        (or ternary) weights, one int multiplier per output row: out = acc * max * m >> 30
    rotary positions    cos / sin tables in Q14
    attention           q, k, v quantized to int8 per token and head (the KV cache: int8 + one int32 max), softmax
                        with a 2048-entry exp table (uint16), output = sum p * v / sum p
    SwiGLU              sigmoid from a 1024-entry table (Q15)
    embedding / output  int8 per row with one int multiplier per row (the output layer is the same table)
The output is z = (int8 embedding row . int8 final state) * m_row (int64): its argmax is the next token. For the loss
on the PC, z is turned into float logits with the final state's integer max and RMS.
Matrix products run as float64 on the GPU: every product here stays below 2^53, so they are exact (checked).
"""
import argparse
import json
import math
import os
import struct

import numpy as np
import torch

import train as T

QF = 12          # fractional bits of the residual stream
MSH = 30         # shift of the per-row multipliers
EXP_N, EXP_Q = 2048, 7      # exp table: exp(-i / 128), i < 2048
SIG_N, SIG_Q = 1024, 6      # sigmoid table: x in [-8, 8) in steps of 1/64
EXP_TABLE = np.round(65535 * np.exp(-np.arange(EXP_N) / 2 ** EXP_Q)).astype(np.int64)
SIG_TABLE = np.round(32767 / (1 + np.exp(-(np.arange(SIG_N) - SIG_N // 2) / 2 ** SIG_Q))).astype(np.int64)
LIMIT = 2 ** 53


def check(t, bits=62, what=""):
    m = int(t.abs().max()) if t.numel() else 0
    assert m < 2 ** bits, f"{what}: {m} does not fit {bits} bits"
    return t


def mm(a, b):
    """Exact integer matrix product through float64 (every partial sum below 2^53, checked on the result bound)."""
    bound = a.abs().max().double() * b.abs().max().double() * a.shape[-1]
    assert bound < LIMIT, f"matrix product bound {bound:.3g} >= 2^53"
    return torch.matmul(a.double(), b.double()).long()


def rdiv(n, d):
    """Signed division rounded half away from zero (d > 0): the same in C."""
    return torch.sign(n) * ((2 * n.abs() + d) // (2 * d))


def isqrt(n):
    r = torch.sqrt(n.double()).floor().long()
    for _ in range(2):
        r = torch.where(r * r > n, r - 1, r)
        r = torch.where((r + 1) * (r + 1) <= n, r + 1, r)
    return r


# ---- export: float model -> integers -------------------------------------------------------------------------------

def integerize(model, kind):
    """Integer parameters. Linear weights: sign (binary) or {-1,0,1} (ternary) and m = round(mean|w| / 127 * 2^30)."""
    def lin(m):
        w = m.weight.detach().double()
        s = w.abs().mean(dim=1).clamp(min=1e-5)
        q = (w / s[:, None]).round().clamp(-1, 1) if kind == "ternary" else torch.where(w >= 0, 1.0, -1.0)
        return {"w": q.long(), "m": torch.round(s / 127 * 2 ** MSH).long()}

    e = model.emb.weight.detach().double()
    emax = e.abs().amax(dim=1).clamp(min=1e-8)
    P = {"emb_q": torch.round(e / emax[:, None] * 127).long(), "emb_m": torch.round(emax / 127 * 2 ** 20).long(),
         "norm_f": torch.round(model.norm.weight.detach().double() * 2 ** QF).long(), "blocks": []}
    for b in model.blocks:
        P["blocks"].append({"n1": torch.round(b.n1.weight.detach().double() * 2 ** QF).long(),
                            "n2": torch.round(b.n2.weight.detach().double() * 2 ** QF).long(),
                            **{k: lin(getattr(b, k)) for k in ("q", "k", "v", "o", "w1", "w3", "w2")}})
    hd = model.cos.shape[1] * 2
    freq = 1.0 / (10000 ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(np.arange(model.cos.shape[0]), freq)
    P["cos"] = torch.from_numpy(np.round(np.cos(ang) * 2 ** 14).astype(np.int64))
    P["sin"] = torch.from_numpy(np.round(np.sin(ang) * 2 ** 14).astype(np.int64))
    P["heads"] = model.blocks[0].heads
    return P


# ---- integer forward (whole sequences; the C runs it token by token with a KV cache: same integers) -----------------

def norm_quant(x, g):
    """RMSNorm + int8 quantization: q (int8 values), hmax = max |normalized input| in Q12 (>= 1)."""
    ss = check((x * x).sum(-1), 62, "sum of squares")
    rms = isqrt(ss // x.shape[-1]).clamp(min=1)
    h = (x * g + (1 << (QF - 1))) >> QF
    hm = h.abs().amax(-1).clamp(min=1)
    q = rdiv(127 * h, hm[..., None])
    hmax = ((hm << QF) + rms // 2) // rms
    return q, hmax.clamp(min=1), rms


def linear(q, hmax, L):
    acc = mm(q, L["w"].T)
    return check((acc * hmax[..., None] * L["m"] + (1 << (MSH - 1))) >> MSH, 31, "linear output")


def quant8(x):
    m = x.abs().amax(-1).clamp(min=1)
    return rdiv(127 * x, m[..., None]), m


def rope(x, C, S):
    x1, x2 = x[..., ::2], x[..., 1::2]
    r1 = (x1 * C - x2 * S + (1 << 13)) >> 14
    r2 = (x1 * S + x2 * C + (1 << 13)) >> 14
    return torch.stack((r1, r2), -1).flatten(-2)


def attention(q, k, v, heads, C, S, exp_t):
    b, t, d = q.shape
    q, k, v = (z.view(b, t, heads, -1).transpose(1, 2) for z in (q, k, v))       # b, h, t, hd
    q, k = rope(q, C[:t], S[:t]), rope(k, C[:t], S[:t])
    qq, qm = quant8(q)
    kq, km = quant8(k)
    vq, vm = quant8(v)
    u = check(mm(qq, kq.transpose(-1, -2)) * km[..., None, :], 61, "scores")       # b, h, t(query), t(key)
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    u = torch.where(causal, u, torch.iinfo(torch.int64).min)
    umax = u.amax(-1, keepdim=True)
    dist = torch.where(causal, umax - u, 0)
    x = check(dist * qm[..., None], 62, "score distance") // (16129 << 20)
    p = torch.where(causal & (x < EXP_N), exp_t[x.clamp(max=EXP_N - 1)], 0)
    P = p.sum(-1, keepdim=True)
    acc = mm(p * vm[..., None, :], vq)
    a = rdiv(acc, P * 127)
    return a.transpose(1, 2).reshape(b, t, d)


def forward(P, idx):
    dev = idx.device
    exp_t, sig_t = torch.from_numpy(EXP_TABLE).to(dev), torch.from_numpy(SIG_TABLE).to(dev)
    C, S = P["cos"].to(dev), P["sin"].to(dev)
    x = (P["emb_q"][idx] * P["emb_m"][idx][..., None] + 128) >> 8                     # Q12
    for B in P["blocks"]:
        hq, hm, _ = norm_quant(x, B["n1"])
        q, k, v = (linear(hq, hm, B[n]) for n in ("q", "k", "v"))
        a = attention(q, k, v, P["heads"], C, S, exp_t)
        aq, am = quant8(a)
        x = check(x + linear(aq, am, B["o"]), 31, "residual")
        hq, hm, _ = norm_quant(x, B["n2"])
        g1, g3 = linear(hq, hm, B["w1"]), linear(hq, hm, B["w3"])
        sig = sig_t[((g1 >> (QF - SIG_Q)) + SIG_N // 2).clamp(0, SIG_N - 1)]
        silu = (g1 * sig + (1 << 14)) >> 15
        f = check((silu * g3 + (1 << (QF - 1))) >> QF, 31, "SwiGLU")
        fq, fm = quant8(f)
        x = check(x + linear(fq, fm, B["w2"]), 31, "residual")
    hq, hm, rms = norm_quant(x, P["norm_f"])
    z = mm(hq, P["emb_q"].T) * P["emb_m"]
    return z, hm


def float_logits(z, hm):
    """z * hmax / (127 * 2^20 * 2^12): the float logits the integer model stands for (PC loss only)."""
    return z.double() * hm[..., None].double() / (127 * 2.0 ** 20 * 2 ** QF)


def to_device(P, dev):
    if isinstance(P, dict):
        return {k: to_device(v, dev) for k, v in P.items()}
    if isinstance(P, list):
        return [to_device(v, dev) for v in P]
    return P.to(dev) if torch.is_tensor(P) else P


# ---- evaluation ----------------------------------------------------------------------------------------------------

@torch.no_grad()
def int_loss(P, data, seq, n_windows, bsz):
    total, count = 0.0, 0
    usable = min(n_windows, (len(data) - 1) // seq)
    for i in range(0, usable, bsz):
        rows = torch.arange(i, min(i + bsz, usable), device=data.device)
        w = data[rows[:, None] * seq + torch.arange(seq + 1, device=data.device)].long()
        z, hm = forward(P, w[:, :-1])
        logits = float_logits(z, hm)
        total += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1),
                                                   reduction="sum").item()
        count += w[:, 1:].numel()
    return total / count


@torch.no_grad()
def float_argmax_agreement(model, P, data, seq, n_windows, bsz):
    agree, n = 0, 0
    for i in range(0, n_windows, bsz):
        rows = torch.arange(i, i + bsz, device=data.device)
        w = data[rows[:, None] * seq + torch.arange(seq, device=data.device)].long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fa = model(w).argmax(-1)
        z, _ = forward(P, w)
        agree += (z.argmax(-1) == fa).sum().item()
        n += fa.numel()
    return agree / n


@torch.no_grad()
def generate(P, sp, prompt, n_tokens, seq, dev):
    ids = [sp.bos_id()] + sp.encode(prompt)
    for _ in range(n_tokens):
        z, _ = forward(P, torch.tensor([ids[-seq:]], device=dev))
        nxt = int(z[0, -1].argmax())
        if nxt == sp.bos_id():
            break
        ids.append(nxt)
    return ids


# ---- files for the C check -----------------------------------------------------------------------------------------

def pack_bits(w):
    """Rows of +-1 (binary) -> uint32 words, bit i of word j = input 32*j + i, 1 = +1."""
    b = (w.cpu().numpy() > 0).astype(np.uint8)
    out, n_in = b.shape
    b = np.pad(b, ((0, 0), (0, (-n_in) % 32)))
    return np.packbits(b.reshape(out, -1, 32), axis=-1, bitorder="little").view("<u4").reshape(out, -1)


def write_model(P, cfg, path):
    """model.bin: little-endian header of int32 (magic, dims), then the arrays in a fixed order (see tinylm.c)."""
    vocab, dim = P["emb_q"].shape
    blk = P["blocks"][0]
    hidden, layers, seq = blk["w1"]["w"].shape[0], len(P["blocks"]), P["cos"].shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<8i", 0x4D4C5954, 1, vocab, dim, hidden, layers, P["heads"], seq))
        f.write(P["emb_q"].cpu().numpy().astype(np.int8).tobytes())
        f.write(P["emb_m"].cpu().numpy().astype("<i4").tobytes())
        f.write(P["norm_f"].cpu().numpy().astype("<i4").tobytes())
        f.write(P["cos"].cpu().numpy().astype("<i2").tobytes())
        f.write(P["sin"].cpu().numpy().astype("<i2").tobytes())
        f.write(EXP_TABLE.astype("<u2").tobytes())
        f.write(SIG_TABLE.astype("<u2").tobytes())
        for B in P["blocks"]:
            f.write(B["n1"].cpu().numpy().astype("<i4").tobytes())
            f.write(B["n2"].cpu().numpy().astype("<i4").tobytes())
            for n in ("q", "k", "v", "o", "w1", "w3", "w2"):
                f.write(pack_bits(B[n]["w"]).tobytes())
                f.write(B[n]["m"].cpu().numpy().astype("<i4").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="runs/<weights>_<name>_s<seed>")
    ap.add_argument("--windows", type=int, default=1024, help="validation windows of 256 tokens (same as train.py)")
    ap.add_argument("--export", action="store_true", help="write model.bin and check.bin for the C check")
    ap.add_argument("--check-windows", type=int, default=4, help="validation windows in check.bin")
    args = ap.parse_args()
    import sentencepiece as spm
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    ck = torch.load(os.path.join(args.run, "model.pt"))
    a = ck["args"]
    kind = a["weights"]
    sp = spm.SentencePieceProcessor(model_file=os.path.join(T.DATA, "tok4096.model"))
    model = T.TinyLM(sp.get_piece_size(), a["dim"], a["layers"], a["heads"], a["hidden"], a["seq"], kind).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    assert kind in ("binary", "ternary"), "the integer model needs quantized weights"
    P = to_device(integerize(model, kind), dev)
    val = T.load_split("val", dev)

    fl = T.evaluate(model, val, a["seq"], args.windows // 32, 32)
    il = int_loss(P, val, a["seq"], args.windows, 16)
    agree = float_argmax_agreement(model, P, val, a["seq"], 64, 16)
    ids = generate(P, sp, "Once upon a time", 120, a["seq"], dev)
    res = {"run": os.path.basename(os.path.normpath(args.run)), "weights": kind, "val_windows": args.windows,
           "val_loss_float": round(fl, 4), "val_loss_integer": round(il, 4),
           "val_perplexity_float": round(math.exp(fl), 3), "val_perplexity_integer": round(math.exp(il), 3),
           "next_token_agreement_float_vs_integer": round(agree, 4), "sample_integer": sp.decode(ids[1:])}
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.run, "intsim.json"), "w") as f:
        json.dump(res, f, indent=2)

    if args.export:
        assert kind == "binary", "the C check packs 1-bit weights (ternary: to do)"
        write_model(P, a, os.path.join(args.run, "model.bin"))
        w = val[torch.arange(args.check_windows, device=dev)[:, None] * a["seq"]
                + torch.arange(a["seq"], device=dev)].long()
        z, hm = forward(P, w)
        with open(os.path.join(args.run, "check.bin"), "wb") as f:
            f.write(struct.pack("<2i", *w.shape))
            f.write(w.cpu().numpy().astype("<i4").tobytes())
            f.write(z.cpu().numpy().astype("<i8").tobytes())
            f.write(hm.cpu().numpy().astype("<i8").tobytes())
            f.write(struct.pack("<2i", len(ids), 1 + len(sp.encode("Once upon a time"))))
            f.write(np.array(ids, dtype="<i4").tobytes())
        print(f"wrote {args.run}/model.bin ({os.path.getsize(os.path.join(args.run, 'model.bin')) / 1e6:.2f} MB)"
              f" and check.bin")


if __name__ == "__main__":
    main()
