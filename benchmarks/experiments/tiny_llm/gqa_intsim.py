"""Additive 64-wide GQA integer reference; no change to the firmware/export ABI."""
import torch

import intsim as I


def validate_heads(heads, kv_heads):
    if type(heads) is not int or type(kv_heads) is not int or heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("heads must be a positive multiple of kv_heads")


def integerize(model, kind):
    """Export in-memory parameters only; the legacy model.bin writer is not GQA-aware."""
    if kind not in ("binary", "ternary"):
        raise ValueError("integer simulation requires binary or ternary weights")
    for block in model.blocks:
        validate_heads(block.heads, block.kv_heads)
        if block.head_dim != 64:
            raise ValueError("integer attention requires 64-wide heads")
    params = I.integerize(model, kind)
    params["kv_heads"] = model.blocks[0].kv_heads
    return params


def quantized_attention(qq, qm, kq, km, vq, vm, exp_t, causal=None):
    """b,h,t,64 queries and b,kv,s,64 cache; return b,h,t,64 Q12.

    Adjacent query heads share one KV head. No expanded KV cache is materialized.
    The C reference uses the same bounded domain: <=512 keys, scales <=2**20.
    """
    heads, kv_heads = qq.shape[1], kq.shape[1]
    validate_heads(heads, kv_heads)
    if qq.shape[-1] != 64 or kq.shape[-1] != 64 or kq.shape != vq.shape:
        raise ValueError("expected matching 64-wide K/V")
    if not 0 < kq.shape[2] <= 512:
        raise ValueError("key length must be in 1..512")
    for values in (qq, kq, vq):
        if values.dtype != torch.int64 or torch.any(values.abs() > 127):
            raise ValueError("quantized values must be int64 in -127..127")
    for scale in (qm, km, vm):
        if scale.dtype != torch.int64 or torch.any((scale < 1) | (scale > 2**20)):
            raise ValueError("scales must be int64 in 1..2**20")
    outputs = []
    for h in range(heads):
        kh = h // (heads // kv_heads)
        u = I.mm(qq[:, h], kq[:, kh].transpose(-1, -2)) * km[:, kh, None, :]
        if causal is not None:
            u = torch.where(causal, u, torch.iinfo(torch.int64).min)
        dist = u.amax(-1, keepdim=True) - u
        if causal is not None:
            dist = torch.where(causal, dist, 0)
        index = I.check(dist * qm[:, h, :, None], 62, "score distance") // (16129 << 20)
        p = torch.where(index < I.EXP_N, exp_t[index.clamp(max=I.EXP_N - 1)], 0)
        if causal is not None:
            p = torch.where(causal, p, 0)
        acc = I.mm(p * vm[:, kh, None, :], vq[:, kh])
        outputs.append(I.rdiv(acc, p.sum(-1, keepdim=True) * 127))
    return torch.stack(outputs, dim=1)


def attention(q, k, v, heads, kv_heads, C, S, exp_t):
    validate_heads(heads, kv_heads)
    b, t, d = q.shape
    if d != heads * 64 or k.shape != (b, t, kv_heads * 64) or v.shape != k.shape:
        raise ValueError("expected Q [b,t,heads*64], K/V [b,t,kv_heads*64]")
    if t < 1 or C.shape[0] < t or S.shape != C.shape:
        raise ValueError("nonempty sequence requires enough RoPE rows")
    q = q.reshape(b, t, heads, 64).transpose(1, 2)
    k, v = (z.reshape(b, t, kv_heads, 64).transpose(1, 2) for z in (k, v))
    qq, qm = I.quant8(I.rope(q, C[:t], S[:t]))
    kq, km = I.quant8(I.rope(k, C[:t], S[:t]))
    vq, vm = I.quant8(v)
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    a = quantized_attention(qq, qm, kq, km, vq, vm, exp_t, causal)
    return a.transpose(1, 2).reshape(b, t, d)


def _block_tail(x, a, B, sig_t):
    aq, am = I.quant_in(a, B["o"])
    x = I.check(x + I.linear(aq, am, B["o"]), 31, "residual")
    h, rms = I.norm_h(x, B["n2"])
    hq, hm = I.quant_in(h, B["w1"], rms)
    g1, g3 = I.linear(hq, hm, B["w1"]), I.linear(hq, hm, B["w3"])
    sig = sig_t[((g1 >> (I.QF - I.SIG_Q)) + I.SIG_N // 2).clamp(0, I.SIG_N - 1)]
    silu = (g1 * sig + (1 << 14)) >> 15
    f = I.check((silu * g3 + (1 << (I.QF - 1))) >> I.QF, 31, "SwiGLU")
    fq, fm = I.quant_in(f, B["w2"])
    return I.check(x + I.linear(fq, fm, B["w2"]), 31, "residual")


def forward(P, idx):
    """Whole-sequence GQA forward, with the legacy intsim score/scale return type."""
    exp_t = torch.as_tensor(I.EXP_TABLE, device=idx.device)
    sig_t = torch.as_tensor(I.SIG_TABLE, device=idx.device)
    C, S = P["cos"].to(idx.device), P["sin"].to(idx.device)
    x = (P["emb_q"][idx] * P["emb_m"][idx][..., None] + 128) >> 8
    for B in P["blocks"]:
        h, rms = I.norm_h(x, B["n1"])
        hq, hm = I.quant_in(h, B["q"], rms)
        q, k, v = (I.linear(hq, hm, B[n]) for n in ("q", "k", "v"))
        a = attention(q, k, v, P["heads"], P["kv_heads"], C, S, exp_t)
        x = _block_tail(x, a, B, sig_t)
    hq, hm, _ = I.norm_quant(x, P["norm_f"])
    return I.mm(hq, P["emb_q"].T) * P["emb_m"], hm
