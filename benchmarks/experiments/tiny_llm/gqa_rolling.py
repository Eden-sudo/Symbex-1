"""Bounded rolling GQA reference with absolute (never wrapped) RoPE positions."""
import numpy as np
import torch

import intsim as I
import gqa_intsim as G


def rope_tables(head_dim, positions, device="cpu"):
    """Export-time float computation only; generation consumes integer tables."""
    if head_dim != 64 or type(positions) is not int or positions < 1:
        raise ValueError("expected 64-wide heads and a positive position limit")
    freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
    angle = np.outer(np.arange(positions), freq)
    return tuple(torch.as_tensor(np.round(f(angle) * (1 << 14)).astype(np.int64), device=device)
                 for f in (np.cos, np.sin))


class RollingAttention:
    """Single-stream ring; rotated keys keep their original absolute positions.

    Stored bytes are int8 K/V and int32 maxima, never one copy per query head.
    The reference gathers chronological rows for computation (temporary memory).
    Position-table exhaustion is an error before cache mutation, not a wrap.
    """
    def __init__(self, heads, kv_heads, window, C, S):
        G.validate_heads(heads, kv_heads)
        if not 1 <= heads <= 64 or type(window) is not int or not 1 <= window <= 512:
            raise ValueError("heads must be <=64 and window in 1..512")
        if C.ndim != 2 or C.shape[1] != 32 or S.shape != C.shape or C.shape[0] < 1:
            raise ValueError("expected matching nonempty RoPE tables [positions,32]")
        if C.dtype != torch.int64 or S.dtype != torch.int64 or torch.any(C.abs() > 16384) or torch.any(S.abs() > 16384):
            raise ValueError("RoPE tables must contain Q14 int64 values")
        self.heads, self.kv_heads, self.window = heads, kv_heads, window
        self.C, self.S = C, S
        self.position = 0
        self.k = torch.zeros(kv_heads, window, 64, dtype=torch.int8, device=C.device)
        self.v = torch.zeros_like(self.k)
        self.km = torch.ones(kv_heads, window, dtype=torch.int32, device=C.device)
        self.vm = torch.ones_like(self.km)
        self.exp = torch.as_tensor(I.EXP_TABLE, device=C.device)

    def step(self, q, k, v):
        """Q [heads,64], K/V [kv_heads,64] Q12 int64 -> [heads,64] Q12."""
        pos = self.position
        if pos >= self.C.shape[0]:
            raise ValueError("absolute RoPE position exceeds exported table")
        for x, heads in ((q, self.heads), (k, self.kv_heads), (v, self.kv_heads)):
            if x.shape != (heads, 64) or x.dtype != torch.int64 or x.device != self.C.device or torch.any(x.abs() > (1 << 19)):
                raise ValueError("Q/K/V must be device-local int64 [heads,64], abs <= 2**19")
        qq, qm = I.quant8(I.rope(q, self.C[pos], self.S[pos]))
        kq, km = I.quant8(I.rope(k, self.C[pos], self.S[pos]))
        vq, vm = I.quant8(v)
        slot = pos % self.window
        self.k[:, slot], self.km[:, slot] = kq.to(torch.int8), km.to(torch.int32)
        self.v[:, slot], self.vm[:, slot] = vq.to(torch.int8), vm.to(torch.int32)
        start = max(0, pos + 1 - self.window)
        indices = torch.arange(start, pos + 1, device=q.device) % self.window
        cache = [x[:, indices].long()[None] for x in (self.k, self.km, self.v, self.vm)]
        a = G.quantized_attention(qq[None, :, None], qm[None, :, None], *cache, self.exp)
        self.position += 1
        return a[0, :, 0]


class RollingModel:
    """One-stream generation state. Full recompute matches before eviction only.

    Beyond the window, retained layer states encode their original histories:
    this is sliding-window attention, not recomputation of a cropped prompt.
    """
    def __init__(self, P, window, max_positions):
        self.P = P
        self.position = 0
        self.C, self.S = rope_tables(64, max_positions, P["emb_q"].device)
        self.caches = [RollingAttention(P["heads"], P["kv_heads"], window, self.C, self.S)
                       for _ in P["blocks"]]
        self.sig = torch.as_tensor(I.SIG_TABLE, device=self.C.device)

    def step(self, token):
        if type(token) is not int or not 0 <= token < self.P["emb_q"].shape[0]:
            raise ValueError("token is outside vocabulary")
        if self.position >= self.C.shape[0]:
            raise ValueError("absolute RoPE position exceeds exported table")
        P = self.P
        x = ((P["emb_q"][token] * P["emb_m"][token] + 128) >> 8)[None, None]
        for B, cache in zip(P["blocks"], self.caches):
            h, rms = I.norm_h(x, B["n1"])
            hq, hm = I.quant_in(h, B["q"], rms)
            q, k, v = (I.linear(hq, hm, B[n]).reshape(-1, 64) for n in ("q", "k", "v"))
            a = cache.step(q, k, v).reshape(1, 1, -1)
            x = G._block_tail(x, a, B, self.sig)
        hq, hm, _ = I.norm_quant(x, P["norm_f"])
        self.position += 1
        return (I.mm(hq, P["emb_q"].T) * P["emb_m"])[0, 0], hm[0, 0]
