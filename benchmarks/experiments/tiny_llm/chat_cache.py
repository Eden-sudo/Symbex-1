"""Single-conversation integer KV reuse; immutable model parameters required."""
import torch
import intsim as I
from chat_memory import retain


def retention_plan(system, turns, suffix, cached_ids=(), reserve=1, context=256):
    """Return (retained IDs, dropped turns, recompute_required).

    cached_ids are tokens actually forwarded, not merely sampled/emitted.
    Eviction always requests replay, even if repeated token values match.
    """
    ids, dropped = retain(system, turns, suffix, reserve, context)
    cached = list(cached_ids)
    recompute = bool(dropped) or ids[:len(cached)] != cached
    return ids, dropped, recompute


class PrefixCache:
    """One batch, at most 256 positions, fixed P and device for its lifetime.

    update returns (last-token scores, last-token scale, number of steps run).
    Pass retention_plan's recompute_required as force_recompute. Scores/scales
    have the same [1,1,vocab]/[1,1] shape as intsim.forward's final position.
    This uses the reference's fixed 64-wide-head attention scaling.
    """
    def __init__(self, parameters):
        self.P = parameters
        self.device = parameters["emb_q"].device
        self.context = min(256, len(parameters["cos"]))
        self.exp = torch.from_numpy(I.EXP_TABLE).to(self.device)
        self.sig = torch.from_numpy(I.SIG_TABLE).to(self.device)
        self.reset()

    def reset(self):
        self.ids = []
        self.kv = [None] * len(self.P["blocks"])
        self.last = None

    @torch.no_grad()
    def update(self, ids, force_recompute=False):
        ids = list(ids)
        if (not 1 <= len(ids) <= self.context or
                any(type(t) is not int or not 0 <= t < len(self.P["emb_q"]) for t in ids)):
            raise ValueError("invalid prompt length or token ID")
        if force_recompute or ids[:len(self.ids)] != self.ids:
            self.reset()
        steps = len(ids) - len(self.ids)
        try:
            for token in ids[len(self.ids):]:
                self.last = self._step(token)
                self.ids.append(token)
        except Exception:
            self.reset()  # Never reuse a partially updated layer cache.
            raise
        return self.last[0].clone(), self.last[1].clone(), steps

    def _attention(self, q, k, v, layer, pos):
        heads = self.P["heads"]
        q, k, v = (z.view(1, 1, heads, -1).transpose(1, 2) for z in (q, k, v))
        c, s = self.P["cos"][pos:pos+1], self.P["sin"][pos:pos+1]
        qq, qm = I.quant8(I.rope(q, c, s))
        kq, km = I.quant8(I.rope(k, c, s))
        vq, vm = I.quant8(v)
        previous = self.kv[layer]
        if previous is not None:
            kq = torch.cat((previous[0], kq), dim=2)
            km = torch.cat((previous[1], km), dim=2)
            vq = torch.cat((previous[2], vq), dim=2)
            vm = torch.cat((previous[3], vm), dim=2)
        self.kv[layer] = kq, km, vq, vm
        u = I.check(I.mm(qq, kq.transpose(-1, -2)) * km[..., None, :], 61, "scores")
        distance = u.amax(-1, keepdim=True) - u
        index = I.check(distance * qm[..., None], 62, "score distance") // (16129 << 20)
        p = torch.where(index < I.EXP_N, self.exp[index.clamp(max=I.EXP_N-1)], 0)
        a = I.rdiv(I.mm(p * vm[..., None, :], vq), p.sum(-1, keepdim=True) * 127)
        return a.transpose(1, 2).reshape(1, 1, -1)

    def _step(self, token):
        P, pos = self.P, len(self.ids)
        idx = torch.tensor([[token]], device=self.device)
        x = (P["emb_q"][idx] * P["emb_m"][idx][..., None] + 128) >> 8
        for layer, B in enumerate(P["blocks"]):
            h, rms = I.norm_h(x, B["n1"])
            hq, hm = I.quant_in(h, B["q"], rms)
            q, k, v = (I.linear(hq, hm, B[n]) for n in ("q", "k", "v"))
            a = self._attention(q, k, v, layer, pos)
            aq, am = I.quant_in(a, B["o"])
            x = I.check(x + I.linear(aq, am, B["o"]), 31, "residual")
            h, rms = I.norm_h(x, B["n2"])
            hq, hm = I.quant_in(h, B["w1"], rms)
            g1, g3 = I.linear(hq, hm, B["w1"]), I.linear(hq, hm, B["w3"])
            sig = self.sig[((g1 >> (I.QF-I.SIG_Q)) + I.SIG_N//2).clamp(0, I.SIG_N-1)]
            silu = (g1 * sig + (1 << 14)) >> 15
            f = I.check((silu * g3 + (1 << (I.QF-1))) >> I.QF, 31, "SwiGLU")
            fq, fm = I.quant_in(f, B["w2"])
            x = I.check(x + I.linear(fq, fm, B["w2"]), 31, "residual")
        hq, hm, _ = I.norm_quant(x, P["norm_f"])
        return I.mm(hq, P["emb_q"].T) * P["emb_m"], hm
