"""
quest_probe.py -- does a Quest page bound (per-channel max / min of the keys of a page) skip pages with T2's 1-bit
keys? Share of far pages (all keys >= 32 back) the bound skips at cut n nats, vs the oracle (pages whose keys are
all below the cut). Validation windows. README "Reading the cache".

    python quest_probe.py runs/binary_attn_k1v2_tau_r32_s42 64
"""
import sys, os, inspect, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import intsim as I, train as T
import sentencepiece as spm
ST = {}
CUTS = (8, 6, 4)
line = "    P = p.sum(-1, keepdim=True)\n"
hook = '''    for _ps in (4, 8, 16):
        _G = t // _ps
        _kp = kq.view(b, heads, _G, _ps, -1)
        _hi, _lo = _kp.amax(3), _kp.amin(3)
        _B2 = mm(qq, (_hi + _lo).transpose(-1, -2)) + mm(qq.abs(), (_hi - _lo).transpose(-1, -2))   # 2 x bound
        _mp = km.view(b, heads, _G, _ps)
        _U = torch.where(_B2 > 0, _B2 * _mp.amax(-1)[..., None, :], _B2 * _mp.amin(-1)[..., None, :]) // 2
        _d = (umax - _U).clamp(min=0)
        _xb = (_d * qm[..., None] // (16129 << 10) * tq.view(1, -1, 1, 1)) >> 26
        _pos = torch.arange(t, device=q.device)
        _valid = ((torch.arange(_G, device=q.device)[None, :] + 1) * _ps - 1 <= _pos[:, None] - 32)   # t, G
        _xmin = torch.where(causal, x, 1 << 40).view(b, heads, t, _G, _ps).amin(-1)                 # best key of the page
        for _n in CUTS:
            _c = _n * 128
            _s = ST.setdefault((_ps, _n), [0, 0, 0])
            _s[0] += (_valid & (_xb >= _c)).sum().item()
            _s[1] += (_valid & (_xmin >= _c)).sum().item()
            _s[2] += (_valid.sum() * b * heads).item()
'''
src = inspect.getsource(I.attention)
assert src.count(line) == 1
src = src.replace(line, hook + line)
ns = dict(I.__dict__); ns.update(ST=ST, CUTS=CUTS); exec(src, ns); I.attention = ns["attention"]
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
sp = spm.SentencePieceProcessor(model_file=os.path.join(T.DATA, "tok4096.model"))
val = T.load_split("val", dev)
run = sys.argv[1]
model, a = I.load_model(run, sp.get_piece_size(), dev)
P = I.to_device(I.integerize(model, a["weights"]), dev)
P["kv"] = {"k_bits": a["k_bits"], "v_bits": a["v_bits"], "k_scale": a["k_scale"], "recent": a["kv_recent"]}
loss = I.int_loss(P, val, a["seq"], int(sys.argv[2]), 16)
print(run, "loss", round(loss, 4))
for (ps, n), (sk, orc, tot) in sorted(ST.items()):
    print(f"page {ps:2d} cut {n}: bound skips {sk / tot:.3f} of far pages, oracle {orc / tot:.3f}")
