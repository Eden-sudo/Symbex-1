"""hadamard_probe.py -- 1-bit cosine of each linear input with and without a block Hadamard rotation, per block:
    python hadamard_probe.py binary_a1p_k1_s42 binary_d384l8_s42   (run folders under runs/)"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from train import TinyLM, W2_ACTS, hadamard, load_split

def cos1(x):
    s = x.abs().mean(-1, keepdim=True)
    return F.cosine_similarity(torch.where(x >= 0, 1.0, -1.0) * s, x, dim=-1).mean().item()

def rot(x, blk):
    H = hadamard(blk).to(x)
    return (x.view(*x.shape[:-1], -1, blk) @ H).reshape(x.shape)

dev = "cuda"
for run in sys.argv[1:]:
    ck = torch.load(f"runs/{run}/model.pt", map_location=dev, weights_only=False); a = ck["args"]
    m = TinyLM(4096, a["dim"], a["layers"], a["heads"], a["hidden"], a["seq"], a["weights"], a.get("head", "fp32"),
               tuple(a.get("int8_blocks", [])), a.get("acts", 8),
               {"ste_mode": a.get("ste", "identity"), "ste_k": a.get("ste_k", 2.0), "learn_thr": a.get("learn_thr", False)},
               tuple(a.get("a8_inputs", [])), W2_ACTS[a.get("w2_acts", "same")], a.get("ffn_act", "silu"),
               tuple(a.get("hadamard", [])), tuple(a.get("a8_blocks", []))).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    acts = {}
    for bi, b in enumerate(m.blocks):
        for n in ("q", "o", "w1", "w2"):
            def hook(mod, inp, bi=bi, n=n):
                acts[(bi, n)] = inp[0].float().reshape(-1, inp[0].shape[-1])
            getattr(b, n).register_forward_pre_hook(hook)
    val = load_split("val", dev)
    with torch.no_grad():
        idx = torch.arange(16, device=dev)[:, None] * 256 + torch.arange(256, device=dev)
        m(val[idx].long())
    print(run)
    for n in ("q", "o", "w1", "w2"):
        plain = [cos1(acts[(b, n)]) for b in range(len(m.blocks))]
        x0 = acts[(0, n)]; blk = 1024 if x0.shape[-1] % 1024 == 0 else 128
        had = [cos1(rot(acts[(b, n)], blk)) for b in range(len(m.blocks))]
        print(f"  {n:3s} sign      " + " ".join(f"{v:.3f}" for v in plain) + f"  mean {sum(plain)/len(plain):.3f}")
        print(f"  {n:3s} hadamard{blk:<4d}" + " ".join(f"{v:.3f}" for v in had) + f"  mean {sum(had)/len(had):.3f}")
