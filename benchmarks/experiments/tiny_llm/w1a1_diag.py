"""
w1a1_diag.py -- why does the W1A1 model (train.py --acts 1) train so badly? Step 1 of the plan in LOGBOOK.md: no
training, only measurements on the saved models, per block and per input of the linear layers:

    python w1a1_diag.py                       # W1A1 and W1A8 models, seeds 42 43 44
    python w1a1_diag.py --seeds 42 --windows 4

Inputs measured (q, k, v share one input; w1, w3 share one):
    attn_in   RMSNorm output into q / k / v
    o_in      attention output into o
    ffn_in    RMSNorm output into w1 / w3
    w2_in     silu(w1 h) * w3 h into w2

Per input, on fixed validation windows (the first --windows x 256 tokens of the split):
    pos        share of values >= 0 (what sign() keeps as +1)
    chan_bias  mean over channels of |2 p_c - 1|, p_c the share >= 0 of channel c (0: every channel balanced,
               1: every channel always one sign, so its bit carries nothing)
    cos_sign   cosine between sign(x) x mean|x| (what train.py uses) and x, mean over tokens
    cos_thr    the same with a per-channel threshold and two levels per channel (median and mean distance, fit on
               these same windows: an UPPER BOUND of what a learned threshold such as ReActNet's RSign could keep)
    out1       share of |x| > 1 (where a clipped straight-through gradient would be zero)
And on one training batch: the gradient norm of each layer's weights relative to the weights' norm.

The W1A8 models are measured too: their cos_sign says how much binarizing their inputs would lose (the start point
of a two-step recipe). Writes results/w1a1_diag.json.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from train import DATA, HERE, QLinear, TinyLM, batch, load_split

POINTS = {"q": "attn_in", "o": "o_in", "w1": "ffn_in", "w2": "w2_in"}


def stats(x):
    """x: [tokens, channels] float32. Returns the per-input measures (see the docstring)."""
    pos_c = (x >= 0).float().mean(0)
    s = x.abs().mean(-1, keepdim=True)
    cos_sign = F.cosine_similarity(torch.where(x >= 0, 1.0, -1.0) * s, x, dim=-1).mean()
    med = x.median(0).values
    d = x - med
    lvl = d.abs().mean(0)
    rec = med + torch.where(d >= 0, 1.0, -1.0) * lvl
    cos_thr = F.cosine_similarity(rec, x, dim=-1).mean()
    return {"pos": (x >= 0).float().mean().item(), "chan_bias": (2 * pos_c - 1).abs().mean().item(),
            "cos_sign": cos_sign.item(), "cos_thr": cos_thr.item(), "out1": (x.abs() > 1).float().mean().item()}


def measure(path, val, train, windows, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    model = TinyLM(4096, a["dim"], a["layers"], a["heads"], a["hidden"], a["seq"], a["weights"], a.get("head", "fp32"),
                   tuple(a.get("int8_blocks", [])), a.get("acts", 8)).to(device)
    model.load_state_dict(ck["model"])
    seq = a["seq"]
    acts = {}
    hooks = []
    for bi, blk in enumerate(model.blocks):
        for name, point in POINTS.items():
            def hook(mod, inp, bi=bi, point=point):
                acts.setdefault((bi, point), []).append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))
            hooks.append(getattr(blk, name).register_forward_pre_hook(hook))
    model.eval()
    with torch.no_grad():
        idx = torch.arange(windows, device=device)[:, None] * seq + torch.arange(seq, device=device)
        model(val[idx].long())
    for h in hooks:
        h.remove()
    per = {}
    for (bi, point), xs in acts.items():
        per.setdefault(point, {})[bi] = stats(torch.cat(xs))
    # one training batch: relative gradient norm per linear layer
    model.train()
    gen = torch.Generator(device=device).manual_seed(0)
    x, y = batch(train, 32, seq, gen)
    loss = F.cross_entropy(model(x).float().reshape(-1, 4096), y.reshape(-1))
    loss.backward()
    grad = {}
    for bi, blk in enumerate(model.blocks):
        for name in ("q", "k", "v", "o", "w1", "w3", "w2"):
            w = getattr(blk, name).weight
            grad.setdefault(name, {})[bi] = (w.grad.norm() / w.norm()).item()
    return {"step": ck["step"], "acts": a.get("acts", 8), "train_batch_loss": loss.item(), "inputs": per,
            "rel_grad": grad}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--windows", type=int, default=16, help="validation windows of 256 tokens")
    args = ap.parse_args()
    device = "cuda"
    val, train = load_split("val", device), load_split("train", device)
    out = {"windows": args.windows, "models": {}}
    for kind, name in (("w1a1", "binary_d384l8_a1"), ("w1a8", "binary_d384l8")):
        for s in args.seeds:
            path = os.path.join(HERE, "runs", f"{name}_s{s}", "model.pt")
            out["models"][f"{kind}_s{s}"] = measure(path, val, train, args.windows, device)
            print(f"measured {kind} seed {s}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "w1a1_diag.json"), "w") as f:
        json.dump(out, f, indent=1)

    def mean(kind, point, key, blocks=None):
        v = [m["inputs"][point][b][key] for k, m in out["models"].items() if k.startswith(kind)
             for b in m["inputs"][point] if blocks is None or b in blocks]
        return sum(v) / len(v)

    print("\nMean over blocks and seeds (W1A1 | W1A8)")
    print(f"{'input':8s} {'pos':>13s} {'chan_bias':>13s} {'cos_sign':>13s} {'cos_thr':>13s} {'out1':>13s}")
    for point in POINTS.values():
        cells = [f"{mean('w1a1', point, k):.3f} | {mean('w1a8', point, k):.3f}"
                 for k in ("pos", "chan_bias", "cos_sign", "cos_thr", "out1")]
        print(f"{point:8s} " + " ".join(f"{c:>13s}" for c in cells))
    print("\ncos_sign per block, W1A1 (mean of seeds)")
    nb = len(next(iter(out["models"].values()))["inputs"]["attn_in"])
    for point in POINTS.values():
        print(f"{point:8s} " + " ".join(f"{mean('w1a1', point, 'cos_sign', [b]):.3f}" for b in range(nb)))
    print("\nRelative gradient norm (x1e-3), mean of seeds, blocks 0..7")
    for kind in ("w1a1", "w1a8"):
        for name in ("q", "o", "w1", "w2"):
            v = [[m["rel_grad"][name][b] for k, m in out["models"].items() if k.startswith(kind)] for b in range(nb)]
            print(f"{kind} {name:3s} " + " ".join(f"{1e3 * sum(x) / len(x):6.2f}" for x in v))


if __name__ == "__main__":
    main()
