"""error_anatomy.py -- which variable hurts the model most, and does the residual sum pile the errors up?

    python error_anatomy.py --run fp32_v512d64l5kv4_lr3e-3_20k --seeds 42 43 44

No training: one quantization source at a time is switched on in a TRAINED fp32 model (as damage_map.py does for
W1A1). Sources: W1A8 (1-bit weights x mean|w|, int8 inputs, as train.py; --kind ternary / int4 for smaller errors) on one linear kind (q, k, v, o, w1, w3, w2)
in every block; the K cache or the V cache at 2 bits (train.py's quant_kv). For each source:
    loss   validation loss rise
    err    relative error of the residual stream after each block, ||x' - x|| / ||x||, against the fp32 model
A1 all seven linears at once vs the sum of the single rises (do the errors add?) and, per block, the joint residual
   error vs the root sum of squares of the single errors (what independent errors would give): the ratio by depth.
A2 one block binarized (its seven linears): the residual error at that block and at every later one; growth = error
   at the last block / error at the binarized block (> 1: the residual stream amplifies it, < 1: later blocks damp it).
The network has not adapted to the change (no retraining): a map of sensitivity, not trained losses.
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

import intsim as I
import train as T

LINEARS = ("q", "k", "v", "o", "w1", "w3", "w2")


@torch.no_grad()
def measure(model, windows):
    """loss over the windows and the residual stream after each block (list of tensors)"""
    outs = []
    hooks = [blk.register_forward_hook(lambda m, i, o: outs.append(o.float().cpu())) for blk in model.blocks]
    total, count, per_block = 0.0, 0, None
    for w in windows:
        outs.clear()
        logits = model(w[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1),
                                 reduction="sum").item()
        count += w[:, 1:].numel()
        per_block = [o.clone() for o in outs] if per_block is None else \
            [torch.cat([p, o]) for p, o in zip(per_block, outs)]
    for h in hooks:
        h.remove()
    return total / count, per_block


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def set_source(model, linears=(), blocks=None, kv=(8, 8, 1.0), kind="binary"):
    for i, blk in enumerate(model.blocks):
        on = blocks is None or i in blocks
        for n in LINEARS:
            getattr(blk, n).kind = kind if on and n in linears else "fp32"
        blk.kv = kv


def one_seed(run, seed, n_windows, bsz, dev, kind, kv_bits):
    model, a = I.load_model(os.path.join(T.HERE, "runs", f"{run}_s{seed}"), a_vocab(run), dev)
    assert a["weights"] == "fp32", "error_anatomy.py starts from a trained fp32 model"
    val = T.load_split("val", dev, a["vocab"])
    seq = a["seq"]
    rows = torch.arange(n_windows, device=dev)
    idx = rows[:, None] * seq + torch.arange(seq + 1, device=dev)
    windows = list(val[idx].long().split(bsz))
    base, ref = measure(model, windows)
    L = len(model.blocks)

    def cell(**kw):
        set_source(model, kind=kind, **kw)
        loss, xs = measure(model, windows)
        set_source(model)
        return {"loss_rise": round(loss - base, 4), "err": [round(rel(x, r), 4) for x, r in zip(xs, ref)]}, xs

    out = {"run": f"{run}_s{seed}", "fp32_loss": round(base, 4), "windows": n_windows, "kind": kind, "kv_bits": kv_bits, "single": {}, "block": {}}
    single_x = {}
    for n in LINEARS:
        out["single"][n], single_x[n] = cell(linears=(n,))
    out["single"][f"K cache {kv_bits}b"], _ = cell(kv=(kv_bits, 8, 1.0))
    out["single"][f"V cache {kv_bits}b"], _ = cell(kv=(8, kv_bits, 1.0))
    joint, jx = cell(linears=LINEARS)
    rss = [math.sqrt(sum(rel(single_x[n][i], ref[i]) ** 2 for n in LINEARS)) for i in range(L)]
    lin_sum = [sum(rel(single_x[n][i], ref[i]) for n in LINEARS) for i in range(L)]
    out["joint"] = {**joint, "sum_of_rises": round(sum(out["single"][n]["loss_rise"] for n in LINEARS), 4),
                    "err_rss": [round(e, 4) for e in rss], "err_linear_sum": [round(e, 4) for e in lin_sum],
                    "ratio_joint_rss": [round(rel(jx[i], ref[i]) / rss[i], 3) for i in range(L)]}
    for b in range(L):
        c, _ = cell(linears=LINEARS, blocks=(b,))
        c["growth"] = round(c["err"][-1] / c["err"][b], 3)
        out["block"][str(b)] = c
    return out


def a_vocab(run):
    return 512 if "v512" in run else 4096


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="fp32_v512d64l5kv4_lr3e-3_20k", help="runs/<run>_s<seed>, a fp32 model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--windows", type=int, default=256, help="validation windows (from the start of the split)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kind", choices=["binary", "ternary", "int4"], default="binary",
                    help="weights of the switched linears (int4: a small error, where errors should add)")
    ap.add_argument("--kv-bits", type=int, choices=[1, 2, 4], default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    res = [one_seed(args.run, s, args.windows, args.batch, dev, args.kind, args.kv_bits) for s in args.seeds]

    def ms(vals):
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        return round(m, 4), round(sd, 4)
    L = len(res[0]["block"])
    print(f"{'source':12s} {'loss rise':>16s}   residual error after block 0..{L - 1} (mean of seeds)")
    summary = {"single": {}, "block": {}}
    for k in res[0]["single"]:
        m, sd = ms([r["single"][k]["loss_rise"] for r in res])
        err = [round(sum(r["single"][k]["err"][i] for r in res) / len(res), 4) for i in range(L)]
        summary["single"][k] = {"loss_rise": [m, sd], "err": err}
        print(f"{k:12s} {m:8.4f} ± {sd:.4f}   " + " ".join(f"{e:.3f}" for e in err))
    j = {k: ms([r["joint"][k] for r in res]) for k in ("loss_rise", "sum_of_rises")}
    ratio = [round(sum(r["joint"]["ratio_joint_rss"][i] for r in res) / len(res), 3) for i in range(L)]
    summary["joint"] = {**j, "ratio_joint_rss": ratio}
    print(f"all 7 linears: rise {j['loss_rise'][0]:.4f} ± {j['loss_rise'][1]:.4f}, sum of single rises "
          f"{j['sum_of_rises'][0]:.4f}; joint / rss error by block: {ratio}")
    for b in range(L):
        m, sd = ms([r["block"][str(b)]["loss_rise"] for r in res])
        g = ms([r["block"][str(b)]["growth"] for r in res])
        err = [round(sum(r["block"][str(b)]["err"][i] for r in res) / len(res), 4) for i in range(L)]
        summary["block"][str(b)] = {"loss_rise": [m, sd], "growth": g, "err": err}
        print(f"block {b} {args.kind}: rise {m:.4f} ± {sd:.4f}, growth {g[0]:.2f} ± {g[1]:.2f}, err " +
              " ".join(f"{e:.3f}" for e in err))
    out = args.out or os.path.join(T.HERE, "results", f"{args.run.split('_', 1)[1].split('_')[0]}_error_anatomy_{args.kind}.json")
    with open(out, "w") as f:
        json.dump({"what": __doc__.strip().split("\n")[0], "args": vars(args), "summary": summary, "seeds": res},
                  f, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
