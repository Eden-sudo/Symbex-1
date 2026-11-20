"""
llm_superweight.py -- README part 5f: is block 0's w2 damage a large output, a weak residual (the embedding), or a
super weight (one weight >> alpha, flattened by sign)? Norms on TRAINING windows, dloss on the validation windows of
part 5d; the test split is never read.

    python llm_superweight.py --seeds 42 43 44 --runs <main>/benchmarks/experiments/tiny_llm/runs --data <data>
    python llm_superweight.py --analyze
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import damage_probe as D
from damage_probe import T

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = (0, 1, 8, 64)


def ratio_to_alpha(w):
    return w.abs() / w.abs().mean(dim=1, keepdim=True)


@torch.no_grad()
def measure_seed(seed, args, dev):
    train, val = T.load_split("train", dev), T.load_split("val", dev)
    fp, a = D.load(args.runs, f"fp32_{args.name}_s{seed}", dev)
    acc, hooks = {}, []

    def add(key, v, n):
        s, c = acc.get(key, (0.0, 0))
        acc[key] = (s + v, c + n)

    def on_linear(key):
        def hook(mod, inp, out):
            x = inp[0].float().reshape(-1, inp[0].shape[-1])
            with torch.autocast("cuda", enabled=False):
                w = mod.weight.float()
                add(f"{key}.out", F.linear(x, w).pow(2).sum().item(), len(x))
                add(f"{key}.err", F.linear(x, w - T.quant_weight(w, "binary")).pow(2).sum().item(), len(x))
                add(f"{key}.chan", x.pow(2).sum(0), len(x))
        return hook

    def on_resid(key):
        def hook(mod, inp):
            x = inp[0].float()
            add(f"{key}.resid", x.pow(2).sum().item(), x.shape[0] * x.shape[1])
        return hook

    for i, b in enumerate(fp.blocks):
        hooks += [b.o.register_forward_hook(on_linear(f"b{i}.o")), b.w2.register_forward_hook(on_linear(f"b{i}.w2")),
                  b.n1.register_forward_pre_hook(on_resid(f"b{i}.o")), b.n2.register_forward_pre_hook(on_resid(f"b{i}.w2"))]
    T.evaluate(fp, train, a["seq"], args.batches, 32)
    for h in hooks:
        h.remove()
    norms = {}
    for i in range(len(fp.blocks)):
        for n in ("o", "w2"):
            k = f"b{i}.{n}"
            norms[k] = {m: round(float(np.sqrt(acc[f"{k}.{m}"][0] / acc[f"{k}.{m}"][1])), 4) for m in ("out", "err", "resid")}
    weights = {}
    for i, b in enumerate(fp.blocks):
        for n in ("w2",) + (("w1", "w3") if i == 0 else ()):
            w = getattr(b, n).weight.float()
            r = ratio_to_alpha(w)
            top = torch.topk(r.flatten(), 5)
            o = {"max": round(r.max().item(), 2), "top5": [round(v, 2) for v in top.values.tolist()],
                 "row_max_median": round(r.max(dim=1).values.median().item(), 2)}
            if n == "w2":
                chan = acc[f"b{i}.w2.chan"][0]
                col = int(top.indices[0]) % w.shape[1]
                o["column"], o["column_energy_rank"] = col, int((chan > chan[col]).sum())
            weights[f"b{i}.{n}"] = o
    base = T.evaluate(fp, val, a["seq"], args.batches, 32)
    lin = fp.blocks[0].w2
    w = lin.weight.float()
    order = torch.argsort(ratio_to_alpha(w).flatten(), descending=True)
    keep = {}
    for k in KEEP:
        q = T.quant_weight(w, "binary").flatten().clone()
        q[order[:k]] = w.flatten()[order[:k]]
        wq = q.view_as(w).to(lin.weight.dtype)
        lin.forward = lambda x, wq=wq: F.linear(x, wq)
        keep[str(k)] = round(T.evaluate(fp, val, a["seq"], args.batches, 32) - base, 4)
        del lin.forward
    print(f"  s{seed}: b0.w2 {norms['b0.w2']} max {weights['b0.w2']['max']} keep {keep}", flush=True)
    del fp
    torch.cuda.empty_cache()
    return {"loss": round(base, 4), "norms": norms, "weights": weights, "b0_w2_keep_dloss": keep}


def analyze(res):
    out = {}
    for s, r in res["seeds"].items():
        nm, wt, kp = r["norms"], r["weights"], r["b0_w2_keep_dloss"]
        nb = len(nm) // 2
        rest = lambda m: [nm[f"b{i}.w2"][m] for i in range(1, nb)]
        r0, o0 = nm["b0.w2"]["resid"], nm["b0.w2"]["out"]
        other_max = max(wt[f"b{i}.w2"]["max"] for i in range(1, nb))
        out[s] = {"b0_w2": nm["b0.w2"], "blocks1_7_resid": [min(rest("resid")), max(rest("resid"))],
                  "blocks1_7_out": [min(rest("out")), max(rest("out"))],
                  "R1_weak_residual": r0 < 0.5 * min(rest("resid")) and 0.5 * min(rest("out")) <= o0 <= 2 * max(rest("out")),
                  "R1_large_output": o0 > 2 * max(rest("out")),
                  "b0_w2_max": wt["b0.w2"]["max"], "other_w2_max": other_max,
                  "R2_super_weight": wt["b0.w2"]["max"] >= 10 and wt["b0.w2"]["max"] > 1.5 * other_max,
                  "keep_dloss": kp, "R3_64_halves": kp["64"] <= 0.5 * kp["0"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs")
    ap.add_argument("--data")
    ap.add_argument("--name", default="d384l8", help="run name between fp32_ and _s<seed> (part 6: d384l8_es / d384l8_nd)")
    ap.add_argument("--batches", type=int, default=20, help="batches of 32 windows (norms: training, dloss: validation)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "llm_superweight.json"))
    ap.add_argument("--analyze", action="store_true", help="only the analysis of --out")
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {"args": {"name": args.name, "batches": args.batches}, "seeds": {}}
    if res["args"].get("name", "d384l8") != args.name:
        raise SystemExit(f"{args.out} holds run {res['args'].get('name', 'd384l8')}: use another --out")
    if not args.analyze:
        args.runs, T.DATA = os.path.expanduser(args.runs), os.path.expanduser(args.data)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in args.seeds:
            if str(seed) not in res["seeds"]:
                res["seeds"][str(seed)] = measure_seed(seed, args, dev)
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                json.dump(res, open(args.out, "w"), indent=1)
    res["analysis"] = analyze(res)
    json.dump(res, open(args.out, "w"), indent=1)
    for s, o in res["analysis"].items():
        print(f"s{s}: {o}")


if __name__ == "__main__":
    main()
