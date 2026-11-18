"""
llm_columns.py -- README part 5e: is w2's excess the 1-bit error on the weights that multiply giant input channels?
Per matrix of the fp32 tiny_llm: the input-weighted relative error of sign(w) x alpha, tr(E C E^T) / tr(W C W^T), with
C = E[x x^T] measured on TRAINING windows, against the plain ||E||^2 / ||W||^2, and how typical the weights on the 8
largest input channels are. Truth: part 5d's weights-only dloss (results/llm_matrices_w.json). No training; the test
split is never read.

    python llm_columns.py --seeds 42 43 44 --runs <main>/benchmarks/experiments/tiny_llm/runs --data <tiny_llm data>
    python llm_columns.py --analyze
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

import damage_probe as D
from damage_probe import T
from llm_matrices import MATRICES

HERE = os.path.dirname(os.path.abspath(__file__))
FANIN = {"q": 384, "k": 384, "v": 384, "o": 384, "w1": 384, "w3": 384, "w2": 1024}
TOP = 8


@torch.no_grad()
def measure_seed(seed, args, dev):
    train = T.load_split("train", dev)
    fp, a = D.load(args.runs, f"fp32_d384l8_s{seed}", dev)
    cov, tokens, resid, hooks = {}, {}, {}, []

    def grab(key):
        def hook(mod, inp):
            x = inp[0].float().reshape(-1, inp[0].shape[-1])
            cov[key] = cov.get(key, 0) + x.T @ x
            tokens[key] = tokens.get(key, 0) + len(x)
        return hook

    def grab_resid(key):
        def hook(mod, inp):
            resid[key] = resid.get(key, 0.0) + inp[0].float().pow(2).sum().item()
        return hook

    for i, b in enumerate(fp.blocks):
        for n in MATRICES:
            hooks.append(getattr(b, n).register_forward_pre_hook(grab(f"b{i}.{n}")))
        hooks.append(b.n1.register_forward_pre_hook(grab_resid(f"b{i}.o")))    # the residual o is added to
        hooks.append(b.n2.register_forward_pre_hook(grab_resid(f"b{i}.w2")))   # the residual w2 is added to
    T.evaluate(fp, train, a["seq"], args.batches, 32)
    for h in hooks:
        h.remove()
    rows = {}
    for i, b in enumerate(fp.blocks):
        for n in MATRICES:
            k = f"b{i}.{n}"
            w = getattr(b, n).weight.float()
            e = w - T.quant_weight(w, "binary")
            c = cov[k] / tokens[k]
            alpha = w.abs().mean(dim=1, keepdim=True)
            top = torch.topk(c.diagonal(), TOP).indices
            r = {"rel_c": (e @ c * e).sum().item() / (w @ c * w).sum().item(),
                 "rel_i": (e.pow(2).sum() / w.pow(2).sum()).item(),
                 "top8_w_over_alpha": (w[:, top].abs() / alpha).mean().item(),
                 "top8_share": (c.diagonal()[top].sum() / c.diagonal().sum()).item()}
            if k in resid:
                r["err_over_residual"] = (e @ c * e).sum().item() / (resid[k] / tokens[k])
            rows[k] = {m: round(v, 5) for m, v in r.items()}
        print(f"  s{seed}: block {i} rel_c " + " ".join(f"{n} {rows[f'b{i}.{n}']['rel_c']:.3f}" for n in MATRICES),
              flush=True)
    del fp
    torch.cuda.empty_cache()
    return {"matrices": rows}


def analyze(res, truth):
    out = {}
    for s, r in res["seeds"].items():
        m, d = r["matrices"], truth["seeds"][s]["matrices"]
        nb = len({k.split(".")[0] for k in m})
        keys = list(m)
        dl = [d[k] for k in keys]

        def ratio(metric, i=None):
            blocks = range(nb) if i is None else [i]
            w2 = np.mean([m[f"b{j}.w2"][metric] for j in blocks])
            return w2 / np.mean([(m[f"b{j}.w1"][metric] + m[f"b{j}.w3"][metric]) / 2 for j in blocks])

        def mean_by(metric):
            return {n: round(float(np.mean([m[f"b{j}.{n}"][metric] for j in range(nb)])), 4) for n in MATRICES}

        out[s] = {"p1_w2_above_w1w3_rel_c": f"{sum(ratio('rel_c', i) > 1 for i in range(nb))}/{nb}",
                  "rho_rel_c": round(float(spearmanr([m[k]['rel_c'] for k in keys], dl).statistic), 3),
                  "rho_rel_i": round(float(spearmanr([m[k]['rel_i'] for k in keys], dl).statistic), 3),
                  "rho_fanin": round(float(spearmanr([FANIN[k.split('.')[1]] for k in keys], dl).statistic), 3),
                  "w2_over_w1w3_rel_c": round(float(ratio("rel_c")), 3),
                  "w2_over_w1w3_rel_i": round(float(ratio("rel_i")), 3),
                  "mean_rel_c": mean_by("rel_c"), "mean_rel_i": mean_by("rel_i"),
                  "mean_top8_w_over_alpha": mean_by("top8_w_over_alpha"), "mean_top8_share": mean_by("top8_share"),
                  "err_over_residual": {n: round(float(np.mean([m[f"b{j}.{n}"]["err_over_residual"] for j in range(nb)])), 5)
                                        for n in ("o", "w2")}}
        o = out[s]
        o["p2"] = o["rho_rel_c"] >= o["rho_fanin"] + 0.1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs")
    ap.add_argument("--data")
    ap.add_argument("--batches", type=int, default=20, help="training batches of 32 windows for C")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "llm_columns.json"))
    ap.add_argument("--truth", default=os.path.join(HERE, "results", "llm_matrices_w.json"))
    ap.add_argument("--analyze", action="store_true", help="only the analysis of --out")
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {"args": {"batches": args.batches}, "seeds": {}}
    if not args.analyze:
        args.runs, T.DATA = os.path.expanduser(args.runs), os.path.expanduser(args.data)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in args.seeds:
            if str(seed) not in res["seeds"]:
                res["seeds"][str(seed)] = measure_seed(seed, args, dev)
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                json.dump(res, open(args.out, "w"), indent=1)
    res["analysis"] = analyze(res, json.load(open(args.truth)))
    json.dump(res, open(args.out, "w"), indent=1)
    for s, o in res["analysis"].items():
        print(f"s{s}: {o}")


if __name__ == "__main__":
    main()
