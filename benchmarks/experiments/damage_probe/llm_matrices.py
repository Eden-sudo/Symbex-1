"""
llm_matrices.py -- README part 5, check 2: the one-shot damage of every single matrix of tiny_llm (fp32 model, only
that matrix 1-bit), to separate fan-in from size (w1 / w3 / w2 have the same number of weights, w2's fan-in is 1024
vs 384) and the number of matrices from their role (q, k, v, o are all 384 x 384). Same validation batches as
llm_layers.py; the test split is never read.

    python llm_matrices.py --seeds 42 43 44 --runs <main>/benchmarks/experiments/tiny_llm/runs --data <tiny_llm data>
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import damage_probe as D
from damage_probe import T
from llm_layers import GROUPS

HERE = os.path.dirname(os.path.abspath(__file__))
MATRICES = ("q", "k", "v", "o", "w1", "w3", "w2")


def only(lin, part):
    """make one QLinear quantize only its weights (1-bit x alpha, input in floating point: part "w") or only its
    input (int8 absmax per token, weights in floating point: part "a"); returns the undo"""
    if part == "w":
        lin.forward = lambda x: F.linear(x, T.quant_weight(lin.weight, "binary"))
    else:
        lin.forward = lambda x: F.linear(T.quant_act8(x), lin.weight)
    return lambda: delattr(lin, "forward")


def measure_seed(seed, args, dev):
    val = T.load_split("val", dev)
    fp, a = D.load(args.runs, f"fp32_d384l8_s{seed}", dev)
    base = T.evaluate(fp, val, a["seq"], args.batches, 32)
    rows = {}
    for i, b in enumerate(fp.blocks):
        for n in MATRICES:
            lin = getattr(b, n)
            if args.part == "wa":
                old = D.apply([(lin, "kind", "binary")])
            else:
                assert lin.kind == "fp32"
                undo = only(lin, args.part)
            rows[f"b{i}.{n}"] = round(T.evaluate(fp, val, a["seq"], args.batches, 32) - base, 4)
            D.apply(old) if args.part == "wa" else undo()
        print(f"  s{seed}: block {i} " + " ".join(f"{n} {rows[f'b{i}.{n}']:+.4f}" for n in MATRICES), flush=True)
    del fp
    torch.cuda.empty_cache()
    return {"loss": round(base, 4), "matrices": rows}


def analyze(res, layers):
    out = {}
    for s, r in res["seeds"].items():
        m = r["matrices"]
        nb = len({k.split(".")[0] for k in m})
        mean = {n: float(np.mean([m[f"b{i}.{n}"] for i in range(nb)])) for n in MATRICES}
        units = layers["seeds"][s]["before"]["units"]
        summed = [sum(m[f"{u.split('.')[0]}.{n}"] for n in GROUPS[u.split(".")[1]]) for u in units]
        out[s] = {"base_equals_llm_layers": r["loss"] == layers["seeds"][s]["before"]["loss"],
                  "mean_by_matrix": {k: round(v, 4) for k, v in mean.items()},
                  "w2_below_w1w3_blocks": f"{sum(m[f'b{i}.w2'] < (m[f'b{i}.w1'] + m[f'b{i}.w3']) / 2 for i in range(nb))}/{nb}",
                  "w2_over_w1w3": round(mean["w2"] / ((mean["w1"] + mean["w3"]) / 2), 3) if mean["w1"] + mean["w3"] else None,
                  "w2_above_w1w3_blocks": f"{sum(m[f'b{i}.w2'] > (m[f'b{i}.w1'] + m[f'b{i}.w3']) / 2 for i in range(nb))}/{nb}",
                  "qkv_mean_over_o": round(float(np.mean([mean['q'], mean['k'], mean['v']]) / mean["o"]), 3),
                  "unit_vs_sum_of_matrices": round(float(spearmanr(summed, [units[u]["dloss"] for u in units]).statistic), 3),
                  "unit_over_sum_of_matrices": round(float(sum(units[u]["dloss"] for u in units) / sum(summed)), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs")
    ap.add_argument("--data")
    ap.add_argument("--batches", type=int, default=20, help="validation batches of 32 windows (as llm_layers.py)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "llm_matrices.json"))
    ap.add_argument("--analyze", action="store_true", help="only the analysis of --out")
    ap.add_argument("--part", choices=("wa", "w", "a"), default="wa",
                    help="1-bit weights x alpha and int8 input (wa, part 5), only the weights (w) or only the input (a)")
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {"args": {"batches": args.batches, "part": args.part},
                                                                      "seeds": {}}
    if res["args"].get("part", "wa") != args.part:
        raise SystemExit(f"{args.out} holds part {res['args'].get('part', 'wa')}: use another --out")
    if not args.analyze:
        args.runs, T.DATA = os.path.expanduser(args.runs), os.path.expanduser(args.data)
        for s in args.seeds:
            torch.manual_seed(s)
            res["seeds"][str(s)] = measure_seed(s, args, "cuda")
            json.dump(res, open(args.out, "w"), indent=1)
    layers = json.load(open(os.path.join(HERE, "results", "llm_layers.json")))
    res["analysis"] = analyze({"seeds": {s: r for s, r in res["seeds"].items() if s in layers["seeds"]}}, layers)
    json.dump(res, open(args.out, "w"), indent=1)
    for s, o in res["analysis"].items():
        print(f"s{s}: {o}")


if __name__ == "__main__":
    main()
