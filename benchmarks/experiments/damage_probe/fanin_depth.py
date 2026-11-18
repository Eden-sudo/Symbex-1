"""
fanin_depth.py -- README part 5: does the one-shot damage of a layer follow its fan-in (count resolution) and its
distance to the output? Reads results/layer_table.json (ResNet-8, part 2) and results/llm_layers.json (tiny_llm), no
model is run. Fan-ins are the checkpoints' shapes (README part 5).

    python fanin_depth.py                    # prints the tables, writes results/fanin_depth.json
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RESNET = {"stem": (216, 7), "b0": (144, 6), "b1": (144, 5), "b2": (144, 4), "ds2": (16, 4), "b3": (288, 3),
          "b4": (288, 2), "ds4": (32, 2), "b5": (576, 1), "fc": (4096, 0)}        # layer: (fan-in, layers left)
LLM_FANIN = {"qkv": 384, "o": 384, "w13": 384, "w2": 1024}
LLM_BLOCKS = 8


def rho(a, b):
    return round(float(spearmanr(a, b).statistic), 3)


def partial_rho(a, b, z):
    """Spearman of a and b with z held fixed: Pearson of the residuals of rank(a), rank(b) on rank(z)"""
    ra, rb, rz = (np.argsort(np.argsort(v)).astype(float) for v in (a, b, z))
    res = [r - np.polyval(np.polyfit(rz, r, 1), rz) for r in (ra, rb)]
    return round(float(np.corrcoef(*res)[0, 1]), 3)


def loo_r2(drop, fanin, left):
    """leave-one-out R^2 of drop = a + b log(fan-in) + c left"""
    X = np.column_stack([np.ones(len(drop)), np.log(fanin), left]).astype(float)
    y = np.asarray(drop, dtype=float)
    pred = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        pred[i] = X[i] @ coef
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return round(float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3), np.round(coef, 4).tolist()


def resnet(table):
    out = {}
    names = list(RESNET)
    fanin = [RESNET[k][0] for k in names]
    left = [RESNET[k][1] for k in names]
    for s, row in table["seeds"].items():
        o = {}
        drop = {k: {"wa": v["w"], **v} for k, v in row["drop"].items()}   # fc has weights only: its wa is its w
        for part in ("wa", "w", "a"):
            ks = [k for k in names if part in drop[k]]
            d = [drop[k][part] for k in ks]
            o[part] = {"fanin": rho(d, [RESNET[k][0] for k in ks]), "left": rho(d, [RESNET[k][1] for k in ks])}
        d = [drop[k]["wa"] for k in names]
        o["wa"]["loo_r2"], o["wa"]["coef"] = loo_r2(d, fanin, left)
        o["wa"]["left_without_stem"] = rho(d[1:], left[1:])                   # names[0] is the stem
        o["wa"]["left_given_fanin"] = partial_rho(d, left, fanin)
        out[s] = o
    return out


def llm(table):
    out = {}
    for s, row in table["seeds"].items():
        u = row["before"]["units"]
        keys = list(u)
        d = [u[k]["dloss"] for k in keys]
        fanin = [LLM_FANIN[k.split(".")[1]] for k in keys]
        left = [LLM_BLOCKS - 1 - int(k.split(".")[0][1:]) for k in keys]
        w2_lt_o = sum(u[f"b{i}.w2"]["dloss"] < u[f"b{i}.o"]["dloss"] for i in range(LLM_BLOCKS))
        w2_lt_w13 = sum(u[f"b{i}.w2"]["dloss"] < u[f"b{i}.w13"]["dloss"] for i in range(LLM_BLOCKS))
        groups = {g: rho([u[f"b{i}.{g}"]["dloss"] for i in range(LLM_BLOCKS)], [LLM_BLOCKS - 1 - i for i in range(LLM_BLOCKS)])
                  for g in LLM_FANIN}
        r2, coef = loo_r2(d, fanin, left)
        out[s] = {"fanin": rho(d, fanin), "left": rho(d, left), "left_inside_groups": groups,
                  "left_inside_groups_mean": round(float(np.mean(list(groups.values()))), 3),
                  "w2_below_o_blocks": f"{w2_lt_o}/{LLM_BLOCKS}", "w2_below_w13_blocks": f"{w2_lt_w13}/{LLM_BLOCKS}",
                  "loo_r2": r2, "coef": coef,
                  "mean_by_group": {g: round(float(np.mean([u[f"b{i}.{g}"]["dloss"] for i in range(LLM_BLOCKS)])), 4)
                                    for g in LLM_FANIN}}
    return out


def main():
    res = {"resnet8": resnet(json.load(open(os.path.join(HERE, "results", "layer_table.json")))),
           "tiny_llm": llm(json.load(open(os.path.join(HERE, "results", "llm_layers.json"))))}
    json.dump(res, open(os.path.join(HERE, "results", "fanin_depth.json"), "w"), indent=1)
    print("ResNet-8 (drop wa / w / a): Spearman with fan-in | with layers left | LOO R^2 (wa)")
    for s, o in res["resnet8"].items():
        print(f"  s{s}: fan-in {o['wa']['fanin']} / {o['w']['fanin']} / {o['a']['fanin']} | left {o['wa']['left']} / "
              f"{o['w']['left']} / {o['a']['left']} | LOO R^2 {o['wa']['loo_r2']} coef {o['wa']['coef']} | left without "
              f"stem {o['wa']['left_without_stem']}, left given fan-in {o['wa']['left_given_fanin']}")
    print("tiny_llm (dloss): Spearman with fan-in | with blocks left (all, inside groups) | w2 < o, w2 < w13 | LOO R^2")
    for s, o in res["tiny_llm"].items():
        print(f"  s{s}: fan-in {o['fanin']} | left {o['left']}, groups {o['left_inside_groups']} mean "
              f"{o['left_inside_groups_mean']} | w2<o {o['w2_below_o_blocks']} w2<w13 {o['w2_below_w13_blocks']} | "
              f"LOO R^2 {o['loo_r2']} | mean {o['mean_by_group']}")


if __name__ == "__main__":
    main()
