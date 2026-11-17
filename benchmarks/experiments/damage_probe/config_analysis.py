"""
config_analysis.py -- README part 3, the checks written AFTER the run (not in the pre-registered rule):
    count       baseline: the number of binary layers (does a measure beat counting?)
    partial     Spearman with the number of binary layers held fixed (ranks regressed on the count, residuals
                correlated): does the measure see WHICH layers, not only HOW MANY?
    drop_cov    Hill's coverage model (arXiv 2607.12266) on part 2's drops: sum of -log(1 - drop / c), c = recalibrated
                twin accuracy - 10 (declared in LOGBOOK.md before the analysis)
    ceiling     the same statistics between the trained accuracies of two seeds (how far a measure could go)
Reads results/config_table.json + results/layer_table.json; writes results/config_analysis.json.

    python config_analysis.py
"""
import argparse
import itertools
import json
import math
import os

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from config_table import MEASURES

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = (("all", lambda b: True), ("stem_real", lambda b: "stem" not in b), ("stem_binary", lambda b: "stem" in b))


def spear(x, y):
    return round(float(spearmanr(x, y).statistic), 3)


def partial(x, y, n):
    """Spearman of x and y with n held fixed: ranks regressed on n, residuals correlated"""
    X = np.c_[np.ones(len(n)), n]
    res = [rankdata(v) - X @ np.linalg.lstsq(X, rankdata(v), rcond=None)[0] for v in (x, y)]
    return round(float(pearsonr(*res).statistic), 3)


def drop_cov(binary, drop, c):
    return sum(-math.log(1 - min(drop[k], c - 1e-6) / c) for k in binary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=os.path.join(HERE, "results", "config_table.json"))
    ap.add_argument("--table", default=os.path.join(HERE, "results", "layer_table.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "config_analysis.json"))
    args = ap.parse_args()
    res, table = json.load(open(args.configs)), json.load(open(args.table))
    seeds = sorted(res["seeds"])
    out = {"seeds": {}, "ceiling": {}}
    for s in seeds:
        rows = res["seeds"][s]["configs"]
        t = table["seeds"][s]
        drop = {k: v.get("wa", v["w"]) for k, v in t["drop"].items()}
        c = t["twin_recalibrated"] - 10
        score = {m: [sign * r[m] for r in rows] for m, sign in MEASURES.items()}
        score["drop_cov"] = [-drop_cov(r["binary"], drop, c) for r in rows]
        score["count"] = [-len(r["binary"]) for r in rows]
        o = {}
        for m, x in score.items():
            o[m] = {g: spear([v for v, r in zip(x, rows) if keep(r["binary"])],
                             [r["trained"] for r in rows if keep(r["binary"])]) for g, keep in GROUPS}
            o[m]["partial_count"] = partial(x, [r["trained"] for r in rows], [len(r["binary"]) for r in rows])
        out["seeds"][s] = o
    n = [len(b) for b in res["configurations"]]
    for a, b in itertools.combinations(seeds, 2):
        ya, yb = ([r["trained"] for r in res["seeds"][k]["configs"]] for k in (a, b))
        o = {g: spear([v for v, cfg in zip(ya, res["configurations"]) if keep(cfg)],
                      [v for v, cfg in zip(yb, res["configurations"]) if keep(cfg)]) for g, keep in GROUPS}
        o["partial_count"] = partial(ya, yb, n)
        out["ceiling"][f"{a}_vs_{b}"] = o
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"{'measure':9s} " + " | ".join(f"{g:>17s}" for g in ("all", "stem_real", "stem_binary", "partial_count")))
    for m in out["seeds"][seeds[0]]:
        cells = [" / ".join("%.2f" % out["seeds"][s][m][g] for s in seeds)
                 for g in ("all", "stem_real", "stem_binary", "partial_count")]
        print(f"{m:9s} " + " | ".join(f"{c:>17s}" for c in cells))
    for k, o in out["ceiling"].items():
        print(f"ceiling {k}: {o}")


if __name__ == "__main__":
    main()
