"""
har_rf.py -- a Random Forest on the raw windows: a yardstick for how hard this split is (README section 6).

    python har_rf.py                    # grid on validation, then the test subjects once; seeds 42, 43, 44

Run from benchmarks/conv1d/har_ultracompact/ (env_bench). Daghero et al. report their "RF raw" (a Random Forest on the
raw accelerometer values) at up to 59 % on UniMiB-SHAR; Micucci et al. 2017 (the data set paper, Tables 7-8) report
the same kind of model, AF-17, raw data, at 81.48 % with 5-fold cross validation (subjects mixed) and 56.53 % leaving
one subject out (macro average accuracy). The same model on this README's split says whether our test subjects are
harder or easier than theirs, independently of any network or training recipe.
    inputs      the int8 windows of har_data.py, flattened (151 x 3 = 453 values), as Micucci's "raw data"
    grid        trees {100, 300} x max depth {10, 20, None}, chosen on the validation subjects (seed 42)
    metrics     accuracy (what this README reports) and macro average accuracy (MAA, the mean per-class recall:
                Micucci's metric)
Writes results/rf/har_rf.json.
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from har_data import load

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 43, 44)


def scores(model, x, y):
    p = model.predict(x)
    maa = np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])
    return 100 * float(np.mean(p == y)), 100 * float(maa)


def main():
    os.chdir(HERE)
    (xtr, ytr), (xva, yva), (xte, yte) = [load(s) for s in ("train", "validation", "test")]
    xtr, xva, xte = [x.reshape(len(x), -1).astype(np.float32) for x in (xtr, xva, xte)]
    grid = []
    for n in (100, 300):
        for d in (10, 20, None):
            m = RandomForestClassifier(n_estimators=n, max_depth=d, n_jobs=-1, random_state=42).fit(xtr, ytr)
            acc, maa = scores(m, xva, yva)
            grid.append(dict(trees=n, depth=d, validation=acc, validation_maa=maa))
            print(f"    trees {n:3d} depth {str(d):>4}: validation {acc:.1f} % (MAA {maa:.1f} %)")
    best = max(grid, key=lambda g: g["validation"])
    print(f"[*] chosen on validation: trees {best['trees']}, depth {best['depth']}")
    runs = []
    for s in SEEDS:
        m = RandomForestClassifier(n_estimators=best["trees"], max_depth=best["depth"], n_jobs=-1,
                                   random_state=s).fit(xtr, ytr)
        va, va_maa = scores(m, xva, yva)
        te, te_maa = scores(m, xte, yte)
        runs.append(dict(seed=s, validation=va, validation_maa=va_maa, test=te, test_maa=te_maa))
        print(f"    seed {s}: validation {va:.1f} % | TEST {te:.1f} % (MAA {te_maa:.1f} %)")
    for k in ("validation", "test", "test_maa"):
        v = [r[k] for r in runs]
        print(f"    {k:<14} {np.mean(v):.1f} +- {np.std(v, ddof=1):.1f} %")
    os.makedirs(os.path.join("results", "rf"), exist_ok=True)
    json.dump(dict(grid=grid, chosen=dict(trees=best["trees"], depth=best["depth"]), runs=runs),
              open(os.path.join("results", "rf", "har_rf.json"), "w"), indent=1)
    print("[*] -> results/rf/har_rf.json")


if __name__ == "__main__":
    main()
