"""
same_bytes.py -- README part 8: the same-bytes control on ResNet-8. Does the one-shot order (part 2's drop table) tell
where extra bytes help after training?

Each configuration protects a set of layers (real weights + hardtanh, part 3's real layers), every other layer binary,
trained from the twin with part 3's recipe (config_table.measure_seed). The pairs (same weight count, P2's control
+33 %) and the rule were written in the README before training. Validation split only; the test set is never read.
Resumable per seed (same settings and inputs only).

    python same_bytes.py --seeds 42 43 44 --runs <training_gap>/runs/train_gap --data <repo>/benchmarks/data/cifar10
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

import torch

import config_table as CT
import layer_table as LT
from layer_table import G, LAYERS

HERE = os.path.dirname(os.path.abspath(__file__))

# pair -> (predicted: the larger drop sum, control), protected layers; fixed in the README before training
PAIRS = {"P1": (["b0"], ["b1"]),
         "P2": (["stem"], ["b2"]),
         "P3": (["b0", "b1"], ["b2"]),
         "P4": (["b0", "b1", "b2", "b3"], ["b4"])}


def protected_sets():
    """every distinct protected set of the pairs, plus the reference (none protected)"""
    out = [[]]
    for pred, ctrl in PAIRS.values():
        for s in (pred, ctrl):
            if s not in out:
                out.append(s)
    return out


def weight_counts():
    net = G.build("twin")
    return {k: m.weight.numel() for k, (m, _) in LT.modules_of(net).items()}


def analyze(res):
    """the README rule: per pair d = mean(predicted) - mean(control), s = larger sd"""
    seeds = sorted(res["seeds"])
    acc = {}
    for seed in seeds:
        for rec in res["seeds"][seed]["configs"]:
            key = "+".join(sorted(set(LAYERS) - set(rec["binary"]), key=LAYERS.index)) or "(none)"
            acc.setdefault(key, []).append(rec["trained"])
    stats = {k: {"per_seed": v, "mean": round(statistics.mean(v), 2),
                 "sd": round(statistics.stdev(v), 2) if len(v) > 1 else None} for k, v in acc.items()}
    out = {"accuracy": stats, "pairs": {}}
    if len(seeds) < 3:
        return out
    wc = res["weight_counts"]
    for name, (pred, ctrl) in PAIRS.items():
        p, c = stats["+".join(pred)], stats["+".join(ctrl)]
        d, s = round(p["mean"] - c["mean"], 2), max(p["sd"], c["sd"])
        verdict = "WHERE" if d > s else "OPPOSITE" if d < -s else "TIE"
        out["pairs"][name] = {"predicted": pred, "control": ctrl, "weights": [sum(wc[k] for k in pred),
                              sum(wc[k] for k in ctrl)], "d": d, "s": s, "verdict": verdict}
    n = sum(v["verdict"] == "WHERE" for v in out["pairs"].values())
    out["where_pairs"] = n
    out["reading"] = ("the one-shot order tells where (keep estimate)" if n >= 3 else
                      "bytes, not where (estimate's per-layer raise not supported)" if n <= 1 else "mixed")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--runs", required=True, help="training_gap's runs/train_gap (twin_s<seed>.pth)")
    ap.add_argument("--data", required=True, help="the CIFAR-10 folder (benchmarks/data/cifar10)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--calib", type=int, default=10000)
    ap.add_argument("--fisher-samples", type=int, default=512)
    ap.add_argument("--table", default=os.path.join(HERE, "results", "layer_table.json"), help="part 2's drops")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "same_bytes.json"))
    args = ap.parse_args()
    args.runs, args.data, args.limit = os.path.expanduser(args.runs), os.path.expanduser(args.data), 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = json.load(open(args.table))
    protect = protected_sets()
    configs = [[k for k in LAYERS if k not in p] for p in protect]   # measure_seed takes the BINARY layers
    same = {k: v for k, v in vars(args).items() if k not in ("seeds", "out", "runs", "data", "table", "limit")}
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    old = res.get("args", {})
    if old and {k: old.get(k) for k in same} != same:
        sys.exit(f"{args.out} was measured with other settings ({old} vs {same}): use another --out")
    tsha = hashlib.sha256(open(args.table, "rb").read()).hexdigest()   # the drops a resumed seed was measured with
    if res.get("table_sha256", tsha) != tsha:
        sys.exit(f"{args.out} was measured with another --table ({res['table_sha256'][:12]} vs {tsha[:12]}): "
                 "use another --out")
    res.update(args=same, layers=LAYERS, pairs=PAIRS, protected=protect, weight_counts=weight_counts(),
               table_sha256=tsha)
    res.setdefault("seeds", {})
    for seed in args.seeds:
        if str(seed) in res["seeds"]:
            if res["seeds"][str(seed)]["provenance"] != LT.provenance(seed, args, dev):
                sys.exit(f"s{seed} in {args.out} was measured on other inputs: use another --out")
            print(f"  s{seed}: already in {args.out}")
            continue
        res["seeds"][str(seed)] = CT.measure_seed(seed, args, dev, configs, table)
        res["analysis"] = analyze(res)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    res["analysis"] = analyze(res)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res["analysis"], indent=1))


if __name__ == "__main__":
    main()
