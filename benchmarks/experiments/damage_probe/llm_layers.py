"""
llm_layers.py -- a fast check (no training, minutes per seed), not a result: which layers of tiny_llm hurt when they
go to 1 bit, how the damage travels through the blocks after them, and whether the layers predicted on the fp32
model are the ones that weigh in the trained binary (W1A8) model.

Units: 8 blocks x 4 groups (qkv, o, w13 = w1 w3, w2).

    BEFORE, fp32 model (fp32_d384l8): one unit gets 1-bit weights + 8-bit inputs (W1A8, as training does), the rest
        stays fp32. Records dloss (validation loss - the model's own) and, per block j, the relative error of the
        residual stream after block j: |x_j' - x_j| / |x_j| (the propagation matrix, unit x block).
        Also every unit at once (sum of the single dlosses vs the joint one: do they add up?).
    AFTER, trained binary model (binary_d384l8): one unit gets its latent weights back at int8 (inputs stay 8-bit),
        the rest stays 1-bit. Records the gain (loss - the binary model's loss, negative = better), the same
        propagation matrix, and the unit's local error: |y_bin - y_int8| / |y_int8| on the same input.
    Spearman of BEFORE's dloss with AFTER's gain and AFTER's local error.

Validation split only (train.py's evaluate, fixed windows). No checkpoint is changed.

    python llm_layers.py --seeds 42 43 44 --runs ~/Proyectos/Symbex1/benchmarks/experiments/tiny_llm/runs \
        --data ~/Proyectos/Symbex1/benchmarks/data/tinystories          # results/llm_layers.json
"""
import argparse
import json
import os
import time

import torch
from scipy.stats import spearmanr

import damage_probe as D
from damage_probe import T

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {"qkv": ("q", "k", "v"), "o": ("o",), "w13": ("w1", "w3"), "w2": ("w2",)}


def units(model):
    return {f"b{i}.{g}": [getattr(b, n) for n in names] for i, b in enumerate(model.blocks) for g, names in GROUPS.items()}


@torch.no_grad()
def residuals(model, idx):
    """the residual stream after every block (one fixed batch)"""
    out = []
    hooks = [b.register_forward_hook(lambda m, i, o: out.append(o.float())) for b in model.blocks]
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(idx)
    for h in hooks:
        h.remove()
    return out


def rel(a, b):
    return round(((a - b).norm() / b.norm()).item(), 4)


@torch.no_grad()
def local_error(model, lins, idx, kind_now, kind_ref):
    """|y(kind_now) - y(kind_ref)| / |y(kind_ref)| of each linear in lins on its own input, summed over lins"""
    ins = {}
    hooks = [l.register_forward_hook(lambda m, i, o, l=l: ins.__setitem__(id(l), i[0])) for l in lins]
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(idx)
    for h in hooks:
        h.remove()
    tot = 0.0
    for l in lins:
        x = ins[id(l)].float()
        old = l.kind
        l.kind = kind_now
        y_now = l(x)
        l.kind = kind_ref
        y_ref = l(x)
        l.kind = old
        tot += ((y_now - y_ref).norm() / y_ref.norm()).item()
    return round(tot, 4)


def scan(model, val, seq, idx, args, kind, local_ref=None):
    """every unit switched to `kind` alone: dloss, propagation row (and local error vs local_ref)"""
    base = T.evaluate(model, val, seq, args.batches, 32)
    clean = residuals(model, idx)
    rows = {}
    for u, lins in units(model).items():
        old = D.apply([(l, "kind", kind) for l in lins])
        loss = T.evaluate(model, val, seq, args.batches, 32)
        prop = [rel(a, b) for a, b in zip(residuals(model, idx), clean)]
        D.apply(old)
        rows[u] = {"dloss": round(loss - base, 4), "propagation": prop}
        if local_ref:
            rows[u]["local"] = local_error(model, lins, idx, local_ref[0], local_ref[1])
    return base, rows


def measure_seed(seed, args, dev):
    val = T.load_split("val", dev)
    fp, a = D.load(args.runs, f"fp32_d384l8_s{seed}", dev)
    seq = a["seq"]
    idx = torch.stack([val[i * seq:(i + 1) * seq].long() for i in range(args.prop_windows)])
    out = {}
    base, rows = scan(fp, val, seq, idx, args, "binary")
    old = D.apply([(l, "kind", "binary") for ls in units(fp).values() for l in ls])
    joint = T.evaluate(fp, val, seq, args.batches, 32) - base
    D.apply(old)
    out["before"] = {"run": f"fp32_d384l8_s{seed}", "loss": round(base, 4), "units": rows,
                     "all_at_once": round(joint, 4), "sum_of_singles": round(sum(r["dloss"] for r in rows.values()), 4)}
    del fp
    bn, _ = D.load(args.runs, f"binary_d384l8_s{seed}", dev)
    base, rows = scan(bn, val, seq, idx, args, "int8", local_ref=("binary", "int8"))
    out["after"] = {"run": f"binary_d384l8_s{seed}", "loss": round(base, 4), "units": rows}
    del bn
    torch.cuda.empty_cache()
    u = list(rows)
    pred = [out["before"]["units"][k]["dloss"] for k in u]
    gain = [-rows[k]["dloss"] for k in u]  # larger = the unit weighs more in the binary model
    loc = [rows[k]["local"] for k in u]
    out["spearman"] = {"before_vs_after_gain": round(float(spearmanr(pred, gain).statistic), 3),
                       "before_vs_after_local": round(float(spearmanr(pred, loc).statistic), 3),
                       "after_gain_vs_after_local": round(float(spearmanr(gain, loc).statistic), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batches", type=int, default=20, help="validation batches of 32 windows (as damage_map.py)")
    ap.add_argument("--prop-windows", type=int, default=16, help="validation windows for the propagation matrix")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "llm_layers.json"))
    args = ap.parse_args()
    args.runs, T.DATA = os.path.expanduser(args.runs), os.path.expanduser(args.data)
    dev = "cuda"
    res = {"args": {k: v for k, v in vars(args).items() if k not in ("runs", "data")}, "seeds": {}}
    for s in args.seeds:
        t0 = time.time()
        torch.manual_seed(s)
        res["seeds"][str(s)] = r = measure_seed(s, args, dev)
        r["seconds"] = round(time.time() - t0, 1)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        b, f = r["before"], r["after"]
        print(f"seed {s}: fp32 {b['loss']} | binary {f['loss']} | all units 1-bit at once +{b['all_at_once']} "
              f"vs sum of singles +{b['sum_of_singles']} | {r['seconds']} s")
        print(f"{'unit':7s} {'before dloss':>12s} {'after gain':>10s} {'local':>7s}  before propagation (blocks 0-7)")
        for k in f["units"]:
            pb = " ".join("%.2f" % v for v in b["units"][k]["propagation"])
            print(f"{k:7s} {b['units'][k]['dloss']:+12.4f} {-f['units'][k]['dloss']:+10.4f} "
                  f"{f['units'][k]['local']:7.3f}  {pb}")
        print(f"Spearman {r['spearman']}")


if __name__ == "__main__":
    main()
