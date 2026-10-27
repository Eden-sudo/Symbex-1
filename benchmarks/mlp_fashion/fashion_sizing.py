"""
fashion_sizing.py -- where should the width of a binary MLP go? Shapes 784 -> h1 -> h2 -> 10, from the data.

    python fashion_sizing.py                          # every shape of the grid, seed 42
    python fashion_sizing.py --shapes 256-512 384-512
    python fashion_sizing.py --show                   # only the table of what is trained
    python fashion_sizing.py --shapes 128-512 256-512 --seeds 3     # confirm: seeds 42, 43, 44 (mean +- sd)
    python fashion_sizing.py --shapes 128-512 --seeds 3 --no-distill  # on the labels only (saved apart: _nodistill)

Run from benchmarks/mlp_fashion/ (environment env_bench; model.py must know the widths: patch_model_widths.py).

Why: on the chip the FIRST layer is the expensive one -- its inputs are 8-bit pixels, so it sums bytes; the hidden
layers are binary (AND + bit count) and cost about half per weight. The benchmark's x2 model (784-512-256) doubles the
first layer; the same width could go to the second, binary one. Every shape trains its own FP32 model (model.get_fp32),
then the CDT binarizes it with the recipe of lowbit_study.py (method plain, every layer 1 bit, 15 epochs, distillation,
no widening). The shapes are ranked on the VALIDATION accuracy (carved out of the training data); the test accuracy is
reported apart, measured once.

Latency is PREDICTED from the chip: the cost per weight of a uint8 x 1-bit layer and of a 1-bit x 1-bit layer, taken
from the layer times of the lowbit_study exports already measured on a board (runs/lowbit/plain_w*_s42/chip_*.json,
lowbit_chip.py), for every board and weight placement measured. Confirm the interesting shapes on the board:
    python ../../tools/chip_test.py runs/sizing/h256_512_s42 --board s3 --port /dev/ttyUSB0 --no-int8
Results: results/fashion_sizing.json.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CDT = os.path.normpath(os.path.join(HERE, "..", "..", "tools", "cdt.py"))
GRID = [(h1, h2) for h1 in (128, 256, 384, 512) for h2 in (128, 256, 512)]
EPOCHS = 15
SUFFIX = ""                                   # "_nodistill" with --no-distill: its own runs and results


def config_for(h1, h2):
    """symbex.toml with the model of this shape (its file, its definition) and no widening, next to it"""
    text = open(os.path.join(HERE, "symbex.toml")).read()
    for key, value in (("file", f'"models/fashion_mlp_{h1}_{h2}.pth"' if (h1, h2) != (256, 128) else '"models/fashion_mlp.pth"'),
                       ("definition", f'"model.py:FashionMLP_{h1}_{h2}"'), ("widths", "[1]")):
        text, n = re.subn(rf"(?m)^{key}(\s*)=.*$", lambda m: f"{key}{m.group(1)}= {value}", text)
        if n != 1:
            sys.exit(f"[X] symbex.toml: expected one '{key} =' line, found {n}")
    path = os.path.join(HERE, f".sizing_{h1}_{h2}.toml")
    open(path, "w").write(text)
    return path


def run(h1, h2, seed):
    out = os.path.join("runs", "sizing", f"h{h1}_{h2}_s{seed}{SUFFIX}")
    if os.path.exists(os.path.join(HERE, out, "export.json")):
        print(f"[*] 784-{h1}-{h2}-10: already done")
        return
    sys.path.insert(0, HERE)
    from model import get_fp32
    get_fp32(h1, h2)                                       # its own FP32 model, trained once
    print(f"\n[*] 784-{h1}-{h2}-10, seed {seed} -> {out}")
    config = config_for(h1, h2)
    cmd = [sys.executable, CDT, config, "--method", "plain", "--seed", str(seed), "--epochs", str(EPOCHS),
           "--search-epochs", "1", "--max-accuracy-drop", "100", "--output-directory", out, "--to", "export", "--no-verbose"
           ] + (["--no-distill"] if SUFFIX else [])
    try:
        code = subprocess.run(cmd, cwd=HERE).returncode
    finally:
        os.remove(config)
    if code != 0:
        sys.exit(f"[X] the CDT failed on {out} (the output above says why)")


def chip_costs():
    """{(board, weights): (ns per uint8 x 1-bit weight, ns per 1-bit x 1-bit weight)} from the measured layer times
    of the lowbit_study exports (784 -> h1 is uint8 x 1 bit; the next layers are 1 bit x 1 bit)"""
    out = {}
    for f in glob.glob(os.path.join(HERE, "runs", "lowbit", "plain_w*_s42", "chip_*.json")):
        d = json.load(open(f))
        layers = d.get("layers") or []
        shapes = [tuple(int(v) for v in re.findall(r"(\d+) -> (\d+)", l["step"])[0]) for l in layers]
        if len(layers) != 3:
            continue
        first = layers[0]["median_ms"] * 1e6 / (shapes[0][0] * shapes[0][1])
        rest = sum(l["median_ms"] for l in layers[1:]) * 1e6 / sum(a * b for a, b in shapes[1:])
        key = (d.get("board"), d.get("weights"))
        out.setdefault(key, []).append((first, rest, shapes[0][1]))
    # the widest measured model of each setting: the most weights, the steadiest per-weight cost
    return {k: max(v, key=lambda t: t[2])[:2] for k, v in out.items()}


def predict(h1, h2, cost):
    first, rest = cost
    return (784 * h1 * first + (h1 * h2 + h2 * 10) * rest) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=[f"{a}-{b}" for a, b in GRID], help="h1-h2 (default: the grid)")
    ap.add_argument("--seed", type=int, default=42, help="the first seed")
    ap.add_argument("--seeds", type=int, default=1, help="how many seeds from --seed (the table: mean +- sd)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-distill", action="store_true", help="on the labels only (no distillation; saved apart)")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = "_nodistill" if args.no_distill else ""
    os.chdir(HERE)
    shapes = [tuple(int(v) for v in s.split("-")) for s in args.shapes]
    if not args.show:
        for h1, h2 in shapes:
            for s in range(args.seed, args.seed + args.seeds):
                run(h1, h2, s)
    costs = chip_costs()
    settings = sorted(costs)
    rows = []
    import statistics
    for h1, h2 in shapes:
        vals, tests, flash = [], [], 0
        for s in range(args.seed, args.seed + args.seeds):
            d = os.path.join("runs", "sizing", f"h{h1}_{h2}_s{s}{SUFFIX}")
            if not os.path.exists(os.path.join(d, "export.json")):
                continue
            tr = json.load(open(os.path.join(d, "train.json")))
            vals.append(tr.get("acc_last"))
            tests.append(tr.get("test", {}).get("acc_symbex"))
            flash = json.load(open(os.path.join(d, "export.json"))).get("flash_bytes", 0)
        if not vals:
            continue
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        sys.path.insert(0, HERE)
        from model import fp32_path
        fp = os.path.join(HERE, fp32_path(h1, h2))
        sha = hashlib.sha256(open(fp, "rb").read()).hexdigest() if os.path.exists(fp) else None
        rows.append(dict(shape=f"784-{h1}-{h2}-10", h1=h1, h2=h2, seeds=len(vals), val=statistics.mean(vals),
                         val_sd=sd(vals), test=statistics.mean(tests), test_sd=sd(tests),
                         weight_kb=(784 * h1 + h1 * h2 + h2 * 10) / 8 / 1024, flash_kb=flash / 1024,
                         ms={f"{b} {w}": predict(h1, h2, costs[(b, w)]) for b, w in settings}, fp32_sha256=sha))
    ref = next((r for r in rows if (r["h1"], r["h2"]) == (512, 256)), None)
    first = f"{settings[0][0]} {settings[0][1]}" if settings else None
    for r in rows:                                       # Pareto on validation (the choice) against the first setting
        r["pareto"] = first is not None and not any(o is not r and o["val"] >= r["val"] and o["ms"][first] <= r["ms"][first]
                                                    and (o["val"] > r["val"] or o["ms"][first] < r["ms"][first]) for o in rows)
    print(f"\n    SIZING THE BINARY MLP  (Fashion-MNIST, every layer 1 bit, {EPOCHS} epochs, "
          f"{'labels only' if SUFFIX else 'distillation'}, "
          f"{args.seeds} seed(s) from {args.seed};"
          f" ranked on VALIDATION, test measured once;\n    latency PREDICTED from the per-weight costs measured on the chip"
          + (f": {', '.join(f'{b} {w}' for b, w in settings)})" if settings else ": none measured yet, run lowbit_chip.py)"))
    print()
    head = f"    {'shape':<18}{'weights KB':>11}{'val acc':>16}{'test acc':>16}" + "".join(f"{'ms ' + b + ' ' + w:>15}" for b, w in settings)
    print(head + "  Pareto")
    print("    " + "-" * (len(head) + 4))
    for r in sorted(rows, key=lambda r: r["ms"].get(first, 0) if first else r["weight_kb"]):
        mark = "  *" if r["pareto"] else ""
        if ref is r:
            mark += "   <- the benchmark x2 model"
        print(f"    {r['shape']:<18}{r['weight_kb']:>11.1f}{r['val']:>9.2f} +-{r['val_sd']:5.2f}{r['test']:>9.2f} +-{r['test_sd']:5.2f}"
              + "".join(f"{r['ms'][f'{b} {w}']:>15.3f}" for b, w in settings) + mark)
    for k, (c1, c2) in sorted(costs.items()):
        print(f"\n    chip cost per weight, {k[0]} {k[1]}: first layer (uint8 x 1 bit) {c1:.2f} ns, binary layers {c2:.2f} ns", end="")
    print("\n\n    * Pareto: no other shape is more accurate on validation AND faster (first setting)")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(seed=args.seed, seeds=args.seeds, epochs=EPOCHS, distill=not SUFFIX,
                   costs={f"{b} {w}": c for (b, w), c in costs.items()}, rows=rows),
              open(os.path.join("results", f"fashion_sizing{SUFFIX}.json"), "w"), indent=1)
    print(f"    -> results/fashion_sizing{SUFFIX}.json")


if __name__ == "__main__":
    main()
