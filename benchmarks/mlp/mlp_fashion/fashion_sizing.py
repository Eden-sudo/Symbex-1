"""
fashion_sizing.py -- where should the width of a binary MLP go? Shapes 784 -> h1 -> h2 -> 10, from the data.

    python fashion_sizing.py                          # every shape of the grid, seed 42
    python fashion_sizing.py --shapes 256-512 384-512
    python fashion_sizing.py --show                   # only the table of what is trained
    python fashion_sizing.py --shapes 128-512 256-512 --seeds 3     # confirm: seeds 42, 43, 44 (mean +- sd)
    python fashion_sizing.py --shapes 128-512 --seeds 3 --no-distill  # on the labels only (saved apart: _nodistill)

Run from benchmarks/mlp/mlp_fashion/ (environment env_bench; model.py must know the widths: patch_model_widths.py).

Why: on the chip the FIRST layer is the expensive one -- its inputs are 8-bit pixels, so it sums bytes; the hidden
layers are binary (AND + bit count) and cost about half per weight. The benchmark's x2 model (784-512-256) doubles the
first layer; the same width could go to the second, binary one. Every shape trains its own FP32 model (model.get_fp32),
then the CDT binarizes it with the recipe of lowbit_study.py (method plain, every layer 1 bit, 15 epochs, distillation,
no widening). The shapes are ranked on the VALIDATION accuracy (carved out of the training data); the test accuracy is
reported apart, measured once.

Latency is PREDICTED from the chip: the cost per weight of a uint8 x 1-bit layer and of a 1-bit x 1-bit layer, taken
from the layer times of the lowbit_study exports already measured on a board (runs/lowbit/plain_w*_s42/chip_*.json,
earlier/lowbit_chip.py), for every board and weight placement measured. The prediction is only a guide: with the weights in
flash the cost per weight is not linear (a small first layer fits the flash cache, a wide one does not: 784-128-512
predicted 1.71 ms, measured 0.42 ms on the S3). So the table shows the MEASURED latency wherever a shape has been run
on the board (runs/sizing/h<h1>_<h2>_s42*/chip_*.json; the latency depends on the shape, not on the weight values),
and the prediction, marked "~", only where it has not. Measure the shapes on the board (weights in flash, then RAM):
    python fashion_sizing.py --shapes 128-512 256-512 384-512 512-256 --seeds 3 --measure --board s3 --port /dev/ttyUSB0
Results: results/sizing/fashion_sizing.json.
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
CDT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "cdt.py"))
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


def measured(h1, h2, board, weights):
    """the latency measured on the board for this shape (any seed-42 export of it: the latency depends on the shape,
    not on the weight values), or None; the newest file wins"""
    files = glob.glob(os.path.join(HERE, "runs", "sizing", f"h{h1}_{h2}_s42*", f"chip_{board}_*_{weights}_*.json"))
    if not files:
        return None
    d = json.load(open(max(files, key=os.path.getmtime)))
    if d.get("equal_simulation") != 100.0:
        print(f"    [!] {os.path.relpath(max(files, key=os.path.getmtime), HERE)}: the board is not 100 % equal to the "
              f"simulation ({d.get('equal_simulation')}): its latency is not used")
        return None
    return d.get("median_ms")


def measure(h1, h2, args):
    """chip_test.py on the seed-42 export of this shape, for every weight placement not measured yet"""
    folder = os.path.join(HERE, "runs", "sizing", f"h{h1}_{h2}_s42{SUFFIX}")
    if not os.path.exists(os.path.join(folder, "export.json")):
        print(f"[!] {os.path.relpath(folder, HERE)}: no export: not measured")
        return
    chip_test = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "chip_test.py"))
    for weights in ("flash", "ram"):
        if measured(h1, h2, args.board, weights) is not None:
            continue
        print(f"\n[*] 784-{h1}-{h2}-10 on the {args.board}, weights in {weights}")
        cmd = [sys.executable, chip_test, folder, "--board", args.board, "--port", args.port, "--weights", weights,
               "--no-int8"] + (["--usb"] if args.usb else [])
        if subprocess.run(cmd, cwd=os.path.dirname(chip_test)).returncode != 0:
            sys.exit(f"[X] chip_test.py failed on {folder} (the output above says why)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=[f"{a}-{b}" for a, b in GRID], help="h1-h2 (default: the grid)")
    ap.add_argument("--seed", type=int, default=42, help="the first seed")
    ap.add_argument("--seeds", type=int, default=1, help="how many seeds from --seed (the table: mean +- sd)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-distill", action="store_true", help="on the labels only (no distillation; saved apart)")
    ap.add_argument("--measure", action="store_true",
                    help="run the seed-42 export of every shape not measured yet on the board (flash, then RAM)")
    ap.add_argument("--board", choices=["esp32", "s3"], default="s3")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--usb", action="store_true", help="the S3's native USB port (chip_test.py --usb)")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = "_nodistill" if args.no_distill else ""
    os.chdir(HERE)
    shapes = [tuple(int(v) for v in s.split("-")) for s in args.shapes]
    if not args.show:
        for h1, h2 in shapes:
            for s in range(args.seed, args.seed + args.seeds):
                run(h1, h2, s)
    if args.measure:
        for h1, h2 in shapes:
            measure(h1, h2, args)
    costs = chip_costs()
    settings = sorted(costs)
    if args.measure:                                     # the measured setting has a column even with no prediction
        settings = sorted(set(settings) | {(args.board, "flash"), (args.board, "ram")})
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
                         ms_predicted={f"{b} {w}": predict(h1, h2, costs[(b, w)]) if (b, w) in costs else None
                                       for b, w in settings},
                         ms_measured={f"{b} {w}": measured(h1, h2, b, w) for b, w in settings}, fp32_sha256=sha))
        r = rows[-1]                                     # measured where it exists, else the prediction
        r["ms"] = {k: r["ms_measured"][k] if r["ms_measured"][k] is not None else r["ms_predicted"][k] for k in r["ms_measured"]}
    ref = next((r for r in rows if (r["h1"], r["h2"]) == (512, 256)), None)
    first = f"{settings[0][0]} {settings[0][1]}" if settings else None
    for r in rows:                                       # Pareto on validation (the choice) against the first setting
        ms = lambda x: x["ms"].get(first)
        r["pareto"] = ms(r) is not None and not any(
            o is not r and ms(o) is not None and o["val"] >= r["val"] and ms(o) <= ms(r)
            and (o["val"] > r["val"] or ms(o) < ms(r)) for o in rows)
    print(f"\n    SIZING THE BINARY MLP  (Fashion-MNIST, every layer 1 bit, {EPOCHS} epochs, "
          f"{'labels only' if SUFFIX else 'distillation'}, "
          f"{args.seeds} seed(s) from {args.seed};"
          f" ranked on VALIDATION, test measured once;\n    latency MEASURED on the board (seed-42 export, median over the "
          f"test images, 100 % equal to the simulation); ~ = PREDICTED from per-weight costs, not measured yet"
          + (f": {', '.join(f'{b} {w}' for b, w in settings)})" if settings else ": none measured yet, run earlier/lowbit_chip.py)"))
    print()
    head = f"    {'shape':<18}{'weights KB':>11}{'val acc':>16}{'test acc':>16}" + "".join(f"{'ms ' + b + ' ' + w:>15}" for b, w in settings)
    print(head + "  Pareto")
    print("    " + "-" * (len(head) + 4))
    def cell(r, k):
        if r["ms_measured"].get(k) is not None:
            return f"{r['ms_measured'][k]:>15.3f}"
        return f"{'~' + format(r['ms_predicted'][k], '.3f'):>15}" if r["ms_predicted"].get(k) is not None else f"{'-':>15}"
    for r in sorted(rows, key=lambda r: (r["ms"].get(first) or 0) if first else r["weight_kb"]):
        mark = "  *" if r["pareto"] else ""
        if ref is r:
            mark += "   <- the benchmark x2 model"
        print(f"    {r['shape']:<18}{r['weight_kb']:>11.1f}{r['val']:>9.2f} +-{r['val_sd']:5.2f}{r['test']:>9.2f} +-{r['test_sd']:5.2f}"
              + "".join(cell(r, f"{b} {w}") for b, w in settings) + mark)
    for k, (c1, c2) in sorted(costs.items()):
        print(f"\n    chip cost per weight, {k[0]} {k[1]}: first layer (uint8 x 1 bit) {c1:.2f} ns, binary layers {c2:.2f} ns", end="")
    print("\n\n    * Pareto: no other shape is more accurate on validation AND faster (first setting)")
    os.makedirs(os.path.join("results", "sizing"), exist_ok=True)
    json.dump(dict(seed=args.seed, seeds=args.seeds, epochs=EPOCHS, distill=not SUFFIX,
                   costs={f"{b} {w}": c for (b, w), c in costs.items()}, rows=rows),
              open(os.path.join("results", "sizing", f"fashion_sizing{SUFFIX}.json"), "w"), indent=1)
    print(f"    -> results/sizing/fashion_sizing{SUFFIX}.json")


if __name__ == "__main__":
    main()
