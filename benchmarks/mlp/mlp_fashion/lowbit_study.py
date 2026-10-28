"""
lowbit_study.py -- 1 bit against ~1.58 bits (ternary), 2 and 4 bits, on the same network, trained the same way.

    python lowbit_study.py                  # run what is missing, then the table
    python lowbit_study.py --show           # only the table of what is already done
    python lowbit_study.py --seeds 1        # a quick first look (seed 42 only)
    python lowbit_study.py --no-distill     # the same, every method on the labels only (saved apart: _nodistill)

Run from benchmarks/mlp/mlp_fashion/ (environment env_bench). The network is this benchmark's MLP (784 -> 256 -> 128 ->
10) and its FP32 model (models/fashion_mlp.pth); every method starts from that same file, with the same recipe:
15 epochs, distillation from the FP32 model (T = 4, alpha = 0.85), seeds 42, 43, 44, the validation split carved out
of the training data (every choice is made on it) and the 10,000 test images evaluated once, at the end.

    SYMBEX 1 bit   the CDT, method plain, every layer 1 bit, hidden layers x1, x1.5, x2
    ternary        the CDT, method ternary (weights -1, 0, +1: two bit planes), x1, x1.5
    INT4, INT2     fashion_int4.py: quantization-aware training of the same FP32 model, per-channel weights, int8
                   activations, integer evaluation (the original width only: it quantizes the model as it is)
    INT8           the official TensorFlow Lite model (results/tflite/tflite_pc.json), as measured before

Each CDT run has its own folder, runs/lowbit/<method>_w<width>_s<seed>/, and is skipped when its export.json exists.
Table: test accuracy (10,000 images; mean +- standard deviation over the seeds), the accuracy of the INTEGER model on the
2,000 exported test images (the model that runs on the chip; the one measure every row has, the official INT8 included),
packed weight bytes and the total model bytes. Pareto (*), on that integer accuracy: no other row is smaller in weight
bytes and at least as accurate. Results: results/lowbit/lowbit_study.json. Speed on the chip: a separate step (INT4 / INT2 have
no kernels on the ESP32 to measure).
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CDT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "cdt.py"))
SHAPE = (784, 256, 128, 10)
EPOCHS = 15
CDT_RUNS = [("plain", 1.0), ("plain", 1.5), ("plain", 2.0), ("ternary", 1.0), ("ternary", 1.5)]
INT_BITS = [8, 4, 2]                          # INT8 QAT too: the same recipe as INT4/INT2, not only TFLite PTQ
SUFFIX = ""                                   # "_nodistill" with --no-distill: its own runs and results


def fp32_sha256(path=os.path.join(HERE, "models", "fashion_mlp.pth")):
    """the fingerprint of the FP32 model every row starts from: the published tables say which file made them"""
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None
LABEL = {"plain": "SYMBEX 1 bit", "ternary": "ternary"}
WEIGHT_BITS = {"plain": 1, "ternary": 2}


def shape_at(width):
    """the layer sizes at a width: the CDT widens every hidden layer (round(n * width))"""
    return (SHAPE[0],) + tuple(int(round(n * width)) for n in SHAPE[1:-1]) + (SHAPE[-1],)


def packed_weight_bytes(shape, bits):
    return sum(-(-a * b * bits // 8) for a, b in zip(shape[:-1], shape[1:]))


def config_with_width(width, name):
    """a copy of symbex.toml with only its [target] widths line changed, next to it (its relative paths stay valid);
    a list cannot go through the command line (the CDT takes the flag as text)"""
    import re
    text = open(os.path.join(HERE, "symbex.toml")).read()
    new, n = re.subn(r"(?m)^widths(\s*)=.*$", lambda m: f"widths{m.group(1)}= [{width:g}]", text)
    if n != 1:
        sys.exit(f"[X] symbex.toml: expected one 'widths =' line, found {n}")
    path = os.path.join(HERE, f".lowbit_{name}.toml")
    open(path, "w").write(new)
    return path


def run_cdt(method, width, seed):
    out = os.path.join("runs", "lowbit", f"{method}_w{width:g}_s{seed}{SUFFIX}")
    if os.path.exists(os.path.join(HERE, out, "export.json")):
        print(f"[*] {out}: already done")
        return
    print(f"\n[*] {LABEL[method]} x{width:g}, seed {seed} -> {out}")
    config = config_with_width(width, f"{method}_w{width:g}_s{seed}")
    cmd = [sys.executable, CDT, config, "--method", method, "--seed", str(seed), "--epochs", str(EPOCHS),
           "--search-epochs", "1", "--max-accuracy-drop", "100", "--output-directory", out, "--to", "export",
           "--no-verbose"] + (["--no-distill"] if SUFFIX else [])
    try:
        code = subprocess.run(cmd, cwd=HERE).returncode
    finally:
        os.remove(config)
    if code != 0:
        sys.exit(f"[X] the CDT failed on {out} (the output above says why)")


def local_fp32():
    """the test accuracy of THIS machine's FP32 model, as the CDT runs measured it (None before the first one)"""
    import glob
    for tj in sorted(glob.glob(os.path.join(HERE, "runs", "lowbit", f"*_s*{SUFFIX}", "train.json"))):
        v = json.load(open(tj)).get("test", {}).get("acc_fp32")
        if v is not None:
            return round(v, 2)
    return None


def run_int(bits, seeds):
    path = os.path.join(HERE, "results", "lowbit", f"int{bits}_e{EPOCHS}{SUFFIX}_pc.json")
    if os.path.exists(path):
        d = json.load(open(path))
        done = qat_seeds(d)
        mine, theirs = local_fp32(), d.get("acc_fp32_10k")
        same = mine is None or theirs is None or abs(round(theirs, 2) - mine) < 0.005
        split = all("split_seed" in s for s in done)             # trained on the CDT's images, fixed validation split
        if len(done) >= seeds and same and split:
            print(f"[*] INT{bits}: already done ({len(done)} seeds, the same FP32 model and training images)")
            return
        if not split:
            print(f"[*] INT{bits}: {os.path.basename(path)} predates the fixed validation split ([data] split_seed): "
                  "running it again")
        if not same:
            print(f"[*] INT{bits}: {os.path.basename(path)} comes from another FP32 model (test {theirs}% against "
                  f"{mine}% here): running it again on this machine")
    print(f"\n[*] INT{bits} QAT, {seeds} seed(s), {EPOCHS} epochs")
    # fashion_int4.py's own option names (Spanish: --epocas, --semillas) -- change here if they are renamed
    cmd = [sys.executable, os.path.join("earlier", "fashion_int4.py"), "--bits", str(bits), "--epocas", str(EPOCHS), "--semillas", str(seeds)]
    if SUFFIX:
        cmd.append("--no-distill")
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        sys.exit(f"[X] fashion_int4.py --bits {bits} failed (the output above says why)")


def qat_seeds(d):
    """fashion_int4.py stores the QAT of one seed directly, and of several seeds as a list "semillas" """
    q = d.get("metodos", {}).get("qat")
    if not q:
        return []
    return q["semillas"] if "semillas" in q else [q]


def stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def collect(seeds):
    rows, fp32 = [], set()
    for method, width in CDT_RUNS:
        test, int2000, total, n = [], [], [], 0
        for s in range(42, 42 + seeds):
            d = os.path.join(HERE, "runs", "lowbit", f"{method}_w{width:g}_s{s}{SUFFIX}")
            if not os.path.exists(os.path.join(d, "export.json")):
                continue
            tr = json.load(open(os.path.join(d, "train.json")))
            ex = json.load(open(os.path.join(d, "export.json")))
            test.append(tr.get("test", {}).get("acc_symbex"))
            if tr.get("test", {}).get("acc_fp32") is not None:
                fp32.add(round(tr["test"]["acc_fp32"], 2))
            int2000.append(ex.get("acc_integer"))
            total.append(ex.get("flash_bytes"))
            n += 1
        if n:
            shape = shape_at(width)
            rows.append(dict(model=f"{LABEL[method]} x{width:g} ({'-'.join(map(str, shape))})", bits=WEIGHT_BITS[method],
                             seeds=n, test=stats(test), int2000=stats(int2000)[0],
                             weight_bytes=packed_weight_bytes(shape, WEIGHT_BITS[method]), total_bytes=max(total)))
    for bits in INT_BITS:
        path = os.path.join(HERE, "results", "lowbit", f"int{bits}_e{EPOCHS}{SUFFIX}_pc.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        sem = qat_seeds(d)[:seeds]
        if not sem:
            continue
        if d.get("acc_fp32_10k") is not None:
            fp32.add(round(d["acc_fp32_10k"], 2))
        rows.append(dict(model=f"INT{bits} QAT x1 ({'-'.join(map(str, SHAPE))})", bits=bits, seeds=len(sem),
                         test=stats([s.get("precision_10k") for s in sem]),
                         int2000=stats([s.get("precision") for s in sem])[0],
                         weight_bytes=sem[0]["pesos_bytes"], total_bytes=sem[0]["total_bytes"]))
    tfl = os.path.join(HERE, "results", "tflite", "tflite_pc.json")
    if os.path.exists(tfl):
        d = json.load(open(tfl))
        rows.append(dict(model=f"INT8 TFLite x1 ({'-'.join(map(str, SHAPE))})", bits=8, seeds=0, test=(None, None),
                         int2000=d.get("accuracy"), weight_bytes=packed_weight_bytes(SHAPE, 8),
                         total_bytes=d.get("tflite_bytes")))
    for r in rows:                      # one measure for every row: the INTEGER model on the same 2,000 test images
        acc = r["int2000"]
        r["pareto"] = acc is not None and not any(
            o is not r and o["int2000"] is not None and o["weight_bytes"] <= r["weight_bytes"] and o["int2000"] >= acc
            and (o["weight_bytes"] < r["weight_bytes"] or o["int2000"] > acc) for o in rows)
    return sorted(rows, key=lambda r: r["weight_bytes"]), fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--show", action="store_true", help="only the table of what is already done")
    ap.add_argument("--no-distill", action="store_true", help="every method on the labels only (saved apart)")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = "_nodistill" if args.no_distill else ""
    os.chdir(HERE)
    if not args.show:
        for method, width in CDT_RUNS:
            for s in range(42, 42 + args.seeds):
                run_cdt(method, width, s)
        for bits in INT_BITS:
            run_int(bits, args.seeds)
    rows, fp32 = collect(args.seeds)
    sha = fp32_sha256()
    print(f"\n    LOW-BIT STUDY  (Fashion-MNIST MLP, the same FP32 model, {EPOCHS} epochs, "
          f"{'labels only' if SUFFIX else 'distillation from the FP32 model'}, up to {args.seeds} seed(s); test = the "
          f"10,000 test images, once)\n    FP32 model: models/fashion_mlp.pth, sha256 {sha[:16] if sha else '?'}...\n")
    print(f"    {'model':<40}{'bits':>5}{'seeds':>6}{'test acc':>16}{'int 2000':>10}{'weights KB':>12}{'total KB':>10}"
          "  Pareto")
    print("    " + "-" * 107)
    for r in rows:
        t, sd = r["test"]
        tt = f"{t:.2f} +- {sd:.2f}" if t is not None else "-"
        i2 = f"{r['int2000']:.2f}" if r["int2000"] is not None else "-"
        tb = f"{r['total_bytes'] / 1024:.1f}" if r["total_bytes"] else "-"
        print(f"    {r['model']:<40}{r['bits']:>5}{r['seeds'] or '-':>6}{tt:>16}{i2:>10}{r['weight_bytes'] / 1024:>12.1f}"
              f"{tb:>10}  {'  *' if r['pareto'] else ''}")
    print("\n    test acc: mean +- sd over the seeds | int 2000: the integer model on the 2,000 exported test images"
          "\n    weights KB: packed weights (bits x weights) | total KB: everything the model file needs"
          "\n    * Pareto (on int 2000, the one measure every row has): no other row has fewer weight bytes and at"
          " least the same accuracy")
    if len(fp32) > 1:
        print(f"    [!] the FP32 references differ ({sorted(fp32)}): the rows did not all start from the same FP32 file")
    os.makedirs(os.path.join("results", "lowbit"), exist_ok=True)
    json.dump(dict(epochs=EPOCHS, seeds=args.seeds, distill=not SUFFIX, fp32_sha256=sha, fp32_test=sorted(fp32),
                   rows=rows), open(os.path.join("results", "lowbit", f"lowbit_study{SUFFIX}.json"), "w"), indent=1)
    print(f"    -> results/lowbit/lowbit_study{SUFFIX}.json")


if __name__ == "__main__":
    main()
