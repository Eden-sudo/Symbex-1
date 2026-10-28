"""
fashion_symbex.py -- mlp_fashion benchmark, SYMBEX side (main environment, .venv).

    python fashion_symbex.py                                     # seed 42: FP32 model (if missing) + CDT up to the export
    python fashion_symbex.py --seeds 3 --board s3                # seeds 42, 43, 44: CDT + the board, each
    python fashion_symbex.py --board s3 --compare-arithmetic     # also the whole model with the other arithmetic
    python fashion_symbex.py --board esp32 --sweep               # every chip setting on the first seed's export

Run from benchmarks/mlp/mlp_fashion/.
  1. models/fashion_mlp.pth: the original FP32 model (trained by model.py if missing). Every seed starts from this
     SAME model: the seed only changes the CDT (1-bit training).
  2. for every seed, the CDT (tools/cdt.py symbex.toml): up to the export, in runs/symbex_s<seed>/ (skipped if that
     seed is already exported; --redo trains it again)
  3. with --board: the CDT again on that export, on the board, with the weights in RAM or flash (--weights), the
     popcount mode (--popcount) and, with --compare-arithmetic, both arithmetics
  4. --sweep: on the first seed's export, every combination of weights (flash, ram) and popcount (swar, lut16),
     each with both arithmetics, sorted by latency
Summary tables at the end; every result .json is copied to results/symbex/ (the only outputs that are published).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CDT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "cdt.py"))


def run_cdt(*flags):
    cmd = [sys.executable, CDT, "symbex.toml", *[str(f) for f in flags]]
    print("\n[*] " + " ".join(os.path.relpath(c) if c == CDT else c for c in cmd[1:]))
    if subprocess.run(cmd).returncode != 0:
        sys.exit("[X] the CDT failed")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.2f} +- {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def chip_on(out_dir, seed, args, weights, popcount, compare, model_flags=()):
    """The CDT on an existing training, always from the EXPORT stage (seconds, no training): the export depends on
    the board (with [output] arithmetic = "auto" the S3 gets AND and the Uno or the ESP32 may get XNOR), so the
    header left in the run folder by another board must not be reused. Returns (main result, other-arithmetic result)."""
    for old in glob.glob(os.path.join(out_dir, "chip*.json")):   # results of an earlier board must not be read as these
        os.remove(old)
    flags = ["--seed", seed, "--output-directory", out_dir, "--from", "export", "--board", args.board,
             "--port", args.port, "--weights", weights, "--popcount", popcount,
             # the same data settings as the training of this export (re-exporting needs them: --compare-arithmetic)
             "--input-bits", args.input_bits, *(["--widths", args.widths] if args.widths else []), *model_flags]
    run_cdt(*flags, *(["--compare-arithmetic"] if compare else []))
    main = load(os.path.join(out_dir, "chip.json"))
    other = None
    if compare and main:
        other = load(os.path.join(out_dir, f"chip_{'xnor' if main.get('arithmetic') == 'and' else 'and'}.json"))
    return main, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds: 42, 43, ... (default 1)")
    ap.add_argument("--board", choices=["esp32", "s3"], default="", help="measure on this board")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--weights", choices=["flash", "ram"], default="ram", help="where the weights live on the board")
    ap.add_argument("--popcount", choices=["swar", "lut16"], default="swar")
    ap.add_argument("--compare-arithmetic", action="store_true", help="also measure the other arithmetic (AND / XNOR)")
    ap.add_argument("--sweep", action="store_true", help="every chip setting on the first seed's export")
    ap.add_argument("--redo", action="store_true", help="run the CDT again even if a seed is already exported")
    ap.add_argument("--widths", default="", help="override [target] widths (e.g. 1 or 1,1.5,2)")
    ap.add_argument("--input-bits", type=int, default=8, help="bits per pixel the model uses (1-8)")
    ap.add_argument("--fp32-epochs", type=int, default=None,
                    help="epochs of the original FP32 model (default model.EPOCHS: models/fashion_mlp.pth); other values "
                         "train and use models/fashion_mlp_e<N>.pth (quick iterations, or a stronger teacher)")
    ap.add_argument("--no-distill", action="store_true",
                    help="train without distillation from the FP32 model (plain cross-entropy)")
    args = ap.parse_args()
    if args.port and not args.port.startswith("/") and not args.port.upper().startswith("COM"):
        args.port = "/dev/" + args.port
    if args.sweep and not args.board:
        sys.exit("[X] --sweep needs --board")
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    from model import get_fp32
    print("[*] original FP32 model (trained if missing)")
    from model import EPOCHS
    other = args.fp32_epochs not in (None, EPOCHS)            # another FP32 than the published one
    fp32_path = get_fp32(epochs=args.fp32_epochs) if other else get_fp32()
    model_flags = ["--file", fp32_path] if other else []
    os.makedirs(os.path.join("results", "symbex"), exist_ok=True)
    seeds = [42 + i for i in range(args.seeds)]
    suffix = ("" if args.input_bits == 8 else f"_b{args.input_bits}") + (f"_w{args.widths.replace(',', '-')}" if args.widths else "") \
        + ("_nd" if args.no_distill else "") + (f"_f{args.fp32_epochs}" if other else "")

    rows = []
    for seed in seeds:
        out_dir = f"runs/symbex_s{seed}{suffix}"
        done = os.path.exists(os.path.join(out_dir, "train.json")) and os.path.exists(os.path.join(out_dir, "model_symbex.h"))
        if done and not args.redo:
            print(f"\n[*] seed {seed}: already exported in {out_dir} (--redo to train it again)")
        else:
            # with a board, the export already chooses the arithmetic for it ([output] arithmetic = "auto": chip profile)
            run_cdt("--seed", seed, "--output-directory", out_dir, "--to", "export", "--input-bits", args.input_bits,
                    *(["--no-distill"] if args.no_distill else []), *model_flags,
                    *(["--widths", args.widths] if args.widths else []),
                    *(["--board", args.board, "--port", args.port, "--popcount", args.popcount] if args.board else []))
        tr, ex = load(os.path.join(out_dir, "train.json")), load(os.path.join(out_dir, "export.json"))
        row = dict(seed=seed, width=tr["width"], val_accuracy=tr["acc_last"], flash_kb=ex["flash_bytes"] / 1024,
                   arithmetic=ex.get("arithmetic"))
        if args.board and not args.sweep:
            chip, other = chip_on(out_dir, seed, args, args.weights, args.popcount, args.compare_arithmetic, model_flags)
            if chip:
                # the board run re-exports (its own arithmetic): the flash of THAT export, not of the one read above
                row["flash_kb"] = load(os.path.join(out_dir, "export.json"))["flash_bytes"] / 1024
                row.update(chip_accuracy=chip["accuracy"], latency_ms=chip["median_ms"], equal=chip["equal_simulation"],
                           ram_kb=chip["ram"]["total"] / 1024, arithmetic=chip.get("arithmetic"),
                           other_latency_ms=other["median_ms"] if other else None)
        rows.append(row)
        tag = f"_s{seed}{suffix}"
        for f in glob.glob(os.path.join(out_dir, "*.json")):
            name = os.path.basename(f)
            if name.startswith("chip"):
                name = name.replace("chip", f"chip_{args.board}_{args.weights}_{args.popcount}", 1)
            shutil.copy(f, os.path.join("results", "symbex", f"symbex{tag}_{name}"))

    on_chip = args.board and not args.sweep
    print(f"\n    SEEDS  (same FP32 original; validation accuracy on the 10,000 Fashion-MNIST test images, last epoch)")
    print(f"    {'seed':>5}{'width':>7}{'val acc':>10}{'flash KB':>10}"
          + (f"{'arith.':>8}{'chip acc':>10}{'ms':>9}{'equal':>8}{'RAM KB':>9}" if on_chip else ""))
    for r in rows:
        print(f"    {r['seed']:>5}{r['width']:>6g}x{r['val_accuracy']:>9.2f}%{r['flash_kb']:>10.1f}"
              + (f"{str(r.get('arithmetic')).upper():>8}{r['chip_accuracy']:>9.2f}%{r['latency_ms']:>9.3f}"
                 f"{r['equal']:>7.0f}%{r['ram_kb']:>9.2f}" if on_chip and "chip_accuracy" in r else ""))
    summary = dict(board=args.board, weights=args.weights, popcount=args.popcount, input_bits=args.input_bits,
                   seeds=rows, val_accuracy=mean_std([r["val_accuracy"] for r in rows]))
    print(f"    validation accuracy over {len(rows)} seed(s): {summary['val_accuracy']} %")
    if on_chip and rows and "latency_ms" in rows[0]:
        summary["latency_ms"] = mean_std([r["latency_ms"] for r in rows])
        summary["chip_accuracy"] = mean_std([r["chip_accuracy"] for r in rows])
        print(f"    on the board: latency {summary['latency_ms']} ms | accuracy {summary['chip_accuracy']} %")
    path = f"results/symbex/symbex_seeds_{args.board or 'pc'}{suffix}.json"
    json.dump(summary, open(path, "w"), indent=1)
    print(f"    -> {path}")

    if args.sweep:
        out_dir, seed = f"runs/symbex_s{seeds[0]}{suffix}", seeds[0]
        results = []
        for weights in ("flash", "ram"):
            for popcount in ("swar", "lut16"):
                main_r, other_r = chip_on(out_dir, seed, args, weights, popcount, True, model_flags)
                for r in (main_r, other_r):
                    if r:
                        results.append(dict(arithmetic=r.get("arithmetic"), weights=weights, popcount=popcount,
                                            ms=r["median_ms"], layer0_ms=r["layers"][0]["median_ms"],
                                            ram_kb=r["ram"]["total"] / 1024, equal=r["equal_simulation"]))
        results.sort(key=lambda r: r["ms"])
        print(f"\n    CHIP SETTINGS ({args.board}, seed {seed}), fastest first")
        print(f"    {'arithmetic':<12}{'weights':<9}{'popcount':<10}{'total ms':>10}{'layer 0 ms':>12}{'RAM KB':>9}{'equal':>8}")
        for r in results:
            print(f"    {str(r['arithmetic']).upper():<12}{r['weights']:<9}{r['popcount']:<10}{r['ms']:>10.3f}"
                  f"{r['layer0_ms']:>12.3f}{r['ram_kb']:>9.2f}{r['equal']:>7.0f}%")
        path = f"results/symbex/symbex_sweep_{args.board}{suffix}.json"
        json.dump(results, open(path, "w"), indent=1)
        print(f"    -> {path}")


if __name__ == "__main__":
    main()
