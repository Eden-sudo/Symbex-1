"""
resnet8_sizing.py -- does a binary network need its own sizing? The same structure as the ResNet-8, with other
numbers of blocks and channels per stage, trained by the same CDT; which ones give more accuracy for less time.

    python resnet8_sizing.py --phase depth               # blocks per stage: 1 or 2 (8 variants, 16-32-64 channels)
    python resnet8_sizing.py --phase width               # channels per stage: x0.5, x1, x2 (27 variants, 2-2-2 blocks)
    python resnet8_sizing.py --phase width --blocks 1-1-2
    python resnet8_sizing.py --show                      # only the table of what is already trained

Run from benchmarks/cnn_cifar_resnet8/ (main environment). For every variant ("16-32-64/2-2-2" = channels / blocks per
stage) resnet8_symbex.py --spec trains it for --epochs (default 30: a quick ranking, not the final accuracy) and exports
it (the flash and the shape of every layer come from that export). The latency is PREDICTED, without the chip:
    every layer costs (outputs) x (cycles per output of the chip's counting method for its packed window size), the
    method's cycles measured by tools/count_bench.py on each chip (chip_profiles/count_<chip>.json), and one factor per
    chip calibrated with the measured ResNet-8 (results/symbex_seeds_s3_swar_w1.json, ..._esp32_lut16_w1.json)
It is an estimate: the variants worth it are then trained in full and measured on the chip, as the main benchmark:
    python resnet8_symbex.py --spec 16-32-64/1-1-2 --seeds 3 --board s3 --port /dev/ttyUSB0
Table: accuracy (short training), predicted ms on both chips, flash, and the variants no other beats in both accuracy
and time (Pareto). Results: results/sizing.json.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.normpath(os.path.join(HERE, "..", "..", "tools", "chip_profiles"))
BASE = "16-32-64/2-2-2"
CHIPS = {"s3": ("pie", "xnor", "symbex_seeds_s3_swar_w1.json"), "esp32": ("lut16", "and", "symbex_seeds_esp32_lut16_w1.json")}


def variants(phase, blocks):
    if phase == "depth":
        return [f"16-32-64/{a}-{b}-{c}" for a, b, c in itertools.product((1, 2), repeat=3)]
    mult = (0.5, 1, 2)
    return [f"{int(16 * a)}-{int(32 * b)}-{int(64 * c)}/{blocks}" for a, b, c in itertools.product(mult, repeat=3)]


def run_dir(spec, epochs, seed=42):
    sys.path.insert(0, HERE)
    from model import spec_name
    return os.path.join("runs", f"{spec_name(spec)}_e{epochs}_s{seed}")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def cost_curve(chip):
    """cycles per output vs window words, for the method the library uses on this chip (count_bench measurements)"""
    prof = load(os.path.join(PROFILES, f"count_{chip}.json"))
    if not prof:
        sys.exit(f"[X] {PROFILES}/count_{chip}.json not found: run tools/count_bench.py --board {chip} first")
    method, form = CHIPS[chip][:2]
    pts = sorted((r["words"], r["per_output"]) for r in prof["measurements"] if r["method"] == method and r["form"] == form)
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def raw_cycles(layers, curve):
    w, c = curve
    total = 0.0
    for L in layers:
        words = -(-(L["k"] * L["k"] * L["cin"]) // 32)
        outputs = L["cout"] * L["out_h"] * L["out_h"]
        total += outputs * float(np.interp(words, w, c))
    return total


def measured_ms(chip):
    r = load(os.path.join("results", CHIPS[chip][2]))
    if not r:
        return None
    ms = [s["latency_ms"] for s in r["seeds"] if s.get("latency_ms")]
    return float(np.mean(ms)) if ms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["depth", "width"], default="depth")
    ap.add_argument("--blocks", default="2-2-2", help="blocks per stage for the width phase (default 2-2-2)")
    ap.add_argument("--epochs", type=int, default=30, help="quick training per variant (default 30)")
    ap.add_argument("--show", action="store_true", help="only print the table of what is already trained")
    args = ap.parse_args()
    os.chdir(HERE)
    specs = variants(args.phase, args.blocks)
    if BASE not in specs:
        specs = [BASE] + specs
    if not args.show:
        for i, spec in enumerate(specs):
            d = run_dir(spec, args.epochs)
            if os.path.exists(os.path.join(d, "export.json")):
                print(f"[*] {i + 1}/{len(specs)} {spec}: already trained and exported ({d})")
                continue
            print(f"\n[*] {i + 1}/{len(specs)} {spec}: {args.epochs} epochs")
            subprocess.run([sys.executable, "resnet8_symbex.py", "--spec", spec, "--epochs", str(args.epochs)])

    curves = {chip: cost_curve(chip) for chip in CHIPS}
    base = load(os.path.join(run_dir(BASE, args.epochs), "export.json"))
    if not base:
        sys.exit(f"[X] the base variant {BASE} has no export yet (run without --show)")
    factor = {}
    for chip in CHIPS:
        ms = measured_ms(chip)
        if ms is None:
            sys.exit(f"[X] results/{CHIPS[chip][2]} not found: measure the ResNet-8 on the {chip} first (resnet8_symbex.py)")
        factor[chip] = ms / (raw_cycles(base["layers"], curves[chip]) / 240e3)
    rows = []
    for spec in specs:
        d = run_dir(spec, args.epochs)
        tr, ex = load(os.path.join(d, "train.json")), load(os.path.join(d, "export.json"))
        if not tr or not ex:
            continue
        pred = {chip: factor[chip] * raw_cycles(ex["layers"], curves[chip]) / 240e3 for chip in CHIPS}
        rows.append(dict(spec=spec, accuracy=tr["acc_last"], flash_kb=ex["flash_bytes"] / 1024, s3_ms=pred["s3"],
                         esp32_ms=pred["esp32"], exact=ex.get("exactness")))
    for r in rows:                                       # Pareto: nobody else more accurate AND faster on the S3
        r["pareto"] = not any(o["accuracy"] > r["accuracy"] and o["s3_ms"] < r["s3_ms"] for o in rows if o is not r)
    b = next((r for r in rows if r["spec"] == BASE), None)

    print(f"\n    SIZING THE BINARY NETWORK  ({args.phase}; {args.epochs}-epoch training per variant: a ranking, not the final"
          f" accuracy;\n    latency PREDICTED from the count_bench profiles, calibrated with the measured ResNet-8)\n")
    print(f"    {'channels / blocks':<20}{'accuracy':>10}{'vs base':>9}{'S3 ms':>9}{'ESP32 ms':>10}{'vs base':>9}"
          f"{'flash KB':>10}  {'C exact':>8}  Pareto")
    print("    " + "-" * 98)
    for r in sorted(rows, key=lambda r: r["s3_ms"]):
        da = r["accuracy"] - b["accuracy"] if b else 0.0
        sp = b["s3_ms"] / r["s3_ms"] if b else 1.0
        print(f"    {r['spec']:<20}{r['accuracy']:>9.2f}%{da:>+8.2f} {r['s3_ms']:>8.1f}{r['esp32_ms']:>10.1f}{sp:>8.2f}x"
              f"{r['flash_kb']:>10.1f}  {(str(round(r['exact'])) + '%') if r['exact'] is not None else '-':>8}  "
              f"{'  *' if r['pareto'] else ''}{'   <- the ResNet-8' if r['spec'] == BASE else ''}")
    print("\n    * Pareto: no other variant is more accurate AND faster. Confirm the interesting ones with a full training"
          "\n      and the chip:  python resnet8_symbex.py --spec <channels/blocks> --seeds 3 --board s3")
    os.makedirs("results", exist_ok=True)
    out = load(os.path.join("results", "sizing.json")) or {}
    out[f"{args.phase}_{args.blocks if args.phase == 'width' else 'all'}_e{args.epochs}"] = dict(
        epochs=args.epochs, calibration=factor, variants=rows)
    json.dump(out, open(os.path.join("results", "sizing.json"), "w"), indent=1)
    print("    -> results/sizing.json")


if __name__ == "__main__":
    main()
