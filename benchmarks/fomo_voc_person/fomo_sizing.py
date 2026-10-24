"""
fomo_sizing.py -- the ideal shape of the binary FOMO detector: blocks and channels per stage, from the data.

    python fomo_sizing.py --phase depth                  # 1 or 2 blocks per stage (8 variants, 32-64-128 channels)
    python fomo_sizing.py --phase width                  # 8/16/32, 16/32/64, 32/64/128 channels (27 variants, 1-2-2)
    python fomo_sizing.py --phase width --blocks 1-1-2
    python fomo_sizing.py --show                         # only the table of what is already trained

Run from benchmarks/fomo_voc_person/ (main environment). The same study as cnn_cifar_resnet8/resnet8_sizing.py:
    accuracy    every variant trained by fomo_train.py (--weight 40, --epochs 100: the recipe of the measured binary
                FOMO), scored by the test F1 with its calibrated threshold
    latency     PREDICTED: every layer costs (outputs) x (the chip's cycles per output for its packed window size,
                tools/count_bench.py), times one factor per chip calibrated with the measured binary FOMO
                (32-64-128 / 1-2-2: --s3-ms 160.8, --esp32-ms 296.2)
    flash, RAM  computed as the CDT exporter lays them out (packed weight rows, thresholds, counts; the four map buffers
                each as large as what they hold, the window, the scratch, the cell grid)
Then the interesting variants are exported and measured as the main one (fomo_export.py, tools/fine_profile.py).
Results: results/fomo_sizing.json.
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
BASE = "32-64-128/1-2-2"
METHOD = {"s3": ("pie", "xnor"), "esp32": ("lut16", "and")}
LEVELS, SIZE = 8, 96


def variants(phase, blocks):
    if phase == "depth":
        return [f"32-64-128/{a}-{b}-{c}" for a, b, c in itertools.product((1, 2), repeat=3)]
    return [f"{a}-{b}-{c}/{blocks}" for a, b, c in itertools.product((8, 16, 32), (16, 32, 64), (32, 64, 128))]


def parse(spec):
    ch, bl = spec.split("/")
    return tuple(int(v) for v in ch.split("-")), tuple(int(v) for v in bl.split("-"))


def layers(spec):
    """the conv layers of BoolFOMO: (cin, cout, k, in_h, out_h); stem 3x3 stride 2, stages at strides 1, 2, 2"""
    ch, bl = parse(spec)
    L = [(3 * LEVELS, ch[0], 3, SIZE, SIZE // 2)]
    cin, h = ch[0], SIZE // 2
    for c, s, n in zip(ch, (1, 2, 2), bl):
        for j in range(n):
            st = s if j == 0 else 1
            oh = h // st
            L.append((cin, c, 3, h, oh))
            if st != 1 or cin != c:
                L.append((cin, c, 1, oh, oh))                    # the shortcut 1x1 (after an OR pool when st = 2)
            cin, h = c, oh
    return L, cin, h


def memory(spec):
    """flash and RAM bytes as compiler_core/export/boolean.py lays them out"""
    L, ch, fh = layers(spec)
    flash = 0
    for cin, cout, k, _, _ in L:
        rw = -(-(-(-(k * k * cin) // 32)) // 4) * 4
        flash += cout * rw * 4 + cout * 4 + cout * k * k * 2
    ppx = (ch + 31) // 32
    fc_rw = -(-ppx // 4) * 4
    flash += 2 * fc_rw * 4 + 2 * 12
    words = lambda h, c: h * h * ((c + 31) // 32)
    size = [words(SIZE, 3 * LEVELS), words(SIZE // 2, L[0][1]), 0, 0]
    ch_, bl = parse(spec)
    x, y, p, s = 1, 0, 2, 3
    cin, h = ch_[0], SIZE // 2
    for c, st0, n in zip(ch_, (1, 2, 2), bl):
        for j in range(n):
            st = st0 if j == 0 else 1
            oh = h // st
            size[y] = max(size[y], words(oh, c))
            if st != 1 or cin != c:
                if st == 2:
                    size[p] = max(size[p], words(h // 2, cin))
                size[s] = max(size[s], words(oh, c))
            x, y = y, x
            cin, h = c, oh
    size[x] = max(size[x], fh * fh * ppx + fc_rw)
    size = [max(-(-v // 4) * 4, 4) for v in size]
    max_rw = max(-(-(-(-(k * k * cin) // 32)) // 4) * 4 for cin, _, k, _, _ in L)
    max_out = max(max(c for _, c, _, _, _ in L), 2)
    ram = 4 * (sum(size) + max_rw + 2 * max_out) + fh * fh
    return flash, ram


def curve(chip):
    prof = json.load(open(os.path.join(PROFILES, f"count_{chip}.json")))
    method, form = METHOD[chip]
    pts = sorted((r["words"], r["per_output"]) for r in prof["measurements"] if r["method"] == method and r["form"] == form)
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def cycles(spec, cv):
    w, c = cv
    total = 0.0
    for cin, cout, k, _, oh in layers(spec)[0]:
        total += cout * oh * oh * float(np.interp(-(-(k * k * cin) // 32), w, c))
    return total


def run_dir(spec, weight, epochs, seed=42):
    ch, bl = spec.split("/")
    return os.path.join("runs", f"c{ch}_b{bl}_w{weight:g}_e{epochs}_s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["depth", "width"], default="depth")
    ap.add_argument("--blocks", default="1-2-2", help="blocks per stage for the width phase (default 1-2-2)")
    ap.add_argument("--weight", type=float, default=40.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--s3-ms", type=float, default=160.8, help="measured S3 latency of the base variant 32-64-128/1-2-2")
    ap.add_argument("--esp32-ms", type=float, default=296.2, help="measured ESP32 (lut16) latency of the base variant")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    os.chdir(HERE)
    specs = variants(args.phase, args.blocks)
    if BASE not in specs:
        specs = [BASE] + specs
    if not args.show:
        for i, spec in enumerate(specs):
            d = run_dir(spec, args.weight, args.epochs)
            if os.path.exists(os.path.join(d, "train.json")):
                print(f"[*] {i + 1}/{len(specs)} {spec}: already trained ({d})")
                continue
            ch, bl = spec.split("/")
            print(f"\n[*] {i + 1}/{len(specs)} {spec}")
            subprocess.run([sys.executable, "fomo_train.py", "--channels", ch, "--blocks", bl, "--weight", f"{args.weight:g}",
                            "--epochs", str(args.epochs)])
    cv = {chip: curve(chip) for chip in METHOD}
    factor = {"s3": args.s3_ms / (cycles(BASE, cv["s3"]) / 240e3), "esp32": args.esp32_ms / (cycles(BASE, cv["esp32"]) / 240e3)}
    rows = []
    for spec in specs:
        tj = os.path.join(run_dir(spec, args.weight, args.epochs), "train.json")
        if not os.path.exists(tj):
            continue
        info = json.load(open(tj))
        flash, ram = memory(spec)
        rows.append(dict(spec=spec, f1=info["calibrated_test"]["f1"], f1_default=info["final"]["f1"],
                         s3_ms=factor["s3"] * cycles(spec, cv["s3"]) / 240e3,
                         esp32_ms=factor["esp32"] * cycles(spec, cv["esp32"]) / 240e3, flash_kb=flash / 1024, ram_kb=ram / 1024))
    for r in rows:
        r["pareto"] = not any(o["f1"] > r["f1"] and o["s3_ms"] < r["s3_ms"] for o in rows if o is not r)
    b = next((r for r in rows if r["spec"] == BASE), None)
    print(f"\n    SIZING THE BINARY FOMO  ({args.phase}; {args.epochs} epochs, person weight {args.weight:g}; F1 with the calibrated"
          f" threshold, VOC 2007 test persons;\n    latency PREDICTED from the count_bench profiles, calibrated with the measured"
          f" binary FOMO {BASE})\n")
    print(f"    {'channels / blocks':<20}{'F1':>8}{'vs base':>9}{'S3 ms':>9}{'ESP32 ms':>10}{'vs base':>9}{'flash KB':>10}"
          f"{'RAM KB':>9}  Pareto")
    print("    " + "-" * 92)
    for r in sorted(rows, key=lambda r: r["s3_ms"]):
        print(f"    {r['spec']:<20}{r['f1']:>7.1f}%{r['f1'] - (b['f1'] if b else 0):>+8.1f} {r['s3_ms']:>8.1f}{r['esp32_ms']:>10.1f}"
              f"{(b['s3_ms'] / r['s3_ms'] if b else 1):>8.2f}x{r['flash_kb']:>10.1f}{r['ram_kb']:>9.1f}  "
              f"{'  *' if r['pareto'] else ''}{'   <- the benchmark FOMO' if r['spec'] == BASE else ''}")
    print("\n    * Pareto: no other variant detects better AND runs faster. Export and measure the interesting ones:"
          "\n      python fomo_export.py runs/<run>  ;  python ../../tools/fine_profile.py runs/<run> --board s3 --plain")
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "fomo_sizing.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out[f"{args.phase}_{args.blocks if args.phase == 'width' else 'all'}"] = dict(epochs=args.epochs, weight=args.weight,
                                                                                   calibration=factor, variants=rows)
    json.dump(out, open(path, "w"), indent=1)
    print(f"    -> {path}")


if __name__ == "__main__":
    main()
