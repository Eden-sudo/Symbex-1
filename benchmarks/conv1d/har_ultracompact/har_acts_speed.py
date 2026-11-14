"""
har_acts_speed.py -- har_ultracompact benchmark: about how fast would the two-bit / ternary networks of README section 9
run on the chips? An ESTIMATE from real measurements, until lib_symbex has a 2-bit / ternary kernel.

    python har_acts_speed.py export                 # the proxy networks: runs/speed_proxy/<arch>_<first>_x2/
    (chip_test.py on each proxy folder, the same flags as the 1-bit runs, see README section 10)
    python har_acts_speed.py report                 # the estimate per net x first layer x board, and the real
                                                    # two-bit / ternary networks where they were measured

Run from benchmarks/conv1d/har_ultracompact/ (main environment). The idea: a 2-bit (or ternary) activation is two bit
planes, so a binary layer that reads it does the work of a binary layer with twice the input channels and the same
outputs. The chips cannot run that directly (no kernel), so:
    proxy       the same network with every conv twice as wide (random 1-bit weights: the time does not depend on them),
                exported with the CDT's 1D exporter, checked exact on the PC and measured on the chip (per-step times)
    upper       the same with every later conv of the proxy whole (it has twice the outputs): an upper bound, nearer
                the truth on the small networks, where a layer's fixed cost (calls, thresholds, pools) dominates
    estimate    first conv: the 1-bit network's time (the same work: int8 / uint8 input, only 2-3 thresholds per
                channel instead of 1); every later conv: half of the proxy's (twice the inputs, but the proxy also has
                twice the outputs); classifier: the proxy's (twice the features, the same 17 classes)
It leaves out the extra thresholds (3 per output for 2 bits, 2 for ternary, instead of 1) and the pools on two planes
(halved with the conv): the real kernel would be a little slower than the estimate, or faster if it counted both
planes in one pass. For reference, TFLite Micro INT8 of the same net (seed 42, flash). Results: results/symbex/har_acts_speed.json (times in ms, median over the windows).
Measured (README section 10): the networks chosen on validation in section 10 (MEASURED below, seed 42), exported with
har_export.py (acts1d.py, kernel planes1d.h) and run with chip_test.py, the same flags as the 1-bit runs.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools"))
LIB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Symbex1_PIO", "lib", "lib_symbex"))
sys.path.insert(0, HERE)
sys.path.insert(1, TOOLS)
from compiler_core.export import bool1d                                                       # noqa: E402
from compiler_core.export.dense import save_validation                                        # noqa: E402
from har_data import load                                                                     # noqa: E402
from model import ARCHS, AXES, LENGTH, TABLE1, BinaryHAR                                      # noqa: E402

PROXY = os.path.join(HERE, "runs", "speed_proxy")
# the 1-bit chip runs the estimate starts from (seed 42, weights in flash, the popcount each board's README row uses)
BOARDS = {"esp32": "lut16", "s3": "swar", "pico2_arm": "lut16", "pico2_riscv": "lut16"}
# the two-bit / ternary network per architecture chosen on validation in README section 10 (int8 first layer)
MEASURED = {"min": "min_int8_e100_a2_aug_s42", "max10": "max10_int8_e100_ter_s42", "max5": "max5_int8_e100_ter_s42",
            "max": "max_int8_e100_a2_s42"}


def export(arch, first, images=1000):
    layers = [(l[0], 2 * l[1], l[2]) if l[0] == "conv" else l for l in ARCHS[arch]]
    torch.manual_seed(0)
    model = BinaryHAR(first=first, layers=layers).eval()
    net = bool1d.fold(model)
    x, y = load("validation")
    n = min(images, len(x))
    x_u8 = (x[:n].astype(np.int16) + 128).astype(np.uint8).reshape(n, -1)
    sim = bool1d.simulate(net, x[:n])
    out = os.path.join(PROXY, f"{arch}_{first}_x2")
    os.makedirs(out, exist_ok=True)
    save_validation(out, x_u8, y[:n], sim)
    header = os.path.join(out, "model_symbex.h")
    flash, ram, steps, shapes = bool1d.write_header(net, header, LENGTH, AXES)
    equal, total, msg = bool1d.validate_c(header, LIB, x_u8, sim)
    print(f"    {arch:<6} {first:<6} x2: {flash / 1024:.2f} KB, exact on the PC "
          + (f"{100 * equal / total:.2f}%" if equal is not None else f"not checked ({msg})") + f" -> {out}")
    json.dump(dict(arch=arch, first=first, layers=layers, flash_bytes=flash, ram_buffers=ram, steps=steps,
                   exactness=None if equal is None else 100 * equal / total), open(os.path.join(out, "export.json"), "w"),
              indent=1)


def chip(folder, board):
    path = os.path.join(folder, f"chip_{board}_{BOARDS[board]}_flash_mask.json")
    return json.load(open(path)) if os.path.exists(path) else None


def report(out):
    rows = []
    for arch in TABLE1:
        for first in ("binary", "int8"):
            for board in BOARDS:
                one = chip(os.path.join(HERE, "runs", f"{arch}_{first}_e100_s42"), board)
                two = chip(os.path.join(PROXY, f"{arch}_{first}_x2"), board)
                if one is None or two is None:
                    continue
                t1 = [s["median_ms"] for s in one["layers"]]
                t2 = [s["median_ms"] for s in two["layers"]]
                est = t1[0] + sum(t / 2 for t in t2[1:-1]) + t2[-1]
                high = t1[0] + sum(t2[1:-1]) + t2[-1]               # the proxy's later convs whole: an upper bound
                tfl = os.path.join(HERE, "results", "tflite", f"tflite_{board}_{arch}_s42.json")   # INT8, flash
                tfl = json.load(open(tfl))["median_ms"] if os.path.exists(tfl) else None
                rows.append(dict(arch=arch, first=first, board=board, popcount=BOARDS[board], one_bit_ms=one["median_ms"],
                                 proxy_x2_ms=two["median_ms"], estimate_ms=est, ratio=est / one["median_ms"],
                                 upper_ms=high, upper_ratio=high / one["median_ms"], tflite_int8_ms=tfl,
                                 one_bit_steps=t1, proxy_steps=t2, proxy_equal_simulation=two["equal_simulation"]))
                print(f"    {arch:<6} {first:<6} {board:<12} 1-bit {one['median_ms']:7.3f} ms | proxy x2 "
                      f"{two['median_ms']:7.3f} | 2-bit ESTIMATE {est:7.3f} ms ({est / one['median_ms']:.2f}x), at most "
                      f"{high:7.3f} ({high / one['median_ms']:.2f}x) | TFLite INT8 "
                      + (f"{tfl:7.3f}" if tfl else "   -   ") + " | "
                      f"proxy = simulation {two['equal_simulation']:.1f}%")
    measured = []
    for arch, run in MEASURED.items():
        for board in BOARDS:
            real = chip(os.path.join(HERE, "runs", run), board)
            one = chip(os.path.join(HERE, "runs", f"{arch}_int8_e100_s42"), board)
            if real is None or one is None:
                continue
            est = next((r for r in rows if r["arch"] == arch and r["first"] == "int8" and r["board"] == board), None)
            measured.append(dict(arch=arch, run=run, board=board, popcount=BOARDS[board], ms=real["median_ms"],
                                 one_bit_ms=one["median_ms"], ratio=real["median_ms"] / one["median_ms"],
                                 estimate_ms=est and est["estimate_ms"], upper_ms=est and est["upper_ms"],
                                 tflite_int8_ms=est and est["tflite_int8_ms"],
                                 steps=[s["median_ms"] for s in real["layers"]],
                                 equal_simulation=real["equal_simulation"]))
            m = measured[-1]
            print(f"    MEASURED {arch:<6} {board:<12} {m['ms']:7.3f} ms ({m['ratio']:.2f}x the 1-bit {m['one_bit_ms']:.3f}) | "
                  f"estimate {m['estimate_ms'] or 0:7.3f}, at most {m['upper_ms'] or 0:7.3f} | TFLite INT8 "
                  + (f"{m['tflite_int8_ms']:7.3f}" if m["tflite_int8_ms"] else "   -   ")
                  + f" | = simulation {m['equal_simulation']:.1f}%")
    json.dump(dict(note="ESTIMATE: first conv of the 1-bit run + half of every later conv of the x2 proxy + the proxy's "
                        "classifier; extra thresholds and two-plane pools left out. measured: the real two-bit / ternary "
                        "networks (acts1d.py + planes1d.h), seed 42, weights in flash", rows=rows, measured=measured),
              open(out, "w"), indent=1)
    print(f"[*] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["export", "report"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "symbex", "har_acts_speed.json"))
    args = ap.parse_args()
    if args.what == "export":
        for arch in TABLE1:
            for first in ("binary", "int8"):
                export(arch, first)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
