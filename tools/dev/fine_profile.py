"""
fine_profile.py -- where the time goes on the chip, step by step AND function by function of lib_symbex.

    python fine_profile.py ../benchmarks/cnn_cifar_resnet8/runs/w1_s44 --board s3 --port /dev/ttyUSB0
    python fine_profile.py <export folder> --board esp32 --popcount lut16 --weights ram
    python fine_profile.py <export folder> --board s3 --plain      # no counters: the latency to report, per step

Takes any export of a fully binary network made by the CDT (method "boolean": the folder with model_symbex.h and
validation_x.u8), builds the library with -DSYMBEX_FINE_PROFILE (CPU cycle counters inside every part of the layers;
without the flag they do not exist) and the firmware chip/firmware/fine_profile.c, sends --images validation images
and prints:
    STEPS       the time of every step of the network (as the chip stage of the CDT)
    FUNCTIONS   one block per part of the library: ms per image, share of the time, calls per image, cycles per call
                (thermometer, window copies, popcount of the window, the HAL's count of a whole layer, thresholds,
                OR pool, OR shortcut, classifier), and what is not inside any of them (loops, calls, the counters)
The counters cost a few cycles each (measured on the chip and subtracted): this latency is not the one to report;
the shares are. Normal measurements (cdt.py, chip_test.py) never include the counters.
"""
import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # tools/: compiler_core
from compiler_core.chip.core import BOARDS, flash, lib_sources, open_port, read_line, wait_ready, write_project  # noqa
from compiler_core.chip.measure import _build_flags                                                                  # noqa

MHZ = 240


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="export folder of a boolean network (model_symbex.h, validation_x.u8)")
    ap.add_argument("--board", choices=["esp32", "s3"], required=True)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--images", type=int, default=100, help="validation images (default 100: the shares do not change)")
    ap.add_argument("--popcount", choices=["swar", "lut16"], default="swar")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash")
    ap.add_argument("--lib", default=os.path.normpath(os.path.join(HERE, "..", "..", "Symbex1_PIO", "lib", "lib_symbex")))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--plain", action="store_true", help="the normal library (no counters): the latency to report")
    args = ap.parse_args()
    header = os.path.join(args.export, "model_symbex.h")
    xs = os.path.join(args.export, "validation_x.u8")
    for p in (header, xs):
        if not os.path.exists(p):
            sys.exit(f"[X] {p} not found: give the folder of an export")
    text = open(header).read()
    if "symbex_bool_net_run" not in text:
        sys.exit("[X] this is not a fully binary network (method 'boolean'): the fine profile instruments layers/conv/symbex_bool")
    n_in = int(re.search(r"#define SYMBEX_MLP_N_INPUTS (\d+)", text).group(1))
    data = open(xs, "rb").read()
    n = min(args.images, len(data) // n_in)

    proj = os.path.join(args.export, f"{'plain' if args.plain else 'fine'}_profile_{args.board}")
    if not args.no_upload:
        write_project(proj, args.board, args.port, lib_sources(args.lib), "fine_profile.c",
                      defines={"BAUD_RATE": args.baud}, extra_files={"model_symbex.h": header},
                      build_flags=list(_build_flags(args.popcount, args.weights, "mask"))
                      + ([] if args.plain else ["-DSYMBEX_FINE_PROFILE"]),
                      lib_path=args.lib)
        flash(proj, what=f"the fine profile to the {args.board}")
    s = open_port(args.port, args.baud)
    ready = wait_ready(s, "PROFILE ready")
    steps = {}
    while True:
        line = read_line(s)
        if line.startswith("STEP:"):
            i, name = line[5:].split("|", 1)
            steps[int(i)] = name
        elif line.startswith("STEPS:"):
            continue
        if line == "" and steps:
            break
    print(f"    board ready | {ready.split('|', 1)[1].strip() if '|' in ready else ''}")
    totals = []
    for i in range(n):
        s.write(b"\xA5" + data[i * n_in:(i + 1) * n_in])
        t0 = time.time()
        while True:
            line = read_line(s)
            if line.startswith("R:"):
                totals.append(int(line[2:].split(",")[1]))
                break
            if time.time() - t0 > 30:
                sys.exit("[X] the board stopped answering")
        if (i + 1) % 25 == 0:
            print(f"      {i + 1}/{n} images")
    s.write(b"\x77")
    pf, st, imgs, cost = [], {}, 0, 0
    while True:
        line = read_line(s)
        if line.startswith("PF_IMG:"):
            imgs = int(line[7:])
        elif line.startswith("PF_COST:"):
            cost = int(line[8:])
        elif line.startswith("PF:"):
            name, cyc, calls = line[3:].rsplit("|", 2)
            pf.append((name, int(cyc), int(calls)))
        elif line.startswith("ST:"):
            i, us = line[3:].split("|")
            st[int(i)] = int(us)
        elif line == "PF_END":
            break
    s.close()
    imgs = max(imgs, 1)
    total_ms = sum(st.values()) / imgs / 1000

    totals_ms = sorted(totals)[len(totals) // 2] / 1000 if totals else 0.0
    if args.plain:
        print(f"\n    LATENCY  ({args.board}, {MHZ} MHz, {imgs} images, popcount {args.popcount}, weights in {args.weights}; "
              f"normal library, no counters): median {totals_ms:.3f} ms")
    else:
        print(f"\n    FINE PROFILE  ({args.board}, {MHZ} MHz, {imgs} images, popcount {args.popcount}, weights in {args.weights};"
              f" counters cost {cost} cycles each, subtracted)")
    print(f"\n    STEPS{'':45}{'ms/img':>9}{'%':>7}")
    print("    " + "-" * 62)
    for i in sorted(steps):
        ms = st.get(i, 0) / imgs / 1000
        print(f"    {steps[i][:48]:<50}{ms:>9.3f}{100 * ms / total_ms:>6.1f}%  " + "#" * int(round(25 * ms / total_ms)))
    print("    " + "-" * 62)
    print(f"    {'total' + ('' if args.plain else ' (with the counters)'):<50}{total_ms:>9.3f}")
    if args.plain:
        return

    print(f"\n    FUNCTIONS OF THE LIBRARY")
    measured = 0.0
    for name, cyc, calls in pf:
        if not calls:
            continue
        net = max(cyc - calls * cost, 0)
        ms = net / imgs / (MHZ * 1e3)
        measured += ms
        print(f"\n    {name}")
        print(f"      ms per image     : {ms:9.3f}   ({100 * ms / total_ms:.1f}% of the time)  " + "#" * int(round(25 * ms / total_ms)))
        print(f"      calls per image  : {calls / imgs:9.0f}")
        print(f"      cycles per call  : {net / calls:9.1f}")
    rest = max(total_ms - measured, 0.0)
    print(f"\n    not inside any of them (loops, calls, the counters themselves)")
    print(f"      ms per image     : {rest:9.3f}   ({100 * rest / total_ms:.1f}% of the time)")


if __name__ == "__main__":
    main()
