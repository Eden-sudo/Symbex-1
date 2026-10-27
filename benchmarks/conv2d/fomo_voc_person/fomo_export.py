"""
fomo_export.py -- a trained binary FOMO detector (fomo_train.py) into lib_symbex, checked, ready for the chip.

    python fomo_export.py runs/c32-64-128_b1-2-2_w40_e100_s42
    python ../../../tools/dev/fine_profile.py runs/c32-64-128_b1-2-2_w40_e100_s42 --board s3 --port /dev/ttyUSB0

Run from benchmarks/conv2d/fomo_voc_person/ (main environment). With the CDT's exporter (compiler_core/export/boolean.py):
    fold        every binary conv + BatchNorm into one integer threshold; the head into an integer scale and bias per
                class, with the CALIBRATED decision threshold of train.json folded into the person bias (free on the chip)
    simulate    the integer network on the 4,952 VOC 2007 test images: its grids, and their F1 (it must match PyTorch's
                calibrated F1: FIDELITY = cells with the same class)
    header      model_symbex.h for symbex_bool_net_run with the per-cell head; the chip reports a checksum of the 12 x 12
                grid of every image, so "equal to the simulation" covers every cell
    C check     the header + lib_symbex compiled on this PC must give the same grids (EXACTNESS 100%)
Writes into the run folder: model_symbex.h, validation_x.u8 (the first --images test images), the expected checksums,
and export.json.
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
from model import BoolFOMO, f1_score, load                                                       # noqa: E402
from compiler_core.export import boolean                                                         # noqa: E402
from compiler_core.export.dense import save_validation                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a run folder of fomo_train.py (model.pth, train.json)")
    ap.add_argument("--images", type=int, default=1000, help="test images for the C check and the chip (default 1000)")
    args = ap.parse_args()
    os.chdir(HERE)
    info = json.load(open(os.path.join(args.run, "train.json")))
    if info.get("fp32"):
        sys.exit("[X] this run is the FP32 reference: only the binary detector goes to lib_symbex")
    model = BoolFOMO(tuple(info["channels"]), tuple(info["blocks"]), info["levels"])
    model.load_state_dict(torch.load(os.path.join(args.run, "model.pth"), map_location="cpu"))
    model.eval()
    th = float(info.get("threshold", 0.0))
    x, _, centers, owner = load("test")

    with torch.no_grad():                                            # PyTorch, with the calibrated threshold
        torch_grids = np.concatenate([
            ((lambda z: (z[:, 1] - z[:, 0]) > th)(model(torch.as_tensor(x[i:i + 250]).permute(0, 3, 1, 2)))).numpy()
            for i in range(0, len(x), 250)]).astype(np.uint8)
    net = boolean.fold(model, threshold=th)
    grids = boolean.simulate(net, x, return_bits=True)
    s_torch, s_int = f1_score(torch_grids, centers, owner), f1_score(grids, centers, owner)
    fid = 100 * (grids == torch_grids).mean()

    header = os.path.join(args.run, "model_symbex.h")
    flash, ram, steps, shapes = boolean.write_header(net, header, x.shape[1])
    n = min(args.images, len(x))
    checks = np.array([boolean._grid_checksum(g) for g in grids[:n]])
    equal, total, msg = boolean.validate_c(header, LIB, x[:n], checks)
    save_validation(args.run, x[:n].reshape(n, -1), checks, checks)

    print(f"\n    FOMO EXPORT  ({args.run})")
    print(f"    network     : thermometer {net['levels']} levels -> stem (stride 2) -> {len(net['blocks'])} blocks -> per-cell head "
          f"{grids.shape[1]} x {grids.shape[2]} | every layer 1 bit, no multiplication")
    print(f"    threshold   : {th:+.3f} (calibrated in fomo_train.py), folded into the person bias")
    print(f"    header      : {header}  ({flash / 1024:.1f} KB of weights + thresholds, {ram / 1024:.1f} KB of buffers)")
    print(f"    F1 (test)   : PyTorch {s_torch['f1']:.1f}%   integer simulation {s_int['f1']:.1f}%  "
          f"(precision {s_int['precision']:.1f}%, recall {s_int['recall']:.1f}%)")
    print(f"    FIDELITY    : {fid:.3f}% of the cells with the same class (PyTorch vs. integers)")
    if equal is None:
        print(f"    EXACTNESS   : not checked ({msg})")
    else:
        print(f"    EXACTNESS   : {100 * equal / total:.2f}%  of {total} grids (compiled C vs. integer simulation, must be 100%) "
              + ("OK" if equal == total else "[X]"))
    json.dump(dict(method="boolean", task="fomo", header=header, flash_bytes=flash, ram_buffers=ram, steps=steps,
                   layers=shapes, threshold=th, f1_torch=s_torch, f1_sim=s_int, fidelity=fid,
                   exactness=None if equal is None else 100 * equal / total, images=n),
              open(os.path.join(args.run, "export.json"), "w"), indent=1)
    print(f"    -> {os.path.join(args.run, 'export.json')}")


if __name__ == "__main__":
    main()
