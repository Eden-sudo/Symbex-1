"""
ecg_export.py -- a trained ECG network (ecg_train.py) into integers, checked, and into lib_symbex.

    python ecg_export.py runs/m_binary_e100_s42           # lib_symbex's network (1-bit first-layer weights)
    python ecg_export.py runs/m_int8_e100_s42             # int8 first-layer weights (SymbexBool1D.weights_i8)
    python ecg_export.py runs/m_binary_e100_s42 --first-bytes   # the 1-bit first layer as +-1 bytes (int8 kernel),
                                                          # written to runs/m_binary_e100_s42_bytes/

Run from benchmarks/conv1d/ecg_mitbih/ (main environment). The same steps as har_ultracompact/har_export.py, with the
CDT's 1D exporter (compiler_core/export/bool1d.py): fold every conv + BatchNorm into integer weights and thresholds,
simulate the integer network on the validation and test beats (FIDELITY = same class as PyTorch), write
model_symbex.h, and check that the header + lib_symbex compiled on this PC give the same classes as the simulation
(EXACTNESS 100 %), with the HAL row functions and the inline two-window path (-mpopcnt).
The chip windows: --images validation beats drawn with numpy seed 0 (the validation records start with long runs of
normal beats; a draw covers every class). Writes into the run folder: validation_x.u8 (uint8 = int8 + 128) /
validation_y / validation_sim, export.json and model_symbex.h.
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
from ecg_data import RR_CHANNELS, WINDOW, load, scores                                        # noqa: E402
from ecg_train import make_model                                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/<arch>_<first>_e<epochs>_s<seed>")
    ap.add_argument("--images", type=int, default=1000, help="validation beats for the C check and the chip (1000)")
    ap.add_argument("--first-bytes", action="store_true",
                    help="binary runs: the 1-bit first layer as +-1 bytes (int8 kernel), written to <run>_bytes/")
    args = ap.parse_args()
    info = json.load(open(os.path.join(args.run, "train.json")))
    if args.first_bytes and info["first"] != "binary":
        sys.exit("[X] --first-bytes is for the runs with a binary first layer")
    rr = bool(info.get("rr", False))                   # the RR-interval channels (ecg_train.py --rr)
    channels = 1 + (RR_CHANNELS if rr else 0)
    model = make_model(info["arch"], info["first"], rr)
    model.load_state_dict(torch.load(os.path.join(args.run, "model.pth"), map_location="cpu"))
    model.eval()
    net = bool1d.fold(model)
    res = dict(arch=info["arch"], first=info["first"], seed=info["seed"])
    print(f"\n    ECG EXPORT  ({args.run}: {info['arch']}, first layer {info['first']})")
    for split in ("validation", "test"):
        x, y = load(split, rr=rr)
        with torch.no_grad():
            p_torch = model(torch.as_tensor(x)).argmax(1).numpy()
        p_sim = bool1d.simulate(net, x)
        res[split] = dict(balanced_torch=scores(y, p_torch)["balanced"], balanced_sim=scores(y, p_sim)["balanced"],
                          fidelity=100 * float((p_sim == p_torch).mean()))
        r = res[split]
        print(f"    {split:<11}: balanced accuracy PyTorch {r['balanced_torch']:.2f}%  integer simulation "
              f"{r['balanced_sim']:.2f}%  FIDELITY {r['fidelity']:.2f}%")
    x, y = load("validation", rr=rr)
    n = min(args.images, len(x))
    pick = np.sort(np.random.default_rng(0).choice(len(x), n, replace=False))
    x, y = x[pick], y[pick]
    x_u8 = (x.astype(np.int16) + 128).astype(np.uint8).reshape(n, -1)
    sim = bool1d.simulate(net, x)
    out = args.run.rstrip("/") + "_bytes" if args.first_bytes else args.run   # its own folder: chip_test.py writes there
    os.makedirs(out, exist_ok=True)
    save_validation(out, x_u8, y, sim)
    header = os.path.join(out, "model_symbex.h")
    flash, ram, steps, shapes = bool1d.write_header(net, header, WINDOW, channels, first_bytes=args.first_bytes)
    print(f"    header     : {header}  ({flash / 1024:.2f} KB of weights + thresholds, {ram / 1024:.2f} KB of buffers)")
    res.update(header=header, flash_bytes=flash, ram_buffers=ram, steps=steps, layers=shapes, images=n,
               first_bytes=args.first_bytes)
    for native in (False, True):
        equal, total, msg = bool1d.validate_c(header, LIB, x_u8, sim, native_popcount=native)
        path = "inline two windows (-mpopcnt)" if native else "HAL row functions"
        if equal is None:
            print(f"    EXACTNESS  : not checked ({msg})")
        else:
            print(f"    EXACTNESS  : {100 * equal / total:.2f}%  of {total} beats, {path} (must be 100%) "
                  + ("OK" if equal == total else "[X]"))
        res["exactness_native" if native else "exactness"] = None if equal is None else 100 * equal / total
    json.dump(res, open(os.path.join(out, "export.json"), "w"), indent=1)
    print(f"    -> {os.path.join(out, 'export.json')}")


if __name__ == "__main__":
    main()
