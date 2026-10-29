"""
har_export.py -- a trained HAR network (har_train.py) into integers, checked, and into lib_symbex.

    python har_export.py runs/max_binary_e100_s42        # lib_symbex's network (1-bit first-layer weights)
    python har_export.py runs/max_int8_e100_s42          # ultracompactBNN's network: also into lib_symbex, its first
                                                          # layer with int8 weights (SymbexBool1D.weights_i8)
    python har_export.py runs/max_binary_e100_s42 --first-bytes   # the same 1-bit network, its first layer as +-1
                                                          # bytes for the int8 kernel: runs/max_binary_e100_s42_bytes/

Run from benchmarks/conv1d/har_ultracompact/ (main environment). With the CDT's 1D exporter (compiler_core/export/
bool1d.py):
    fold        every conv + BatchNorm into integer weights and one threshold per channel; the classifier into an
                integer scale and bias per class
    simulate    the integer network on the validation and test subjects: FIDELITY = same class as PyTorch
    header      model_symbex.h for lib_symbex's symbex_bool1d layers
    C check     the header + lib_symbex compiled on this PC must give the same classes as the
                simulation (EXACTNESS 100 %), twice: the HAL row functions and the inline two-window path (-mpopcnt)
Writes into the run folder: net.npz (the integer network: what the ultracompactBNN side reads), validation_x.u8
(uint8 = int8 + 128) / validation_y / validation_sim (the classes of the integer simulation: the chip test and
har_gvsoc.py), export.json and model_symbex.h.
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
from model import AXES, LENGTH, BinaryHAR                                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/<arch>_<first>_e<epochs>_s<seed>")
    ap.add_argument("--images", type=int, default=1000, help="validation windows for the C check and the chip (1000)")
    ap.add_argument("--first-bytes", action="store_true",
                    help="binary runs: the 1-bit first layer as +-1 bytes (int8 kernel), written to <run>_bytes/")
    args = ap.parse_args()
    info = json.load(open(os.path.join(args.run, "train.json")))
    model = BinaryHAR(info["arch"], info["first"])
    model.load_state_dict(torch.load(os.path.join(args.run, "model.pth"), map_location="cpu"))
    model.eval()
    net = bool1d.fold(model)
    np.savez(os.path.join(args.run, "net.npz"), first=net["first"], length=net["length"], fc_w=net["fc_w"],
             fc_scale=net["fc_scale"], fc_bias=net["fc_bias"],
             **{f"{k}{i}": c[k] for i, c in enumerate(net["layers"]) for k in ("w", "thr", "k", "stride", "pool")})
    res = dict(arch=info["arch"], first=info["first"], seed=info["seed"])
    print(f"\n    HAR EXPORT  ({args.run}: {info['arch']}, first layer {info['first']})")
    for split in ("validation", "test"):
        x, y = load(split)
        with torch.no_grad():
            p_torch = model(torch.as_tensor(x)).argmax(1).numpy()
        p_sim = bool1d.simulate(net, x)
        res[split] = dict(accuracy_torch=100 * float((p_torch == y).mean()), accuracy_sim=100 * float((p_sim == y).mean()),
                          fidelity=100 * float((p_sim == p_torch).mean()))
        r = res[split]
        print(f"    {split:<11}: PyTorch {r['accuracy_torch']:.2f}%  integer simulation {r['accuracy_sim']:.2f}%  "
              f"FIDELITY {r['fidelity']:.2f}%")
    x, y = load("validation")
    n = min(args.images, len(x))
    x_u8 = (x[:n].astype(np.int16) + 128).astype(np.uint8).reshape(n, -1)
    sim = bool1d.simulate(net, x[:n])
    out = args.run.rstrip("/") + "_bytes" if args.first_bytes else args.run   # its own folder: chip_test.py writes there
    if args.first_bytes and info["first"] != "binary":
        sys.exit("[X] --first-bytes is for the runs with a binary first layer")
    os.makedirs(out, exist_ok=True)
    save_validation(out, x_u8, y[:n], sim)               # both kinds: har_gvsoc.py reads them (numpy only)
    header = os.path.join(out, "model_symbex.h")
    flash, ram, steps, shapes = bool1d.write_header(net, header, LENGTH, AXES, first_bytes=args.first_bytes)
    print(f"    header     : {header}  ({flash / 1024:.2f} KB of weights + thresholds, {ram / 1024:.2f} KB of buffers)")
    res.update(header=header, flash_bytes=flash, ram_buffers=ram, steps=steps, layers=shapes, images=n,
               first_bytes=args.first_bytes)
    for native in (False, True):
        equal, total, msg = bool1d.validate_c(header, LIB, x_u8, sim, native_popcount=native)
        path = "inline two windows (-mpopcnt)" if native else "HAL row functions"
        if equal is None:
            print(f"    EXACTNESS  : not checked ({msg})")
        else:
            print(f"    EXACTNESS  : {100 * equal / total:.2f}%  of {total} windows, {path} (must be 100%) "
                  + ("OK" if equal == total else "[X]"))
        res["exactness_native" if native else "exactness"] = None if equal is None else 100 * equal / total
    json.dump(res, open(os.path.join(out, "export.json"), "w"), indent=1)
    print(f"    -> {os.path.join(out, 'export.json')}")


if __name__ == "__main__":
    main()
