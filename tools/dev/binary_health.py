"""
binary_health.py -- audit of trained runs: how many latent 1-bit weights ended frozen (|w| > 1).

    python tools/dev/binary_health.py                              # every runs/**/*.pth under benchmarks/
    python tools/dev/binary_health.py benchmarks/conv2d/cnn_cifar_resnet8/runs/w1_s42
    python tools/dev/binary_health.py benchmarks/conv1d --tensors  # one line per weight tensor too
    python tools/dev/binary_health.py --limit 5                    # only the files above 5 % frozen

The straight-through sign of the benchmarks passes no gradient where |w| > 1, so a latent weight that crossed 1 kept
its bit for the rest of the training (compiler_core/train/health.py). New trainings record binary_health in train.json;
this reads the checkpoints already on disk, which do not say which tensors are 1-bit: every weight tensor with two or
more dimensions is counted (convolutions and linear layers). An int8 first layer or int8 classifier is counted too and
its |w| > 1 means nothing (per-channel scale): read it with --tensors. Bop runs show 0 % (every weight is +-0.5);
FP32 runs (fp32_* folders) are skipped. Prints one line per file: % frozen, largest |w|, number of weights.
"""
import argparse
import glob
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def checkpoints(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for f in sorted(glob.glob(os.path.join(p, "**", "*.pth"), recursive=True)):
            if os.sep + "runs" + os.sep in f and not os.path.basename(os.path.dirname(f)).startswith("fp32"):
                yield f


def weight_tensors(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: v.float() for k, v in state.items()
            if torch.is_tensor(v) and k.endswith("weight") and v.dim() >= 2 and v.is_floating_point()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[os.path.join(REPO, "benchmarks")])
    ap.add_argument("--tensors", action="store_true", help="one line per weight tensor as well")
    ap.add_argument("--limit", type=float, default=0.0, help="list only the files with more than this %% frozen")
    args = ap.parse_args()
    rows = 0
    for f in checkpoints(args.paths):
        try:
            ws = weight_tensors(torch.load(f, map_location="cpu", weights_only=True))
        except Exception as e:                       # noqa: BLE001 -- an unreadable file is reported, not fatal
            print(f"    [?] {os.path.relpath(f)}: not read ({type(e).__name__})")
            continue
        if not ws:
            continue
        n = sum(w.numel() for w in ws.values())
        frozen = 100.0 * sum(int((w.abs() > 1).sum()) for w in ws.values()) / n
        if frozen <= args.limit and args.limit > 0:
            continue
        top = max(float(w.abs().max()) for w in ws.values())
        flag = " [!]" if frozen > 1 else ""
        print(f"{frozen:6.1f} % frozen  max|w| {top:7.3f}  {n:9d} w  {os.path.relpath(f)}{flag}")
        rows += 1
        if args.tensors:
            for k, w in ws.items():
                print(f"          {100.0 * float((w.abs() > 1).float().mean()):6.1f} %  max|w| {float(w.abs().max()):7.3f}"
                      f"  {tuple(w.shape)}  {k}")
    if not rows:
        print("    no checkpoint found (or none above --limit)")


if __name__ == "__main__":
    main()
