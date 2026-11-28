"""
deep_mlp.py -- deeper binary MLPs for damage.py (section 4 of the README): does the damage grow through the layers
because of the convolutions or because of the depth?

DeepMLP_<n>: 784 -> 128 -> 512 x (n - 1) -> 10, every hidden layer Linear -> BatchNorm1d -> ReLU, so it has n binary
layers like the convolutional nets of section 3 (HAR / ECG 3, ResNet-8 7). It is the Fashion MLP of section 3
(784-128-512-10, n = 2) made deeper with more 512-wide layers. Same recipe as that MLP (mlp_fashion/fashion_sizing.py
--no-distill): its own FP32 model (model.train_fp32, 30 epochs, seed 42, the CDT's validation split held out), then
the CDT: method plain, every layer 1 bit, 15 epochs, labels only, seeds 42, 43, 44. The CDT prints its own held-out
test accuracy; this study never uses it.
The FP32 models of section 8 (models/deep_mlp_3.pth, deep_mlp_7.pth) were trained before the seed was set ahead of
DeepMLP's random init (fixed Nov 28): their training order was seeded, their initial weights were not.

    python deep_mlp.py --depths 3 7 --seeds 42 43 44        # -> runs/deep_mlp/d<n>_s<seed> (gitignored)
    python damage.py mlp --depth 3 --seeds 42 43 44 --out results/damage.json
"""
import argparse
import os
import re
import subprocess
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FASHION = os.path.normpath(os.path.join(HERE, "..", "..", "mlp", "mlp_fashion"))
CDT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "cdt.py"))
EPOCHS = 15                                              # the CDT epochs of fashion_sizing.py


class DeepMLP(nn.Module):
    """784 -> 128 -> 512 x (depth - 1) -> 10"""
    def __init__(self, depth=3):
        super().__init__()
        widths = [784, 128] + [512] * (depth - 1)
        layers = [nn.Flatten()]
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b, bias=False), nn.BatchNorm1d(b), nn.ReLU()]
        self.red = nn.Sequential(*layers, nn.Linear(widths[-1], 10))

    def forward(self, x):
        return self.red(x)


def __getattr__(name):
    """deep_mlp.DeepMLP_<n> -> a DeepMLP of depth n buildable with no arguments (what the CDT needs)"""
    m = re.fullmatch(r"DeepMLP_(\d+)", name)
    if m:
        n = int(m.group(1))
        return type(name, (DeepMLP,), {"__init__": lambda self: DeepMLP.__init__(self, n)})
    raise AttributeError(f"module 'deep_mlp' has no attribute {name!r}")


FIXED_INIT = False      # --fixed-init: the runs trained with init_thresholds one layer at a time (PR #218), apart


def use_fixed_init():
    """read / write the runs trained with the fixed init_thresholds (section 8): runs/deep_mlp_fixinit, the 2-layer
    MLP's mlp_fashion/runs/initfix, result keys prefixed fixinit_"""
    global FIXED_INIT
    FIXED_INIT = True


def mlp_tag(depth):
    """the result key of an MLP: mlp / mlp_d<n>, fixinit_ in front with --fixed-init"""
    tag = f"mlp_d{depth}" if depth else "mlp"
    return "fixinit_" + tag if FIXED_INIT else tag


def run_dir(depth, seed):
    return os.path.join(HERE, "runs", "deep_mlp_fixinit" if FIXED_INIT else "deep_mlp", f"d{depth}_s{seed}")


def fp32(depth):
    """models/deep_mlp_<n>.pth, trained with the FP32 recipe of the benchmark if missing"""
    path = os.path.join(HERE, "models", f"deep_mlp_{depth}.pth")
    if not os.path.exists(path):
        sys.path.insert(0, FASHION)
        import model as M
        cwd = os.getcwd()
        os.chdir(FASHION)                                # model.py's paths are relative to its folder
        try:
            torch.manual_seed(M.SEED)                    # DeepMLP's random init too, not only train_fp32's batches
            acc_val, _ = M.train_fp32(path, M.DATA_DIR, net=DeepMLP(depth))
        finally:
            os.chdir(cwd)
        print(f"[*] saved {path} (FP32 validation {acc_val:.2f}%)")
    return path


def run(depth, seed):
    out = run_dir(depth, seed)
    if os.path.exists(os.path.join(out, "export.json")):
        print(f"[*] depth {depth}, seed {seed}: already done")
        return
    text = open(os.path.join(FASHION, "symbex.toml")).read()
    for key, value in (("file", f'"{fp32(depth)}"'), ("definition", f'"{os.path.join(HERE, "deep_mlp.py")}:DeepMLP_{depth}"'),
                       ("widths", "[1]")):
        text, n = re.subn(rf"(?m)^{key}(\s*)=.*$", lambda m: f"{key}{m.group(1)}= {value}", text)
        if n != 1:
            sys.exit(f"[X] symbex.toml: expected one '{key} =' line, found {n}")
    config = os.path.join(FASHION, f".deep_mlp_{depth}_s{seed}.toml")   # next to symbex.toml: same relative paths
    open(config, "w").write(text)
    print(f"\n[*] DeepMLP depth {depth}, seed {seed} -> {out}")
    cmd = [sys.executable, CDT, config, "--method", "plain", "--seed", str(seed), "--epochs", str(EPOCHS),
           "--search-epochs", "1", "--max-accuracy-drop", "100", "--output-directory", out, "--to", "export",
           "--no-verbose", "--no-distill"]
    try:
        code = subprocess.run(cmd, cwd=FASHION).returncode
    finally:
        os.remove(config)
    if code != 0:
        sys.exit(f"[X] the CDT failed on {out} (the output above says why)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--depths", nargs="+", type=int, default=[3, 7], help="binary layers (the MLP of section 3 has 2)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--fixed-init", action="store_true",
                    help="train into runs/deep_mlp_fixinit (the CDT with init_thresholds one layer at a time, PR #218)")
    args = ap.parse_args()
    if args.fixed_init:
        use_fixed_init()
    for d in args.depths:
        for s in args.seeds:
            run(d, s)


if __name__ == "__main__":
    main()
