"""
layer_sensitivity.py -- does the CDT's sensitivity scan (tools/compiler_core/sensitivity.py) point at the same layers
of tiny_llm as the real damage to the model? Layer by layer, on the fp32 model, three measures side by side:

  1. cdt_bin    the CDT's estimate() as it is: cosine to the fp32 output of ONE layer with the SYMBEX binarization
                ("bin": 1-bit weights AND 1-bit input, the fully binary path), also its 2/4/8-bit options.
  2. w1a8       the same cosine with tiny_llm's own format: 1-bit weights (sign x mean|w| per row) and 8-bit inputs
                (absmax per token), train.py's quantizers. The CDT has no option for it (weights only at 1 bit).
  3. real       the truth the cosines try to predict: binarize ONLY that layer (W1A8) in the fp32 model and measure
                the validation loss of the whole model; damage = loss - fp32 loss. No retraining.

Then the rank correlation (Spearman) of each cosine with the real damage, and the top layers of each. Also groups
(one layer type in every block, one block, everything) and whether the single-layer damages add up.

The output layer is the token embedding used through F.linear (tied), which the scan cannot see; here it becomes a
module (`head`, sharing the embedding's weight) so it is measured like any other layer. Binarizing it binarizes
only the output side (the 1-bit head of the chip's future work); the embedding lookup stays fp32.

Calibration: 32 windows of 256 tokens from the TRAINING split (what the CDT uses). Real damage: the first --windows
windows of the validation split (train.py's evaluate). The test split is not read.

    python layer_sensitivity.py --seeds 42 43 44        # results/layer_sensitivity.json + one table per seed
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import train as T                                             # noqa: E402
from compiler_core.sensitivity import Sensitivity             # noqa: E402

TYPES = ("q", "k", "v", "o", "w1", "w3", "w2")


class HeadLM(T.TinyLM):
    """TinyLM with the tied output layer as a module (`head`, same Parameter as emb.weight) so hooks can see it."""

    def __init__(self, *a):
        super().__init__(*a)
        self.head = T.QLinear(self.emb.weight.shape[1], self.emb.weight.shape[0], "fp32")
        self.head.weight = self.emb.weight

    def forward(self, idx):
        t = idx.shape[1]
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[:t], self.sin[:t])
        return self.head(self.norm(x))


def load(run, device):
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device)
    a = ck["args"]
    if a["weights"] != "fp32":
        raise SystemExit(f"{run}: needs an fp32 run (the scan starts from the original model)")
    m = HeadLM(4096, a["dim"], a["layers"], a["heads"], a["hidden"], a["seq"], "fp32").to(device)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m, a


def layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, T.QLinear)]


@torch.no_grad()
def w1a8_cosine(model, calib):
    """Cosine of every layer's output with W1A8 against its fp32 output, on its real input (one layer at a time)."""
    inputs, hooks = {}, []
    for n, m in layers(model):
        hooks.append(m.register_forward_hook(lambda mod, i, o, n=n: inputs.__setitem__(n, i[0].detach())))
    model(calib)
    for h in hooks:
        h.remove()
    out = {}
    for n, m in layers(model):
        x = inputs[n]
        ref = F.linear(x, m.weight)
        q = F.linear(T.quant_act8(x), T.quant_weight(m.weight, "binary"))
        out[n] = F.cosine_similarity(q.flatten(), ref.flatten(), dim=0).item()
    return out


def loss_with(model, binary, val, seq, windows):
    """Validation loss with the layers in `binary` at W1A8, the rest fp32."""
    for n, m in layers(model):
        m.kind = "binary" if n in binary else "fp32"
    loss = T.evaluate(model, val, seq, windows // 128, 128)
    for _, m in layers(model):
        m.kind = "fp32"
    model.eval()
    return loss


def run_seed(seed, args, device):
    run = os.path.join(HERE, "runs", f"fp32_d384l8_s{seed}")
    model, a = load(run, device)
    seq = a["seq"]
    torch.manual_seed(seed)
    train = T.load_split("train", device)
    val = T.load_split("val", device)
    gen = torch.Generator(device=device).manual_seed(seed)
    calib, _ = T.batch(train, args.calib, seq, gen)

    names = [n for n, _ in layers(model)]
    est = Sensitivity(model).estimate(calib, ram_kb=10 ** 6, flash_kb=10 ** 6, budget=1.0, verbose=False)
    cdt = {r["name"]: {str(op): round(o["fid"], 5) for op, o in r["options"].items()} for r in est["rows"]}
    w1a8 = w1a8_cosine(model, calib)

    base = loss_with(model, set(), val, seq, args.windows)
    real = {}
    for n in names:
        real[n] = loss_with(model, {n}, val, seq, args.windows) - base
        print(f"  s{seed} {n:<14} cdt_bin {cdt[n]['bin']:.4f}  w1a8 {w1a8[n]:.4f}  damage {real[n]:+.4f}", flush=True)

    blocks = sorted({n.split(".")[1] for n in names if n.startswith("blocks.")}, key=int)
    groups = {f"type {t}": [n for n in names if n.endswith("." + t)] for t in TYPES}
    groups.update({f"block {b}": [n for n in names if n.startswith(f"blocks.{b}.")] for b in blocks})
    groups["all blocks (no head)"] = [n for n in names if n != "head"]
    groups["all + head"] = names
    group = {}
    for g, members in groups.items():
        d = loss_with(model, set(members), val, seq, args.windows) - base
        group[g] = {"damage": round(d, 5), "sum_of_single": round(sum(real[n] for n in members), 5)}
        print(f"  s{seed} {g:<22} damage {d:+.4f}  (sum of single layers {group[g]['sum_of_single']:+.4f})",
              flush=True)

    dmg = np.array([real[n] for n in names])
    corr = {}
    for key, vals in (("cdt_bin", [cdt[n]["bin"] for n in names]), ("cdt_8bit", [cdt[n]["8"] for n in names]),
                      ("w1a8", [w1a8[n] for n in names])):
        rho = spearmanr(-np.array(vals), dmg).statistic        # low cosine = sensitive: rank the negated cosine
        rho_nohead = spearmanr(-np.array(vals[:-1]), dmg[:-1]).statistic
        corr[key] = {"spearman": round(float(rho), 4), "spearman_without_head": round(float(rho_nohead), 4)}
    top = lambda score, rev: [n for n, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=rev)[:args.top]]
    tops = {"real": top(real, True), "cdt_bin": top({n: cdt[n]["bin"] for n in names}, False),
            "w1a8": top(w1a8, False)}
    overlap = {k: len(set(tops[k]) & set(tops["real"])) for k in ("cdt_bin", "w1a8")}
    return {"seed": seed, "run": os.path.relpath(run, HERE), "fp32_val_loss": round(base, 5),
            "layers": {n: {"cdt": cdt[n], "w1a8_cosine": round(w1a8[n], 5), "damage": round(real[n], 5)}
                       for n in names},
            "groups": group, "spearman_vs_damage": corr, "top": tops, f"top{args.top}_overlap_with_real": overlap}


def summary(res, args):
    seeds = [r["seed"] for r in res]
    ms = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
    print(f"\nSeeds {seeds}, real damage on {args.windows} validation windows (loss - fp32 loss, no retraining)")
    print(f"{'type':<6}{'cdt_bin cos':>16}{'w1a8 cos':>16}{'damage per layer':>22}{'damage all 8 together':>26}")
    for t in TYPES:
        names = [n for n in res[0]["layers"] if n.endswith("." + t)]
        cb = [np.mean([r["layers"][n]["cdt"]["bin"] for n in names]) for r in res]
        wa = [np.mean([r["layers"][n]["w1a8_cosine"] for n in names]) for r in res]
        dm = [np.mean([r["layers"][n]["damage"] for n in names]) for r in res]
        gt = [r["groups"][f"type {t}"]["damage"] for r in res]
        print(f"{t:<6}{ms(cb):>16}{ms(wa):>16}{ms(dm):>22}{ms(gt):>26}")
    h = lambda k: [r["layers"]["head"][k] for r in res]
    print(f"{'head':<6}{ms([r['layers']['head']['cdt']['bin'] for r in res]):>16}{ms(h('w1a8_cosine')):>16}"
          f"{ms(h('damage')):>22}")
    for g in ("all blocks (no head)", "all + head"):
        print(f"{g}: damage {ms([r['groups'][g]['damage'] for r in res])}, "
              f"sum of single layers {ms([r['groups'][g]['sum_of_single'] for r in res])}")
    for key in ("cdt_bin", "cdt_8bit", "w1a8"):
        print(f"Spearman({key}, damage): {ms([r['spearman_vs_damage'][key]['spearman'] for r in res])}"
              f"  without head {ms([r['spearman_vs_damage'][key]['spearman_without_head'] for r in res])}")
    for r in res:
        print(f"s{r['seed']} top {args.top} real: {', '.join(r['top']['real'])}")
        print(f"      cdt_bin: {', '.join(r['top']['cdt_bin'])}  (in common {r[f'top{args.top}_overlap_with_real']['cdt_bin']})")
        print(f"      w1a8:    {', '.join(r['top']['w1a8'])}  (in common {r[f'top{args.top}_overlap_with_real']['w1a8']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--calib", type=int, default=32, help="calibration windows (training split) for the cosines")
    ap.add_argument("--windows", type=int, default=512, help="validation windows for the real damage (x128)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "layer_sensitivity.json"))
    args = ap.parse_args()
    device = "cuda"
    res = [run_seed(s, args, device) for s in args.seeds]
    summary(res, args)
    with open(args.out, "w") as f:
        json.dump({"what": __doc__.strip().split("\n\n")[0], "calib_windows": args.calib,
                   "val_windows": args.windows, "seeds": res}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
