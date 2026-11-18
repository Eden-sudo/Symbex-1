"""
weight_cosine.py -- README part 5c: cos(w, sign w) = ||w||_1 / (sqrt(n) ||w||_2) of every output row of every layer
(1 = equal magnitudes, lower = signal in few weights), mean per layer, against fan-in and the one-shot drop of
part 2 (ResNet-8) and part 5's single matrices (tiny_llm). No model is run, only the saved weights.

    python weight_cosine.py --resnet-runs <training_gap>/runs/train_gap --llm-runs <main>/benchmarks/experiments/tiny_llm/runs
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from fanin_depth import RESNET
from llm_matrices import MATRICES

HERE = os.path.dirname(os.path.abspath(__file__))
RESNET_KEYS = {"stem": "stem.weight", "fc": "fc.weight", "b0": "blocks.0.conv.weight", "b1": "blocks.1.conv.weight",
               "b2": "blocks.2.conv.weight", "ds2": "blocks.2.ds_conv.weight", "b3": "blocks.3.conv.weight",
               "b4": "blocks.4.conv.weight", "ds4": "blocks.4.ds_conv.weight", "b5": "blocks.5.conv.weight"}


def row_cosine(w):
    """mean over output rows of ||w||_1 / (sqrt(n) ||w||_2), rows = dim 0, n = the rest flattened (the fan-in)"""
    w = w.detach().double().reshape(w.shape[0], -1)
    cos = w.abs().sum(1) / (w.shape[1] ** 0.5 * w.norm(dim=1))
    return float(cos.mean()), w.shape[1]


def rho(a, b):
    return round(float(spearmanr(a, b).statistic), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resnet-runs", required=True)
    ap.add_argument("--llm-runs", required=True)
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "weight_cosine.json"))
    args = ap.parse_args()
    table = json.load(open(os.path.join(HERE, "results", "layer_table.json")))
    mats = json.load(open(os.path.join(HERE, "results", "llm_matrices.json")))
    res = {"resnet8": {}, "tiny_llm": {}}
    for s in args.seeds:
        sd = torch.load(os.path.join(os.path.expanduser(args.resnet_runs), f"twin_s{s}.pth"), map_location="cpu")
        names = list(RESNET)
        cos, fan = zip(*(row_cosine(sd[RESNET_KEYS[k]]) for k in names))
        assert list(fan) == [RESNET[k][0] for k in names], fan                  # the fan-ins of part 5
        drop = {k: {"wa": v["w"], **v} for k, v in table["seeds"][s]["drop"].items()}
        o = {"cosine": dict(zip(names, np.round(cos, 4).tolist())), "cos_vs_fanin": rho(cos, fan)}
        for part in ("w", "wa"):
            d = [drop[k][part] for k in names]
            o[part] = {"cos_vs_drop": rho(cos, d), "fanin_vs_drop": rho(fan, d)}
        res["resnet8"][s] = o
        ck = torch.load(os.path.join(os.path.expanduser(args.llm_runs), f"fp32_d384l8_s{s}", "model.pt"),
                        map_location="cpu", weights_only=False)["model"]
        keys = list(mats["seeds"][s]["matrices"])
        cos, fan = zip(*(row_cosine(ck[f"blocks.{k.split('.')[0][1:]}.{k.split('.')[1]}.weight"]) for k in keys))
        d = [mats["seeds"][s]["matrices"][k] for k in keys]
        res["tiny_llm"][s] = {"cos_vs_fanin": rho(cos, fan), "cos_vs_drop": rho(cos, d), "fanin_vs_drop": rho(fan, d),
                              "mean_cosine": {n: round(float(np.mean([c for k, c in zip(keys, cos) if k.endswith("." + n)])), 4)
                                              for n in MATRICES}}
    json.dump(res, open(args.out, "w"), indent=1)
    for s in args.seeds:
        r, l = res["resnet8"][s], res["tiny_llm"][s]
        print(f"s{s} ResNet-8: cos vs fan-in {r['cos_vs_fanin']} | drop w: cos {r['w']['cos_vs_drop']} fan-in "
              f"{r['w']['fanin_vs_drop']} | drop wa: cos {r['wa']['cos_vs_drop']} fan-in {r['wa']['fanin_vs_drop']}")
        print(f"     cosine {r['cosine']}")
        print(f"s{s} tiny_llm: cos vs fan-in {l['cos_vs_fanin']} | drop: cos {l['cos_vs_drop']} fan-in "
              f"{l['fanin_vs_drop']} | mean cosine {l['mean_cosine']}")


if __name__ == "__main__":
    main()
