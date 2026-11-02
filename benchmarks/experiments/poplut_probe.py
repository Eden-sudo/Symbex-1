"""
poplut_probe.py -- mlp_fashion benchmark, falsification probe for PopLUT (codebook-compressed binary weights).

    python poplut_probe.py --model runs/symbex_local_s42/model_binarized.pth
    python poplut_probe.py --model ... --k 16 64 256 --ft-epochs 3

(The earlier Spanish options --modelo, --epocas-ft, --semilla, --salida still work; the JSON keys stay as they were.)

The question: can the 1-bit weights of the trained SYMBEX x2 MLP (784 -> 512 -> 256 -> 10) be stored as indices
into a small codebook of 32-bit binary words, at well under 1 bit per weight, without losing accuracy?

PopLUT, as probed here:
  - every weight row of a layer is cut into 32-bit words (the XNOR / popcount word of the C engine; the last
    word of a row is zero-padded, padded bits are ignored by the distance);
  - one codebook of K 32-bit words per layer, and every word is stored as (index, complement flag): the word
    is c_k or ~c_k. Cost per word: log2(K) + 1 bits, plus K * 4 bytes of codebook per layer;
  - the codebook is learned post hoc by binary k-means in Hamming distance, symmetric in sign (a word joins the
    nearest of c_k / ~c_k; the update is a per-bit majority vote after undoing the complements);
  - on the chip, popcount(x_j ^ c_k) (or, for the uint8 first layer, the masked sum of x over c_k) is computed
    once per input word j and codeword k, and every row only looks it up: rows * words lookups plus
    words * K table entries instead of rows * words popcounts.

The per-row scale alpha = mean|w| is kept (it folds into the BatchNorm threshold anyway); only the signs change.
red.1 (512 x 784) and red.4 (256 x 512) are compressed; red.7 (10 x 256, 320 bytes) stays plain 1 bit.

Measured, on the 10,000 Fashion-MNIST test images:
  base       the model as trained
  posthoc    codebook signs, nothing retrained
  ft         posthoc + a short fine-tune (Adam lr 1e-3, cross-entropy, train split) of everything except the
             compressed signs: BatchNorm, input thresholds (RSign beta), the last layer. The codebook and the
             assignments stay fixed, so the stored model is exactly the measured one.
Also: Hamming error (fraction of weight bits that changed), entropy of the (index, flag) stream (the bits per
word an entropy coder could reach), and the packed weight bytes.

Kill criterion (fixed before running): if K = 256 after the fine-tune loses more than 1 point against base,
PopLUT post hoc is discarded for this model. Results: mlp_fashion/results/earlier/poplut_probe.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FASHION_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mlp", "mlp_fashion"))
# this probe lives in benchmarks/experiments/ and works on mlp_fashion: run it from benchmarks/mlp/mlp_fashion/
sys.path.insert(0, os.path.join(FASHION_DIR, "..", "..", "..", "tools"))
sys.path.insert(0, FASHION_DIR)

from compiler_core.methods import apply_plan, widen  # noqa: E402
from model import DATA_DIR, MEAN, STD, FashionMLP  # noqa: E402

PLAN = {"red.1": "bin", "red.4": "bin", "red.7": "bin"}
COMPRESSED = ("red.1", "red.4")
WIDTH = 2.0
WORD_BITS = 32
KMEANS_ITERATIONS = 25
MAX_LOSS = 1.0                 # kill criterion, accuracy points at K = 256 after the fine-tune


def build(model_path):
    base = FashionMLP()
    widen(base, WIDTH, torch.zeros(2, 1, 28, 28), seed=42)
    model = apply_plan(base, PLAN, first="red.1", bin_method="plain")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model.eval()


def data():
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize([MEAN], [STD])])
    tr = torchvision.datasets.FashionMNIST(os.path.join(FASHION_DIR, DATA_DIR), train=True, download=True, transform=tf)
    te = torchvision.datasets.FashionMNIST(os.path.join(FASHION_DIR, DATA_DIR), train=False, download=True, transform=tf)
    return tr, te


def accuracy(model, te):
    model.eval()
    x = te.data.float().div(255).sub(MEAN).div(STD).unsqueeze(1)
    with torch.no_grad():
        return 100 * float((model(x).argmax(1) == te.targets).float().mean())


def words(signs):
    """(rows, n) of +-1 -> (rows * words, 32) of +-1 with 0 on the padded bits."""
    rows, n = signs.shape
    n_words_row = math.ceil(n / WORD_BITS)
    x = np.zeros((rows, n_words_row * WORD_BITS), dtype=np.float32)
    x[:, :n] = signs
    return x.reshape(rows * n_words_row, WORD_BITS), n_words_row


def binary_kmeans(x, k, seed):
    """Sign-symmetric binary k-means. x: (N, 32) in {-1, 0, +1}. Returns codebook (k, 32) in +-1, indices, flags
    (flag -1 = the word is stored as the complement of its codeword)."""
    rng = np.random.default_rng(seed)
    valid = np.abs(x).sum(1) > 0
    distinct = np.unique(x[valid], axis=0)
    c = distinct[rng.choice(len(distinct), size=min(k, len(distinct)), replace=False)].copy()
    c[c == 0] = 1
    for _ in range(KMEANS_ITERATIONS):
        prod = x @ c.T                                       # agreement - disagreement on the valid bits
        idx = np.abs(prod).argmax(1)
        flag = np.where(prod[np.arange(len(x)), idx] >= 0, 1.0, -1.0).astype(np.float32)
        oriented = x * flag[:, None]
        new = np.zeros_like(c)
        np.add.at(new, idx, oriented)
        empty = np.abs(new).sum(1) == 0
        c = np.where(new >= 0, 1.0, -1.0).astype(np.float32)
        if empty.any():                                     # reseed empty clusters with the worst-fitted words
            worst = np.argsort(np.abs(prod).max(1))[:empty.sum()]
            c[empty] = np.where(x[worst] >= 0, 1.0, -1.0)
    prod = x @ c.T
    idx = np.abs(prod).argmax(1)
    flag = np.where(prod[np.arange(len(x)), idx] >= 0, 1.0, -1.0).astype(np.float32)
    return c, idx, flag


def entropy_bits(idx, flag):
    _, counts = np.unique(idx * 2 + (flag > 0), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def compress(model, k, seed):
    """Replaces the signs of the compressed layers by their codebook reconstruction (in place).
    Returns per-layer statistics."""
    layers = dict(model.named_modules())
    stats = {}
    for name in COMPRESSED:
        w = layers[name].weight.data
        rows, n = w.shape
        alpha = w.abs().mean(1, keepdim=True)
        signs = np.where(w.numpy() >= 0, 1.0, -1.0).astype(np.float32)
        x, n_words_row = words(signs)
        c, idx, flag = binary_kmeans(x, k, seed)
        rec = (c[idx] * flag[:, None]).reshape(rows, n_words_row * WORD_BITS)[:, :n]
        layers[name].weight.data = alpha * torch.from_numpy(rec)
        n_words = rows * n_words_row
        code_bits = math.ceil(math.log2(len(c))) + 1
        stats[name] = dict(
            forma=[rows, n], palabras=n_words, codebook=len(c),
            error_hamming=float((rec != signs).mean()),
            entropia_bits_por_palabra=entropy_bits(idx, flag),
            bytes_indices=math.ceil(n_words * code_bits / 8), bytes_codebook=len(c) * WORD_BITS // 8,
            bytes_1bit=math.ceil(rows * n / 8),
            tabla_por_inferencia=n_words_row * len(c), busquedas_por_inferencia=n_words)
        stats[name]["bytes"] = stats[name]["bytes_indices"] + stats[name]["bytes_codebook"]
        stats[name]["bits_por_peso"] = 8 * stats[name]["bytes"] / (rows * n)
    return stats


def fine_tune(model, tr, epochs, seed):
    """Short fine-tune of everything except the compressed signs."""
    torch.manual_seed(seed)
    layers = dict(model.named_modules())
    for name in COMPRESSED:
        layers[name].weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    loader = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model.eval()


def model_bytes(stats, model):
    layers = dict(model.named_modules())
    last = layers["red.7"].weight.shape
    return sum(s["bytes"] for s in stats.values()) + math.ceil(last[0] * last[1] / 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "--modelo", dest="model", required=True,
                    help="model_binarized.pth of a CDT run of symbex.toml (x2, plain)")
    ap.add_argument("--k", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--ft-epochs", "--epocas-ft", dest="ft_epochs", type=int, default=3)
    ap.add_argument("--seed", "--semilla", dest="seed", type=int, default=0)
    ap.add_argument("--out", "--salida", dest="out",
                    default=os.path.join(FASHION_DIR, "results", "earlier", "poplut_probe.json"))
    args = ap.parse_args()

    tr, te = data()
    base = build(args.model)
    acc_base = accuracy(base, te)
    layers = dict(base.named_modules())
    bytes_base = sum(math.ceil(layers[n].weight.numel() / 8) for n in PLAN)
    print(f"[*] base x2: {acc_base:.2f}% | 1-bit weights {bytes_base} B")

    rows = []
    for k in args.k:
        model = build(args.model)
        stats = compress(model, k, args.seed)
        acc_post = accuracy(model, te)
        acc_ft = accuracy(fine_tune(model, tr, args.ft_epochs, args.seed), te)
        total = model_bytes(stats, model)
        row = dict(k=k, acc_posthoc=acc_post, acc_ft=acc_ft, perdida_ft=acc_base - acc_ft, bytes_pesos=total,
                    capas=stats)
        rows.append(row)
        print(f"    K={k:4d} | posthoc {acc_post:.2f}% | ft {acc_ft:.2f}% ({acc_ft - acc_base:+.2f}) | "
              f"{total} B ({total / bytes_base:.2f}x of 1 bit) | "
              + " | ".join(f"{n}: {s['bits_por_peso']:.3f} bpw, hamming {100 * s['error_hamming']:.1f}%, "
                           f"H {s['entropia_bits_por_palabra']:.2f} b/word" for n, s in stats.items()))

    k256 = next((f for f in rows if f["k"] == 256), None)
    verdict = None if k256 is None else (
        "discarded" if k256["perdida_ft"] > MAX_LOSS else "survives")
    out = dict(modelo=os.path.relpath(args.model, FASHION_DIR), acc_base=acc_base, bytes_pesos_base=bytes_base,
               comprimidas=list(COMPRESSED), epocas_ft=args.ft_epochs, semilla_kmeans=args.seed,
               criterio=f"K=256 after the fine-tune loses more than {MAX_LOSS} point(s) vs. base -> discarded",
               veredicto=verdict, resultados=rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] verdict: {verdict} | {args.out}")


if __name__ == "__main__":
    main()
