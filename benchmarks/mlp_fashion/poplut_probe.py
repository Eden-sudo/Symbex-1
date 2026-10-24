"""
poplut_probe.py -- mlp_fashion benchmark, falsification probe for PopLUT (codebook-compressed binary weights).

    python poplut_probe.py --modelo runs/symbex_local_s42/model_binarized.pth
    python poplut_probe.py --modelo ... --k 16 64 256 --epocas-ft 3

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
PopLUT post hoc is discarded for this model. Results: results/poplut_probe.json.
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

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "tools"))
sys.path.insert(0, AQUI)

from compiler_core.methods import apply_plan, widen  # noqa: E402
from model import DATA_DIR, MEAN, STD, FashionMLP  # noqa: E402

PLAN = {"red.1": "bin", "red.4": "bin", "red.7": "bin"}
COMPRIMIDAS = ("red.1", "red.4")
ANCHO = 2.0
BITS_PALABRA = 32
ITERACIONES_KMEANS = 25
PERDIDA_MAXIMA = 1.0                 # kill criterion, accuracy points at K = 256 after the fine-tune


def construir(ruta_modelo):
    base = FashionMLP()
    widen(base, ANCHO, torch.zeros(2, 1, 28, 28), seed=42)
    modelo = apply_plan(base, PLAN, first="red.1", bin_method="plain")
    modelo.load_state_dict(torch.load(ruta_modelo, map_location="cpu", weights_only=True))
    return modelo.eval()


def datos():
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize([MEAN], [STD])])
    tr = torchvision.datasets.FashionMNIST(os.path.join(AQUI, DATA_DIR), train=True, download=True, transform=tf)
    te = torchvision.datasets.FashionMNIST(os.path.join(AQUI, DATA_DIR), train=False, download=True, transform=tf)
    return tr, te


def precision(modelo, te):
    modelo.eval()
    x = te.data.float().div(255).sub(MEDIA).div(DESV).unsqueeze(1)
    with torch.no_grad():
        return 100 * float((modelo(x).argmax(1) == te.targets).float().mean())


def palabras(signos):
    """(rows, n) of +-1 -> (rows * words, 32) of +-1 with 0 on the padded bits."""
    filas, n = signos.shape
    n_pal = math.ceil(n / BITS_PALABRA)
    x = np.zeros((filas, n_pal * BITS_PALABRA), dtype=np.float32)
    x[:, :n] = signos
    return x.reshape(filas * n_pal, BITS_PALABRA), n_pal


def kmeans_binario(x, k, semilla):
    """Sign-symmetric binary k-means. x: (N, 32) in {-1, 0, +1}. Returns codebook (k, 32) in +-1, indices, flags
    (flag -1 = the word is stored as the complement of its codeword)."""
    rng = np.random.default_rng(semilla)
    validos = np.abs(x).sum(1) > 0
    distintos = np.unique(x[validos], axis=0)
    c = distintos[rng.choice(len(distintos), size=min(k, len(distintos)), replace=False)].copy()
    c[c == 0] = 1
    for _ in range(ITERACIONES_KMEANS):
        prod = x @ c.T                                       # agreement - disagreement on the valid bits
        idx = np.abs(prod).argmax(1)
        flag = np.where(prod[np.arange(len(x)), idx] >= 0, 1.0, -1.0).astype(np.float32)
        orientado = x * flag[:, None]
        nuevo = np.zeros_like(c)
        np.add.at(nuevo, idx, orientado)
        vacios = np.abs(nuevo).sum(1) == 0
        c = np.where(nuevo >= 0, 1.0, -1.0).astype(np.float32)
        if vacios.any():                                     # reseed empty clusters with the worst-fitted words
            peor = np.argsort(np.abs(prod).max(1))[:vacios.sum()]
            c[vacios] = np.where(x[peor] >= 0, 1.0, -1.0)
    prod = x @ c.T
    idx = np.abs(prod).argmax(1)
    flag = np.where(prod[np.arange(len(x)), idx] >= 0, 1.0, -1.0).astype(np.float32)
    return c, idx, flag


def entropia_bits(idx, flag):
    _, cuentas = np.unique(idx * 2 + (flag > 0), return_counts=True)
    p = cuentas / cuentas.sum()
    return float(-(p * np.log2(p)).sum())


def comprimir(modelo, k, semilla):
    """Replaces the signs of the compressed layers by their codebook reconstruction (in place).
    Returns per-layer statistics."""
    capas = dict(modelo.named_modules())
    stats = {}
    for nombre in COMPRIMIDAS:
        w = capas[nombre].weight.data
        filas, n = w.shape
        alpha = w.abs().mean(1, keepdim=True)
        signos = np.where(w.numpy() >= 0, 1.0, -1.0).astype(np.float32)
        x, n_pal = palabras(signos)
        c, idx, flag = kmeans_binario(x, k, semilla)
        rec = (c[idx] * flag[:, None]).reshape(filas, n_pal * BITS_PALABRA)[:, :n]
        capas[nombre].weight.data = alpha * torch.from_numpy(rec)
        n_palabras = filas * n_pal
        bits_codigo = math.ceil(math.log2(len(c))) + 1
        stats[nombre] = dict(
            forma=[filas, n], palabras=n_palabras, codebook=len(c),
            error_hamming=float((rec != signos).mean()),
            entropia_bits_por_palabra=entropia_bits(idx, flag),
            bytes_indices=math.ceil(n_palabras * bits_codigo / 8), bytes_codebook=len(c) * BITS_PALABRA // 8,
            bytes_1bit=math.ceil(filas * n / 8),
            tabla_por_inferencia=n_pal * len(c), busquedas_por_inferencia=n_palabras)
        stats[nombre]["bytes"] = stats[nombre]["bytes_indices"] + stats[nombre]["bytes_codebook"]
        stats[nombre]["bits_por_peso"] = 8 * stats[nombre]["bytes"] / (filas * n)
    return stats


def ajustar(modelo, tr, epocas, semilla):
    """Short fine-tune of everything except the compressed signs."""
    torch.manual_seed(semilla)
    capas = dict(modelo.named_modules())
    for nombre in COMPRIMIDAS:
        capas[nombre].weight.requires_grad_(False)
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    carga = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True,
                                        generator=torch.Generator().manual_seed(semilla))
    for _ in range(epocas):
        modelo.train()
        for xb, yb in carga:
            opt.zero_grad()
            F.cross_entropy(modelo(xb), yb).backward()
            opt.step()
    return modelo.eval()


def bytes_modelo(stats, modelo):
    capas = dict(modelo.named_modules())
    ultima = capas["red.7"].weight.shape
    return sum(s["bytes"] for s in stats.values()) + math.ceil(ultima[0] * ultima[1] / 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, help="model_binarized.pth of a CDT run of symbex.toml (x2, plain)")
    ap.add_argument("--k", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--epocas-ft", type=int, default=3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default=os.path.join(AQUI, "results", "poplut_probe.json"))
    args = ap.parse_args()

    tr, te = datos()
    base = construir(args.modelo)
    acc_base = precision(base, te)
    capas = dict(base.named_modules())
    bytes_base = sum(math.ceil(capas[n].weight.numel() / 8) for n in PLAN)
    print(f"[*] base x2: {acc_base:.2f}% | 1-bit weights {bytes_base} B")

    filas = []
    for k in args.k:
        modelo = construir(args.modelo)
        stats = comprimir(modelo, k, args.semilla)
        acc_post = precision(modelo, te)
        acc_ft = precision(ajustar(modelo, tr, args.epocas_ft, args.semilla), te)
        total = bytes_modelo(stats, modelo)
        fila = dict(k=k, acc_posthoc=acc_post, acc_ft=acc_ft, perdida_ft=acc_base - acc_ft, bytes_pesos=total,
                    capas=stats)
        filas.append(fila)
        print(f"    K={k:4d} | posthoc {acc_post:.2f}% | ft {acc_ft:.2f}% ({acc_ft - acc_base:+.2f}) | "
              f"{total} B ({total / bytes_base:.2f}x of 1 bit) | "
              + " | ".join(f"{n}: {s['bits_por_peso']:.3f} bpw, hamming {100 * s['error_hamming']:.1f}%, "
                           f"H {s['entropia_bits_por_palabra']:.2f} b/word" for n, s in stats.items()))

    k256 = next((f for f in filas if f["k"] == 256), None)
    veredicto = None if k256 is None else (
        "discarded" if k256["perdida_ft"] > PERDIDA_MAXIMA else "survives")
    out = dict(modelo=os.path.relpath(args.modelo, AQUI), acc_base=acc_base, bytes_pesos_base=bytes_base,
               comprimidas=list(COMPRIMIDAS), epocas_ft=args.epocas_ft, semilla_kmeans=args.semilla,
               criterio=f"K=256 after the fine-tune loses more than {PERDIDA_MAXIMA} point(s) vs. base -> discarded",
               veredicto=veredicto, resultados=filas)
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    with open(args.salida, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] verdict: {veredicto} | {args.salida}")


if __name__ == "__main__":
    main()
