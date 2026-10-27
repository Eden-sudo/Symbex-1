"""
fashion_int4.py -- mlp_fashion benchmark, low-bit integer baselines (INT4 and INT2), independent from SYMBEX.

    python fashion_int4.py                       # INT4 and INT2: PTQ + QAT (5 epochs), seed 42
    python fashion_int4.py --bits 4 --epocas 15  # only INT4, QAT with as many epochs as the SYMBEX CDT
    python fashion_int4.py --semillas 3          # QAT with seeds 42, 43, 44 (PTQ has no seed)

Run from benchmarks/mlp_fashion/ with the main environment (.venv, PyTorch CPU is enough). Nothing here uses
the SYMBEX library or the CDT: it answers "how good is an ordinary INT4 / INT2 model of the SAME network?",
so the 1-bit result can be compared with the next integer widths below INT8, not only with INT8.

Quantization scheme (the usual integer-only scheme of TFLite / CMSIS-NN, with fewer weight bits):
  - BatchNorm and the input normalization folded into the Linear layers (the same folding as fashion_tflite.py),
    so the network takes pixels / 255 and has 3 layers 784 -> 256 -> 128 -> 10
  - weights: per output channel, symmetric, narrow range: INT4 in [-7, 7], INT2 in [-1, 1] (ternary codes
    stored in 2 bits); one float32 scale per channel; bias int32
  - activations: int8 per tensor (input scale 1/255 zero point -128, exactly the TFLite input; hidden
    ReLU outputs [0, max] with zero point -128; logits asymmetric int8)
  - evaluation in INTEGER arithmetic (numpy, int64 accumulators, float64 requantization multiplier), not
    with fake-quant floats; the agreement with the fake-quant model is reported as a check

Methods, all from the SAME original FP32 model (models/fashion_mlp.pth, model.get_fp32):
  ptq_maxabs  post-training, channel scale = max|w| / qmax, activations calibrated on 1000 training images
  ptq_mse     post-training, channel scale = the clipping that minimizes the weight MSE (grid search)
  qat         quantization-aware fine-tuning from ptq_mse: fake-quant with the straight-through estimator,
              learnable channel scales, activation ranges by moving average; same loss and optimizer as the
              CDT (Adam lr 1e-3, cosine, distillation from the FP32 model, T = 4, alpha = 0.85)
An INT8 reference (--bits 8) uses the same code, as a sanity check against results/tflite_pc.json.

Accuracy on the first 2000 Fashion-MNIST test images (the same images as tflite_pc.json and
symbex_*_export.json) and on the whole 10,000-image test set (the same set as acc_last in symbex_*_train.json).
Bytes: packed weights (ceil(n * bits / 8), nibbles for INT4) and the total with scales and int32 biases.
Results: results/int<bits>_pc.json (int4_pc.json, int2_pc.json, int8_pc.json); with --epocas other than 5,
results/int<bits>_e<epochs>_pc.json.
"""
import argparse
import json
import tomllib
import math
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
MEDIA, DESV = 0.2860, 0.3530
TEMPERATURA, ALFA, LR, WEIGHT_DECAY = 4.0, 0.85, 1e-3, 1e-5     # the CDT defaults ([training] in cdt.py)
MOMENTO_RANGO = 0.99
N_CALIBRACION = 1000
EPOCAS_QAT = 5


# ---------------------------------------------------------------------------------------------------
# 1. Folded FP32 network
# ---------------------------------------------------------------------------------------------------
def pesos_plegados(ruta_pth):
    """state_dict of FashionMLP -> [(W (out x in), b, relu)] with BatchNorm and the input normalization folded,
    so the network takes pixels / 255 directly (same folding as fashion_tflite.py)."""
    import torch
    sd = torch.load(ruta_pth, map_location="cpu")
    capas = []
    for lin, bn, relu in (("red.1", "red.2", True), ("red.4", "red.5", True), ("red.7", None, False)):
        W = sd[f"{lin}.weight"].double().numpy()
        b = sd[f"{lin}.bias"].double().numpy() if f"{lin}.bias" in sd else np.zeros(W.shape[0])
        if bn:
            g = sd[f"{bn}.weight"].double().numpy() / np.sqrt(sd[f"{bn}.running_var"].double().numpy() + 1e-5)
            b = g * (b - sd[f"{bn}.running_mean"].double().numpy()) + sd[f"{bn}.bias"].double().numpy()
            W = W * g[:, None]
        capas.append([W, b, relu])
    W0, b0, r0 = capas[0]
    capas[0] = [W0 / DESV, b0 - W0.sum(1) * MEDIA / DESV, r0]
    return capas


def escala_maxabs(W, qmax):
    return np.maximum(np.abs(W).max(1), 1e-12) / qmax


def escala_mse(W, qmax, pasos=100):
    """Per-row clipping ratio of max|w| that minimizes the squared error of the quantized row."""
    maximo = np.maximum(np.abs(W).max(1), 1e-12)
    mejor_s, mejor_e = maximo / qmax, np.full(W.shape[0], np.inf)
    for r in np.linspace(0.05, 1.0, pasos):
        s = maximo * r / qmax
        e = ((np.clip(np.round(W / s[:, None]), -qmax, qmax) * s[:, None] - W) ** 2).sum(1)
        mejor = e < mejor_e
        mejor_s[mejor], mejor_e[mejor] = s[mejor], e[mejor]
    return mejor_s


# ---------------------------------------------------------------------------------------------------
# 2. Fake-quant model (PyTorch): the same arithmetic as the integer simulation, differentiable with STE
# ---------------------------------------------------------------------------------------------------
def construir_modelo(capas, bits, escalas):
    import torch
    import torch.nn as nn

    def redondeo_ste(x):
        return x + (torch.round(x) - x).detach()

    def fq_activacion(x, escala, cero):
        return (torch.clamp(redondeo_ste(x / escala) + cero, -128, 127) - cero) * escala

    class MLPCuantizado(nn.Module):
        def __init__(self):
            super().__init__()
            self.qmax = 2 ** (bits - 1) - 1
            self.W = nn.ParameterList([nn.Parameter(torch.tensor(W, dtype=torch.float32)) for W, _, _ in capas])
            self.b = nn.ParameterList([nn.Parameter(torch.tensor(b, dtype=torch.float32)) for _, b, _ in capas])
            self.s = nn.ParameterList([nn.Parameter(torch.tensor(s, dtype=torch.float32)) for s in escalas])
            self.relu = [r for _, _, r in capas]
            self.register_buffer("rango_min", torch.zeros(len(capas)))
            self.register_buffer("rango_max", torch.ones(len(capas)))
        def pesos_fq(self, i):
            s = self.s[i].abs().clamp_min(1e-8)[:, None]
            return torch.clamp(redondeo_ste(self.W[i] / s), -self.qmax, self.qmax) * s

        def parametros_activacion(self, i):
            """(scale, zero point) of the int8 output of layer i: [0, max] for ReLU layers, [min, max] otherwise."""
            lo = 0.0 if self.relu[i] else min(float(self.rango_min[i]), 0.0)
            hi = max(float(self.rango_max[i]), lo + 1e-6)
            escala = (hi - lo) / 255.0
            return escala, int(round(-128 - lo / escala))

        def forward(self, x, hasta=None):
            """hasta = i: stop at layer i and return its output before the int8 fake-quant (for calibration)."""
            x = fq_activacion(x, 1.0 / 255.0, -128)
            for i in range(len(self.W)):
                x = x @ self.pesos_fq(i).t() + self.b[i]
                if self.relu[i]:
                    x = torch.relu(x)
                if hasta == i:
                    return x
                if self.training:                        # moving-average ranges while fine-tuning
                    with torch.no_grad():
                        self.rango_min[i] = MOMENTO_RANGO * self.rango_min[i] + (1 - MOMENTO_RANGO) * x.min()
                        self.rango_max[i] = MOMENTO_RANGO * self.rango_max[i] + (1 - MOMENTO_RANGO) * x.max()
                escala, cero = self.parametros_activacion(i)
                x = fq_activacion(x, escala, cero)
            return x

    return MLPCuantizado()


def calibrar(modelo, x_calib):
    """Activation ranges = min/max over the calibration images, layer by layer with the quantized weights
    (layer i is calibrated on the quantized outputs of the layers before it)."""
    import torch
    modelo.eval()
    with torch.no_grad():
        for i in range(len(modelo.W)):
            sal = modelo(x_calib, hasta=i)
            modelo.rango_min[i], modelo.rango_max[i] = sal.min(), sal.max()


def predecir_fq(modelo, x):
    import torch
    modelo.eval()
    with torch.no_grad():
        return modelo(torch.as_tensor(x, dtype=torch.float32)).argmax(1).numpy()


# ---------------------------------------------------------------------------------------------------
# 3. Integer-only export and simulation
# ---------------------------------------------------------------------------------------------------
def exportar_entero(modelo):
    """The model as integers: per layer Wq (int8 codes), bq (int32), requantization multiplier, zero points."""
    import torch
    capas, escala_in, cero_in = [], 1.0 / 255.0, -128
    with torch.no_grad():
        for i in range(len(modelo.W)):
            s_w = modelo.s[i].abs().clamp_min(1e-8).double().numpy()
            Wq = np.clip(np.round(modelo.W[i].double().numpy() / s_w[:, None]), -modelo.qmax, modelo.qmax).astype(np.int64)
            bq = np.round(modelo.b[i].double().numpy() / (escala_in * s_w)).astype(np.int64)
            escala_out, cero_out = modelo.parametros_activacion(i)
            capas.append(dict(Wq=Wq, bq=bq, cero_in=cero_in, multiplicador=escala_in * s_w / escala_out,
                              cero_out=cero_out, s_w=s_w))
            escala_in, cero_in = escala_out, cero_out
    return capas


def predecir_entero(capas, x_u8):
    """Integer inference: int8 activations, int64 accumulators, int8 requantization with clipping."""
    x = x_u8.astype(np.int64) - 128                      # round((p / 255) / (1 / 255)) - 128
    for c in capas:
        acc = (x - c["cero_in"]) @ c["Wq"].T + c["bq"]
        x = np.clip(np.round(acc * c["multiplicador"]) + c["cero_out"], -128, 127).astype(np.int64)
    return x.argmax(1)


def guardar_entero(modelo, bits, semilla, suffix=""):
    """the integer model (exportar_entero) -> models/int<bits>_qat_s<seed>.npz, layer i as W<i>, b<i>, m<i> (the
    float64 requantization multipliers), zin<i>, zout<i>: what int_export.py writes as C for the board"""
    capas = exportar_entero(modelo)
    arrays = {}
    for i, c in enumerate(capas):
        arrays.update({f"W{i}": c["Wq"], f"b{i}": c["bq"], f"m{i}": np.asarray(c["multiplicador"], dtype=np.float64),
                       f"zin{i}": np.int64(c["cero_in"]), f"zout{i}": np.int64(c["cero_out"])})
    os.makedirs(os.path.join(AQUI, "models"), exist_ok=True)
    np.savez(os.path.join(AQUI, "models", f"int{bits}_qat_s{semilla}{suffix}.npz"), bits=bits, n_layers=len(capas),
             **arrays)


def bytes_modelo(capas, bits):
    pesos = sum(math.ceil(c["Wq"].size * bits / 8) for c in capas)
    escalas = sum(4 * len(c["s_w"]) for c in capas)
    sesgos = sum(4 * len(c["bq"]) for c in capas)
    return dict(pesos_bytes=pesos, escalas_bytes=escalas, sesgos_bytes=sesgos, total_bytes=pesos + escalas + sesgos)


def niveles_usados(capas, qmax):
    return [int(len(np.unique(c["Wq"]))) for c in capas] + [2 * qmax + 1]


# ---------------------------------------------------------------------------------------------------
# 4. Quantization-aware fine-tuning
# ---------------------------------------------------------------------------------------------------
def entrenar_qat(modelo, maestro, x_tr, y_tr, x_te, y_te, epocas, semilla, lote=256, distill=True):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(semilla)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas, eta_min=1e-5)
    xt, yt = torch.as_tensor(x_tr), torch.as_tensor(y_tr)
    with torch.no_grad():
        logits_maestro = torch.cat([maestro(xt[i:i + 4096]) for i in range(0, len(xt), 4096)])
    historia = []
    for ep in range(epocas):
        modelo.train()
        orden = torch.randperm(len(xt))
        total = 0.0
        for k in range(0, len(xt), lote):
            idx = orden[k:k + lote]
            sal = modelo(xt[idx])
            kd = F.kl_div(F.log_softmax(sal / TEMPERATURA, 1), F.softmax(logits_maestro[idx] / TEMPERATURA, 1),
                          reduction="batchmean") * TEMPERATURA ** 2
            ce = F.cross_entropy(sal, yt[idx])
            loss = ALFA * kd + (1 - ALFA) * ce if distill else ce          # --no-distill: the labels only
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        acc = 100 * float((predecir_fq(modelo, x_te) == y_te).mean())
        historia.append([ep, total / len(xt), acc])
        print(f"      QAT epoch {ep + 1}/{epocas} | loss {total / len(xt):.4f} | test (fake-quant) {acc:.2f}%")
    return historia


def cdt_training_indices(n):
    """the training images the CDT uses: it keeps [data] validation_split apart for validation, with the fixed
    [data] split_seed (tools/cdt.py, load_data); model.split_indices makes the same call, so the QAT trains on
    exactly the same images as the SYMBEX runs it is compared with, and as the FP32 model it starts from"""
    from model import split_indices
    return np.array(split_indices(n)[0])


def maestro_fp32(capas):
    import torch

    def f(x):
        for W, b, relu in capas:
            x = x @ torch.tensor(W.T, dtype=torch.float32) + torch.tensor(b, dtype=torch.float32)
            if relu:
                x = torch.relu(x)
        return x
    return f


# ---------------------------------------------------------------------------------------------------
def evaluar(nombre, modelo, bits, x_val, y_val, x_te_u8, y_te):
    capas_q = exportar_entero(modelo)
    pred_val = predecir_entero(capas_q, x_val)
    pred_te = predecir_entero(capas_q, x_te_u8)
    pred_fq = predecir_fq(modelo, x_te_u8.astype(np.float32) / 255.0)
    r = dict(precision=100 * float((pred_val == y_val).mean()), n=len(y_val),
             precision_10k=100 * float((pred_te == y_te).mean()), n_10k=len(y_te),
             acuerdo_fake_quant=100 * float((pred_te == pred_fq).mean()),
             niveles_por_capa=niveles_usados(capas_q, modelo.qmax)[:-1], niveles_maximos=2 * modelo.qmax + 1,
             **bytes_modelo(capas_q, bits))
    print(f"    {nombre:<12} {r['precision']:6.2f}% ({r['n']}) | {r['precision_10k']:6.2f}% (10k) | integer = fake-quant "
          f"{r['acuerdo_fake_quant']:.2f}% | weights {r['pesos_bytes']} B, total {r['total_bytes']} B")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 2], choices=[2, 3, 4, 8])
    ap.add_argument("--epocas", type=int, default=EPOCAS_QAT, help="QAT epochs (the SYMBEX CDT uses 15)")
    ap.add_argument("--semillas", type=int, default=1, help="QAT seeds: 42, 43, ... (default 1)")
    ap.add_argument("--validacion", type=int, default=2000, help="validation images (the first N of the test set)")
    ap.add_argument("--no-distill", action="store_true",
                    help="QAT on the labels only (no distillation from the FP32 model); results get _nodistill")
    args = ap.parse_args()
    os.chdir(AQUI)
    sys.path.insert(0, AQUI)
    import torch
    from model import calibration_u8, get_fp32, validation_u8, _fashion
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ruta_fp32 = get_fp32()
    x_val, y_val = validation_u8(args.validacion)
    x_te, y_te = validation_u8(10000)
    tr = _fashion(True)
    x_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy().astype(np.int64)
    x_calib = torch.as_tensor(calibration_u8(N_CALIBRACION).astype(np.float32) / 255.0)

    capas = pesos_plegados(ruta_fp32)
    maestro = maestro_fp32(capas)
    with torch.no_grad():
        pred_fp32 = maestro(torch.as_tensor(x_te.astype(np.float32) / 255.0)).argmax(1).numpy()
    acc_fp32_10k = 100 * float((pred_fp32 == y_te).mean())
    acc_fp32 = 100 * float((pred_fp32[:len(y_val)] == y_val).mean())
    print(f"[*] FP32 folded: {acc_fp32:.2f}% ({len(y_val)} images) | {acc_fp32_10k:.2f}% (10k) | {ruta_fp32}")
    os.makedirs("results", exist_ok=True)

    for bits in args.bits:
        qmax = 2 ** (bits - 1) - 1
        t0 = time.time()
        print(f"\n[*] INT{bits} weights (per channel, symmetric, codes in [-{qmax}, {qmax}]), int8 activations")
        res = dict(bits=bits, esquema="pesos por canal simetricos rango estrecho, activaciones int8 por tensor, "
                                      "simulacion entera", modelo_fp32=ruta_fp32,
                   acc_fp32=acc_fp32, acc_fp32_10k=acc_fp32_10k, metodos={})
        for nombre, f_escala in (("ptq_maxabs", escala_maxabs), ("ptq_mse", escala_mse)):
            m = construir_modelo(capas, bits, [f_escala(W, qmax) for W, _, _ in capas])
            calibrar(m, x_calib)
            res["metodos"][nombre] = evaluar(nombre, m, bits, x_val, y_val, x_te, y_te)

        qat = []
        for k in range(args.semillas):
            semilla = 42 + k
            print(f"    qat, seed {semilla}, {args.epocas} epochs (from ptq_mse, distillation from FP32)")
            m = construir_modelo(capas, bits, [escala_mse(W, qmax) for W, _, _ in capas])
            calibrar(m, x_calib)
            keep = cdt_training_indices(len(x_tr))                 # the images the CDT trains on (every seed)
            historia = entrenar_qat(m, maestro, x_tr[keep], y_tr[keep], x_te.astype(np.float32) / 255.0, y_te,
                                    args.epocas, semilla, distill=not args.no_distill)
            r = evaluar(f"qat s{semilla}", m, bits, x_val, y_val, x_te, y_te)
            r.update(semilla=semilla, epocas=args.epocas, history=historia, train_images=len(keep),
                     split_seed=tomllib.load(open(os.path.join(AQUI, "symbex.toml"), "rb"))["data"]["split_seed"])
            guardar_entero(m, bits, semilla, "_nodistill" if args.no_distill else "")   # for int_export.py
            qat.append(r)
        res["metodos"]["qat"] = qat[0] if len(qat) == 1 else dict(
            semillas=qat, **{c: qat[0][c] for c in ("pesos_bytes", "escalas_bytes", "sesgos_bytes", "total_bytes")},
            precision=float(np.mean([q["precision"] for q in qat])),
            precision_10k=float(np.mean([q["precision_10k"] for q in qat])),
            precision_10k_std=float(np.std([q["precision_10k"] for q in qat], ddof=1)), n=len(y_val), epocas=args.epocas)
        res["segundos"] = round(time.time() - t0, 1)
        ruta = (f"results/int{bits}{'' if args.epocas == EPOCAS_QAT else f'_e{args.epocas}'}"
                f"{'_nodistill' if args.no_distill else ''}_pc.json")
        res["distill"] = not args.no_distill
        json.dump(res, open(ruta, "w"), indent=1)
        print(f"    -> {ruta} ({res['segundos']} s)")


if __name__ == "__main__":
    main()
