"""
fashion_int4.py -- mlp_fashion benchmark, low-bit integer baselines (INT4 and INT2), independent from SYMBEX.

    python fashion_int4.py                       # INT4 and INT2: PTQ + QAT (5 epochs), seed 42
    python fashion_int4.py --bits 4 --epochs 15  # only INT4, QAT with as many epochs as the SYMBEX CDT
    python fashion_int4.py --seeds 3             # QAT with seeds 42, 43, 44 (PTQ has no seed)

Run from benchmarks/mlp/mlp_fashion/ with the main environment (.venv, PyTorch CPU is enough). Nothing here uses
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
An INT8 reference (--bits 8) uses the same code, as a sanity check against results/tflite/tflite_pc.json.

Accuracy on the first 2000 Fashion-MNIST test images (the same images as tflite_pc.json and
symbex_*_export.json) and on the whole 10,000-image test set (the same set as acc_last in symbex_*_train.json).
(--epocas, --semillas, --validacion: the earlier Spanish names of these options, still accepted.)
Bytes: packed weights (ceil(n * bits / 8), nibbles for INT4) and the total with scales and int32 biases.
Results: results/lowbit/int<bits>_pc.json (int4_pc.json, int2_pc.json, int8_pc.json); with --epochs other than 5,
results/int<bits>_e<epochs>_pc.json. --shape 128-512: the same for another FP32 shape (results/..._h128_512_...).
"""
import argparse
import json
import tomllib
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mlp_fashion/ (this script lives in earlier/)
MEAN, STD = 0.2860, 0.3530
TEMPERATURE, ALPHA, LR, WEIGHT_DECAY = 4.0, 0.85, 1e-3, 1e-5     # the CDT defaults ([training] in cdt.py)
RANGE_MOMENTUM = 0.99
N_CALIBRATION = 1000
QAT_EPOCHS = 5


# ---------------------------------------------------------------------------------------------------
# 1. Folded FP32 network
# ---------------------------------------------------------------------------------------------------
def folded_weights(pth_path):
    """state_dict of FashionMLP -> [(W (out x in), b, relu)] with BatchNorm and the input normalization folded,
    so the network takes pixels / 255 directly (same folding as fashion_tflite.py)."""
    import torch
    sd = torch.load(pth_path, map_location="cpu")
    layers = []
    for lin, bn, relu in (("red.1", "red.2", True), ("red.4", "red.5", True), ("red.7", None, False)):
        W = sd[f"{lin}.weight"].double().numpy()
        b = sd[f"{lin}.bias"].double().numpy() if f"{lin}.bias" in sd else np.zeros(W.shape[0])
        if bn:
            g = sd[f"{bn}.weight"].double().numpy() / np.sqrt(sd[f"{bn}.running_var"].double().numpy() + 1e-5)
            b = g * (b - sd[f"{bn}.running_mean"].double().numpy()) + sd[f"{bn}.bias"].double().numpy()
            W = W * g[:, None]
        layers.append([W, b, relu])
    W0, b0, r0 = layers[0]
    layers[0] = [W0 / STD, b0 - W0.sum(1) * MEAN / STD, r0]
    return layers


def scale_maxabs(W, qmax):
    return np.maximum(np.abs(W).max(1), 1e-12) / qmax


def scale_mse(W, qmax, steps=100):
    """Per-row clipping ratio of max|w| that minimizes the squared error of the quantized row."""
    wmax = np.maximum(np.abs(W).max(1), 1e-12)
    best_s, best_e = wmax / qmax, np.full(W.shape[0], np.inf)
    for r in np.linspace(0.05, 1.0, steps):
        s = wmax * r / qmax
        e = ((np.clip(np.round(W / s[:, None]), -qmax, qmax) * s[:, None] - W) ** 2).sum(1)
        better = e < best_e
        best_s[better], best_e[better] = s[better], e[better]
    return best_s


# ---------------------------------------------------------------------------------------------------
# 2. Fake-quant model (PyTorch): the same arithmetic as the integer simulation, differentiable with STE
# ---------------------------------------------------------------------------------------------------
def build_model(layers, bits, scales):
    import torch
    import torch.nn as nn

    def round_ste(x):
        return x + (torch.round(x) - x).detach()

    def fq_activation(x, scale, zero):
        return (torch.clamp(round_ste(x / scale) + zero, -128, 127) - zero) * scale

    class QuantizedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.qmax = 2 ** (bits - 1) - 1
            self.W = nn.ParameterList([nn.Parameter(torch.tensor(W, dtype=torch.float32)) for W, _, _ in layers])
            self.b = nn.ParameterList([nn.Parameter(torch.tensor(b, dtype=torch.float32)) for _, b, _ in layers])
            self.s = nn.ParameterList([nn.Parameter(torch.tensor(s, dtype=torch.float32)) for s in scales])
            self.relu = [r for _, _, r in layers]
            self.register_buffer("range_min", torch.zeros(len(layers)))
            self.register_buffer("range_max", torch.ones(len(layers)))
        def weights_fq(self, i):
            s = self.s[i].abs().clamp_min(1e-8)[:, None]
            return torch.clamp(round_ste(self.W[i] / s), -self.qmax, self.qmax) * s

        def activation_params(self, i):
            """(scale, zero point) of the int8 output of layer i: [0, max] for ReLU layers, [min, max] otherwise."""
            lo = 0.0 if self.relu[i] else min(float(self.range_min[i]), 0.0)
            hi = max(float(self.range_max[i]), lo + 1e-6)
            scale = (hi - lo) / 255.0
            return scale, int(round(-128 - lo / scale))

        def forward(self, x, until=None):
            """until = i: stop at layer i and return its output before the int8 fake-quant (for calibration)."""
            x = fq_activation(x, 1.0 / 255.0, -128)
            for i in range(len(self.W)):
                x = x @ self.weights_fq(i).t() + self.b[i]
                if self.relu[i]:
                    x = torch.relu(x)
                if until == i:
                    return x
                if self.training:                        # moving-average ranges while fine-tuning
                    with torch.no_grad():
                        self.range_min[i] = RANGE_MOMENTUM * self.range_min[i] + (1 - RANGE_MOMENTUM) * x.min()
                        self.range_max[i] = RANGE_MOMENTUM * self.range_max[i] + (1 - RANGE_MOMENTUM) * x.max()
                scale, zero = self.activation_params(i)
                x = fq_activation(x, scale, zero)
            return x

    return QuantizedMLP()


def calibrate(model, x_calib):
    """Activation ranges = min/max over the calibration images, layer by layer with the quantized weights
    (layer i is calibrated on the quantized outputs of the layers before it)."""
    import torch
    model.eval()
    with torch.no_grad():
        for i in range(len(model.W)):
            out = model(x_calib, until=i)
            model.range_min[i], model.range_max[i] = out.min(), out.max()


def predict_fq(model, x):
    import torch
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).argmax(1).numpy()


# ---------------------------------------------------------------------------------------------------
# 3. Integer-only export and simulation
# ---------------------------------------------------------------------------------------------------
def export_integer(model):
    """The model as integers: per layer Wq (int8 codes), bq (int32), requantization multiplier, zero points."""
    import torch
    layers, scale_in, zero_in = [], 1.0 / 255.0, -128
    with torch.no_grad():
        for i in range(len(model.W)):
            s_w = model.s[i].abs().clamp_min(1e-8).double().numpy()
            Wq = np.clip(np.round(model.W[i].double().numpy() / s_w[:, None]), -model.qmax, model.qmax).astype(np.int64)
            bq = np.round(model.b[i].double().numpy() / (scale_in * s_w)).astype(np.int64)
            scale_out, zero_out = model.activation_params(i)
            layers.append(dict(Wq=Wq, bq=bq, zero_in=zero_in, multiplier=scale_in * s_w / scale_out,
                               zero_out=zero_out, s_w=s_w))
            scale_in, zero_in = scale_out, zero_out
    return layers


def predict_integer(layers, x_u8):
    """Integer inference: int8 activations, int64 accumulators, int8 requantization with clipping."""
    x = x_u8.astype(np.int64) - 128                      # round((p / 255) / (1 / 255)) - 128
    for c in layers:
        acc = (x - c["zero_in"]) @ c["Wq"].T + c["bq"]
        x = np.clip(np.round(acc * c["multiplier"]) + c["zero_out"], -128, 127).astype(np.int64)
    return x.argmax(1)


def save_integer(model, bits, seed, suffix=""):
    """the integer model (export_integer) -> models/int<bits>_qat_s<seed>.npz, layer i as W<i>, b<i>, m<i> (the
    float64 requantization multipliers), zin<i>, zout<i>: what int_export.py writes as C for the board"""
    layers = export_integer(model)
    arrays = {}
    for i, c in enumerate(layers):
        arrays.update({f"W{i}": c["Wq"], f"b{i}": c["bq"], f"m{i}": np.asarray(c["multiplier"], dtype=np.float64),
                       f"zin{i}": np.int64(c["zero_in"]), f"zout{i}": np.int64(c["zero_out"])})
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    np.savez(os.path.join(HERE, "models", f"int{bits}_qat_s{seed}{suffix}.npz"), bits=bits, n_layers=len(layers),
             **arrays)


def model_bytes(layers, bits):
    weights = sum(math.ceil(c["Wq"].size * bits / 8) for c in layers)
    scales = sum(4 * len(c["s_w"]) for c in layers)
    biases = sum(4 * len(c["bq"]) for c in layers)
    return dict(pesos_bytes=weights, escalas_bytes=scales, sesgos_bytes=biases, total_bytes=weights + scales + biases)


def levels_used(layers, qmax):
    return [int(len(np.unique(c["Wq"]))) for c in layers] + [2 * qmax + 1]


# ---------------------------------------------------------------------------------------------------
# 4. Quantization-aware fine-tuning
# ---------------------------------------------------------------------------------------------------
def train_qat(model, teacher, x_tr, y_tr, x_te, y_te, epochs, seed, batch=256, distill=True):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    xt, yt = torch.as_tensor(x_tr), torch.as_tensor(y_tr)
    with torch.no_grad():
        teacher_logits = torch.cat([teacher(xt[i:i + 4096]) for i in range(0, len(xt), 4096)])
    history = []
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for k in range(0, len(xt), batch):
            idx = order[k:k + batch]
            out = model(xt[idx])
            kd = F.kl_div(F.log_softmax(out / TEMPERATURE, 1), F.softmax(teacher_logits[idx] / TEMPERATURE, 1),
                          reduction="batchmean") * TEMPERATURE ** 2
            ce = F.cross_entropy(out, yt[idx])
            loss = ALPHA * kd + (1 - ALPHA) * ce if distill else ce          # --no-distill: the labels only
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        acc = 100 * float((predict_fq(model, x_te) == y_te).mean())
        history.append([ep, total / len(xt), acc])
        print(f"      QAT epoch {ep + 1}/{epochs} | loss {total / len(xt):.4f} | test (fake-quant) {acc:.2f}%")
    return history


def cdt_training_indices(n):
    """the training images the CDT uses: it keeps [data] validation_split apart for validation, with the fixed
    [data] split_seed (tools/cdt.py, load_data); model.split_indices makes the same call, so the QAT trains on
    exactly the same images as the SYMBEX runs it is compared with, and as the FP32 model it starts from"""
    from model import split_indices
    return np.array(split_indices(n)[0])


def fp32_teacher(layers):
    import torch

    def f(x):
        for W, b, relu in layers:
            x = x @ torch.tensor(W.T, dtype=torch.float32) + torch.tensor(b, dtype=torch.float32)
            if relu:
                x = torch.relu(x)
        return x
    return f


# ---------------------------------------------------------------------------------------------------
def evaluate(name, model, bits, x_val, y_val, x_te_u8, y_te):
    layers_q = export_integer(model)
    pred_val = predict_integer(layers_q, x_val)
    pred_te = predict_integer(layers_q, x_te_u8)
    pred_fq = predict_fq(model, x_te_u8.astype(np.float32) / 255.0)
    r = dict(precision=100 * float((pred_val == y_val).mean()), n=len(y_val),
             precision_10k=100 * float((pred_te == y_te).mean()), n_10k=len(y_te),
             acuerdo_fake_quant=100 * float((pred_te == pred_fq).mean()),
             niveles_por_capa=levels_used(layers_q, model.qmax)[:-1], niveles_maximos=2 * model.qmax + 1,
             **model_bytes(layers_q, bits))
    print(f"    {name:<12} {r['precision']:6.2f}% ({r['n']}) | {r['precision_10k']:6.2f}% (10k) | integer = fake-quant "
          f"{r['acuerdo_fake_quant']:.2f}% | weights {r['pesos_bytes']} B, total {r['total_bytes']} B")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 2], choices=[2, 3, 4, 8])
    ap.add_argument("--epochs", "--epocas", type=int, default=QAT_EPOCHS, help="QAT epochs (the SYMBEX CDT uses 15)")
    ap.add_argument("--seeds", "--semillas", type=int, default=1, help="QAT seeds: 42, 43, ... (default 1)")
    ap.add_argument("--validation", "--validacion", type=int, default=2000, help="validation images (the first N of the test set)")
    ap.add_argument("--no-distill", action="store_true",
                    help="QAT on the labels only (no distillation from the FP32 model); results get _nodistill")
    ap.add_argument("--shape", default="256-128",
                    help="hidden widths h1-h2 of the FP32 model (its own models/fashion_mlp_<h1>_<h2>.pth, as in "
                         "fashion_sizing.py); other than 256-128, results and models get _h<h1>_<h2>")
    args = ap.parse_args()
    h1, h2 = (int(v) for v in args.shape.split("-"))
    shape_tag = "" if (h1, h2) == (256, 128) else f"_h{h1}_{h2}"
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    import torch
    from model import calibration_u8, get_fp32, validation_u8, _fashion
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    fp32_path = get_fp32(h1, h2)
    x_val, y_val = validation_u8(args.validation)
    x_te, y_te = validation_u8(10000)
    tr = _fashion(True)
    x_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy().astype(np.int64)
    x_calib = torch.as_tensor(calibration_u8(N_CALIBRATION).astype(np.float32) / 255.0)

    layers = folded_weights(fp32_path)
    teacher = fp32_teacher(layers)
    with torch.no_grad():
        pred_fp32 = teacher(torch.as_tensor(x_te.astype(np.float32) / 255.0)).argmax(1).numpy()
    acc_fp32_10k = 100 * float((pred_fp32 == y_te).mean())
    acc_fp32 = 100 * float((pred_fp32[:len(y_val)] == y_val).mean())
    print(f"[*] FP32 folded: {acc_fp32:.2f}% ({len(y_val)} images) | {acc_fp32_10k:.2f}% (10k) | {fp32_path}")
    os.makedirs(os.path.join("results", "lowbit"), exist_ok=True)

    for bits in args.bits:
        qmax = 2 ** (bits - 1) - 1
        t0 = time.time()
        print(f"\n[*] INT{bits} weights (per channel, symmetric, codes in [-{qmax}, {qmax}]), int8 activations")
        res = dict(bits=bits, esquema="per-channel symmetric narrow-range weights, per-tensor int8 activations, "
                                      "integer simulation", modelo_fp32=fp32_path,
                   acc_fp32=acc_fp32, acc_fp32_10k=acc_fp32_10k, metodos={})
        for name, f_scale in (("ptq_maxabs", scale_maxabs), ("ptq_mse", scale_mse)):
            m = build_model(layers, bits, [f_scale(W, qmax) for W, _, _ in layers])
            calibrate(m, x_calib)
            res["metodos"][name] = evaluate(name, m, bits, x_val, y_val, x_te, y_te)

        qat = []
        for k in range(args.seeds):
            seed = 42 + k
            print(f"    qat, seed {seed}, {args.epochs} epochs (from ptq_mse, "
                  f"{'labels only' if args.no_distill else 'distillation from FP32'})")
            m = build_model(layers, bits, [scale_mse(W, qmax) for W, _, _ in layers])
            calibrate(m, x_calib)
            keep = cdt_training_indices(len(x_tr))                 # the images the CDT trains on (every seed)
            history = train_qat(m, teacher, x_tr[keep], y_tr[keep], x_te.astype(np.float32) / 255.0, y_te,
                                args.epochs, seed, distill=not args.no_distill)
            r = evaluate(f"qat s{seed}", m, bits, x_val, y_val, x_te, y_te)
            r.update(semilla=seed, epocas=args.epochs, history=history, train_images=len(keep),
                     split_seed=tomllib.load(open(os.path.join(HERE, "symbex.toml"), "rb"))["data"]["split_seed"])
            save_integer(m, bits, seed, shape_tag + ("_nodistill" if args.no_distill else ""))  # int_export.py
            qat.append(r)
        res["metodos"]["qat"] = qat[0] if len(qat) == 1 else dict(
            semillas=qat, **{c: qat[0][c] for c in ("pesos_bytes", "escalas_bytes", "sesgos_bytes", "total_bytes")},
            precision=float(np.mean([q["precision"] for q in qat])),
            precision_10k=float(np.mean([q["precision_10k"] for q in qat])),
            precision_10k_std=float(np.std([q["precision_10k"] for q in qat], ddof=1)), n=len(y_val), epocas=args.epochs)
        res["segundos"] = round(time.time() - t0, 1)
        path = (f"results/lowbit/int{bits}{'' if args.epochs == QAT_EPOCHS else f'_e{args.epochs}'}{shape_tag}"
                f"{'_nodistill' if args.no_distill else ''}_pc.json")
        res["distill"] = not args.no_distill
        res["shape"] = f"784-{h1}-{h2}-10"
        json.dump(res, open(path, "w"), indent=1)
        print(f"    -> {path} ({res['segundos']} s)")


if __name__ == "__main__":
    main()
