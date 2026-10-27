"""
study.py -- the low-bit integer baselines of a benchmark: the SAME FP32 network at INT8, INT4 and INT2 (Brevitas,
quantize.py), post-training and quantization-aware, evaluated with the integer simulation (what the chip runs).

    from lowbit.study import Data, run
    res = run("mlp_fashion", model_fp32, Data(x_train, y_train, x_val, y_val, x_test, y_test), bits=[8, 4, 2],
              seeds=[42, 43, 44], epochs=15, distill=True, out_dir="runs/lowbit")

Methods, all from the same FP32 model:
  ptq_maxabs  post-training: weight scale per channel = max|w| / qmax; activation ranges calibrated on 1000 training images
  ptq_mse     post-training: the weight scale that minimizes the weight error (Brevitas' MSE quantizer)
  qat         quantization-aware training (fake-quant, straight-through estimator, weight and activation scales learned),
              starting from the FP32 weights with calibrated activations; the CDT's recipe: Adam lr 1e-3, weight decay
              1e-5, cosine to 1e-5, batch 256, distillation from the FP32 model (T = 4, alpha = 0.85) or the labels only
Every number is the integer simulation (quantize.run_integer), on the validation split (carved out of the training data)
and on the test set, which is evaluated once per final model. The integer models are saved for the chip export:
<out_dir>/int<bits>_<method>[_s<seed>].npy.
"""
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from lowbit.quantize import QNet, export, fold, quantize_input, run_integer, weight_bytes

LR, WEIGHT_DECAY, ETA_MIN = 1e-3, 1e-5, 1e-5          # the CDT defaults ([training] in tools/cdt.py)
TEMPERATURE, ALPHA = 4.0, 0.85
BATCH = 256
N_CALIBRATION = 1000


@dataclass
class Data:
    """float tensors exactly as the FP32 model takes them (normalized), labels int64; validation from the TRAINING data"""
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(x, size=4096):
    for i in range(0, len(x), size):
        yield x[i:i + size]


def calibrate(qnet, x_cal):
    """activation ranges from the calibration images (Brevitas calibration mode), then eval"""
    from brevitas.graph.calibrate import calibration_mode
    qnet.eval()
    with torch.no_grad(), calibration_mode(qnet):
        for xb in _batches(x_cal, BATCH):
            qnet(xb)
    qnet.eval()
    with torch.no_grad():
        qnet(x_cal[:BATCH])                                  # fixes qnet.scales for export()


def integer_accuracy(layers, x, y, chunk=2000):
    """accuracy (%) of the integer simulation, and its predictions"""
    pred = np.concatenate([run_integer(layers, quantize_input(layers, xb.cpu().numpy())).reshape(len(xb), -1).argmax(1)
                           for xb in _batches(x, chunk)])
    return 100 * float((pred == y.cpu().numpy()).mean()), pred


def train_qat(qnet, teacher, data, epochs, seed, distill):
    dev = _device()
    torch.manual_seed(seed)
    qnet.to(dev).train()
    xt, yt = data.x_train.to(dev), data.y_train.to(dev)
    with torch.no_grad():
        t_logits = torch.cat([teacher(xb) for xb in _batches(xt)]) if distill else None
    opt = torch.optim.Adam(qnet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=ETA_MIN)
    history = []
    for ep in range(epochs):
        qnet.train()
        order = torch.randperm(len(xt), device=dev)
        total = 0.0
        for k in range(0, len(xt), BATCH):
            idx = order[k:k + BATCH]
            out = qnet(xt[idx])
            ce = F.cross_entropy(out, yt[idx])
            if distill:
                kd = F.kl_div(F.log_softmax(out / TEMPERATURE, 1), F.softmax(t_logits[idx] / TEMPERATURE, 1),
                              reduction="batchmean") * TEMPERATURE ** 2
                loss = ALPHA * kd + (1 - ALPHA) * ce
            else:
                loss = ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        qnet.eval()
        with torch.no_grad():
            acc = 100 * float(torch.cat([qnet(xb).argmax(1) for xb in _batches(data.x_val.to(dev))])
                              .eq(data.y_val.to(dev)).float().mean())
        history.append([ep + 1, total / len(xt), acc])
        print(f"      epoch {ep + 1}/{epochs} | loss {total / len(xt):.4f} | validation (fake-quant) {acc:.2f}%")
    return qnet.cpu().eval(), history


def evaluate(name, qnet, data, out_path):
    with torch.no_grad():
        qnet(data.x_val[:BATCH])
    layers = export(qnet)
    acc_val, _ = integer_accuracy(layers, data.x_val, data.y_val)
    acc_test, pred = integer_accuracy(layers, data.x_test, data.y_test)
    with torch.no_grad():
        fq = torch.cat([qnet(xb).argmax(1) for xb in _batches(data.x_test)]).numpy()
    np.save(out_path, np.array(layers, dtype=object), allow_pickle=True)
    r = dict(validation=acc_val, test=acc_test, agree_fake_quant=100 * float((pred == fq).mean()), **weight_bytes(layers),
             model=os.path.relpath(out_path))
    print(f"    {name:<14} validation {acc_val:6.2f}% | test {acc_test:6.2f}% | integer = fake-quant "
          f"{r['agree_fake_quant']:.2f}% | weights {r['weight_bytes']} B, total {r['total_bytes']} B")
    return r


def _stats(v):
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def run(bench, model_fp32, data, bits=(8, 4, 2), seeds=(42, 43, 44), epochs=15, distill=True, out_dir="runs/lowbit",
        ptq=True):
    """every method at every bit width; returns (and the caller saves) the results dict"""
    os.makedirs(out_dir, exist_ok=True)
    model_fp32 = model_fp32.eval()
    ops = fold(model_fp32, data.x_train[:64])
    teacher = model_fp32.to(_device()).eval()
    with torch.no_grad():
        p32 = torch.cat([teacher(xb.to(_device())).argmax(1).cpu() for xb in _batches(data.x_test)])
        v32 = torch.cat([teacher(xb.to(_device())).argmax(1).cpu() for xb in _batches(data.x_val)])
    res = dict(bench=bench, recipe=dict(epochs=epochs, seeds=list(seeds), distill=distill, lr=LR, batch=BATCH),
               fp32=dict(validation=100 * float((v32 == data.y_val).float().mean()),
                         test=100 * float((p32 == data.y_test).float().mean())),
               scheme="weights per channel, symmetric, narrow range; activations int8 per tensor; ESP-NN integer "
                      "arithmetic (quantize.py)", tool="brevitas " + __import__("brevitas").__version__, bits={})
    print(f"[*] {bench}: FP32 validation {res['fp32']['validation']:.2f}% | test {res['fp32']['test']:.2f}%")
    gen = torch.Generator().manual_seed(0)
    x_cal = data.x_train[torch.randperm(len(data.x_train), generator=gen)[:N_CALIBRATION]]
    tag = "" if distill else "_nodistill"
    for b in bits:
        t0 = time.time()
        print(f"\n[*] INT{b}: weight codes in [-{2 ** (b - 1) - 1}, {2 ** (b - 1) - 1}], int8 activations")
        out = {}
        if ptq:
            for method, init in (("ptq_maxabs", "maxabs"), ("ptq_mse", "mse")):
                q = QNet(ops, b, init)
                calibrate(q, x_cal)
                out[method] = evaluate(method, q, data, os.path.join(out_dir, f"int{b}_{method}.npy"))
        runs = []
        for s in seeds:
            print(f"    qat, seed {s}, {epochs} epochs, {'distillation from FP32' if distill else 'labels only'}")
            q = QNet(ops, b, "learned")
            calibrate(q, x_cal)
            q, hist = train_qat(q, teacher, data, epochs, s, distill)
            r = evaluate(f"qat s{s}", q, data, os.path.join(out_dir, f"int{b}_qat{tag}_s{s}.npy"))
            r.update(seed=s, history=hist)
            runs.append(r)
        out["qat"] = dict(seeds=runs, validation=_stats([r["validation"] for r in runs]),
                          test=_stats([r["test"] for r in runs]), weight_bytes=runs[0]["weight_bytes"],
                          total_bytes=runs[0]["total_bytes"])
        out["seconds"] = round(time.time() - t0, 1)
        res["bits"][str(b)] = out
        q_ = out["qat"]["test"]
        print(f"    -> INT{b} QAT test {q_['mean']:.2f} +- {q_['std']:.2f}% ({out['seconds']} s)")
    return res


def table(res):
    """the summary rows: method, bits, weight bytes, validation and test (mean +- std)"""
    rows = [("FP32", 32, None, res["fp32"]["validation"], res["fp32"]["test"], 0.0)]
    for b, out in res["bits"].items():
        for m in ("ptq_maxabs", "ptq_mse"):
            if m in out:
                rows.append((f"INT{b} {m}", int(b), out[m]["weight_bytes"], out[m]["validation"], out[m]["test"], 0.0))
        q = out["qat"]
        rows.append((f"INT{b} qat", int(b), q["weight_bytes"], q["validation"]["mean"], q["test"]["mean"], q["test"]["std"]))
    print(f"\n    {'method':<18} {'weights B':>10} {'validation':>11} {'test':>16}")
    for name, _, wb, va, te, sd in rows:
        print(f"    {name:<18} {wb if wb else '-':>10} {va:10.2f}% {te:9.2f} +- {sd:.2f}%")
    return rows


def save(res, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)
    print(f"[*] -> {path}")
