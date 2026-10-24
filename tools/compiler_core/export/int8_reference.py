"""
int8_reference.py -- the INT8 reference of the ORIGINAL model, to compare against on the same chip.

Post-training INT8 like TFLite (per-tensor weights, calibrated activations, BatchNorm and the input normalization
folded), with exactly the integer arithmetic of TFLite / Espressif's ESP-NN:
    prepare_int8()       the original FP32 model -> int8 weights, int32 biases, multiplier + shift per layer
    simulate_int8()      exact integer simulation (numpy)
    write_header_int8()  model_int8.h: symbex_int8_infer(); with SYMBEX_WITH_ESP_NN it calls esp_nn_fully_connected_s8,
                         otherwise a plain C version of the same arithmetic (used to check it on the PC)
    export_int8()        all of it, checked on the PC against the simulation
It is OUR quantization running with the official kernel; the official TFLite converter + TFLite Micro comparison
lives in benchmarks/ (its own environment).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------------------------------
# 8. INT8 reference of the ORIGINAL model, for the comparison on the chip (TFLite / ESP-NN arithmetic)
# ---------------------------------------------------------------------------------------------------
def _fp32_groups(model, x):
    """Original FP32 MLP: [(Linear, BatchNorm1d or None, ReLU?)] in forward order (Flatten / Dropout skipped)."""
    order, hooks = [], []
    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(lambda mod, e, s_: order.append(mod)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(x)
    model.train(was_training)
    for h in hooks:
        h.remove()
    groups, seen, current = [], set(), None
    for m in order:
        if id(m) in seen:
            continue
        seen.add(id(m))
        if isinstance(m, (nn.Flatten, nn.Dropout, nn.Identity)):
            continue
        if isinstance(m, nn.Linear):
            current = [m, None, False]
            groups.append(current)
        elif isinstance(m, nn.BatchNorm1d) and current is not None and current[1] is None and not current[2]:
            current[1] = m
        elif isinstance(m, nn.ReLU) and current is not None:
            current[2] = True
        else:
            raise ValueError(f"INT8 reference: unsupported module {type(m).__name__} (Linear -> [BatchNorm1d] -> [ReLU] chains)")
    return groups


def quantize_multiplier(M):
    """Real multiplier -> (Q31 multiplier, shift), TFLite convention (shift > 0: left)."""
    if M == 0:
        return 0, 0
    q, shift = np.frexp(M)
    qf = int(round(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    return qf, int(shift)


def quantized_multiply(x, mult, shift):
    """TFLite MultiplyByQuantizedMultiplier in exact integers (vectorized): what ESP-NN computes."""
    x = x.astype(np.int64)
    left, right = max(shift, 0), max(-shift, 0)
    ab = (x * (1 << left)) * mult
    high = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    high = np.where(high >= 0, high >> 31, -((-high) >> 31))            # division truncated toward zero
    high = np.clip(high, -(1 << 31), (1 << 31) - 1)
    if right == 0:
        return high
    mask = (1 << right) - 1
    return (high >> right) + ((high & mask) > ((mask >> 1) + (high < 0)))


def prepare_int8(model, calib, mean=(), std=()):
    """Post-training INT8 of the ORIGINAL model, like TFLite: per-tensor weights (symmetric, +-127), uint8
    input as int8 with zero point -128 (x / 255), every hidden output calibrated on `calib` (normalized
    tensors) as int8 with zero point -128, bias int32, multiplier + shift per layer. BatchNorm and the input
    normalization are folded into the Linear layers. Returns a list of layer dicts."""
    if std and len(set(std)) != 1:
        raise ValueError("INT8 reference: the input normalization must use ONE std for every channel")
    model = model.cpu().eval()
    groups = _fp32_groups(model, calib[:2])
    float_layers = []
    for i, (lin, bn, relu) in enumerate(groups):
        W = lin.weight.detach().double().cpu().numpy()
        b = lin.bias.detach().double().cpu().numpy() if lin.bias is not None else np.zeros(W.shape[0])
        if bn is not None:
            g = (bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)).numpy()
            b = g * (b - bn.running_mean.detach().double().numpy()) + bn.bias.detach().double().numpy()
            W = W * g[:, None]
        if i == 0:                                     # x = (p / 255 - m) / s  ->  weights on p / 255
            s = std[0] if std else 1.0
            n_in = W.shape[1]
            m = np.repeat(np.asarray(mean, dtype=np.float64), n_in // len(mean)) if mean else np.zeros(n_in)
            b = b - (W * m[None, :]).sum(1) / s
            W = W / s
        float_layers.append((W, b, relu))
    # calibration of every hidden output range (after its ReLU), in float, from the folded layers
    a = (calib.reshape(len(calib), -1).double().numpy() * (std[0] if std else 1.0)
         + (np.repeat(np.asarray(mean), calib[0].numel() // len(mean)) if mean else 0.0))          # = p / 255
    maxima = []
    for W, b, relu in float_layers[:-1]:
        a = a @ W.T + b
        if relu:
            a = np.maximum(a, 0)
        maxima.append(max(float(np.percentile(a, 99.99)), 1e-6))
    s_in = 1.0 / 255.0
    out = []
    for i, (W, b, relu) in enumerate(float_layers):
        s_w = max(np.abs(W).max(), 1e-12) / 127.0
        Wq = np.clip(np.rint(W / s_w), -127, 127).astype(np.int8)
        bias = np.rint(b / (s_in * s_w)).astype(np.int64)
        if i == len(float_layers) - 1:
            out.append(dict(W=Wq, bias=bias, mult=None, shift=None))
        else:
            if not relu:
                raise ValueError("INT8 reference: hidden layers must end in ReLU (unsigned activations)")
            s_out = maxima[i] / 255.0
            mult, shift = quantize_multiplier(s_in * s_w / s_out)
            out.append(dict(W=Wq, bias=bias, mult=mult, shift=shift))
            s_in = s_out
    return out


def simulate_int8(q, x_u8):
    """Exact integer simulation of the INT8 reference (ESP-NN / TFLite arithmetic). Returns the classes."""
    a = x_u8.astype(np.int64) - 128                                       # int8, zero point -128
    for c in q:
        acc = (a + 128) @ c["W"].astype(np.int64).T + c["bias"]          # input_offset = +128
        if c["mult"] is None:
            return acc.argmax(1)                                          # last layer: argmax in int32
        a = np.clip(quantized_multiply(acc, c["mult"], c["shift"]) - 128, -128, 127)


def write_header_int8(q, path):
    """model_int8.h: symbex_int8_infer(const uint8_t* x). With SYMBEX_WITH_ESP_NN it uses
    esp_nn_fully_connected_s8 (Espressif's optimized kernel); otherwise a plain C version of the same
    arithmetic (used to check the header on the PC)."""
    dims = [q[0]["W"].shape[1]] + [c["W"].shape[0] for c in q]
    L = [f"/* Generated by the SYMBEX-1 CDT: INT8 reference (post-training, TFLite / ESP-NN arithmetic). */",
         "#ifndef SYMBEX_INT8_H", "#define SYMBEX_INT8_H", "", "#include <stdint.h>", "#include <string.h>",
         "#ifdef SYMBEX_WITH_ESP_NN", '#include "esp_nn.h"', "#endif",
         "/* Where the INT8 weights live: const = flash (default); -DSYMBEX_INT8_ROM= (empty) puts them in RAM.",
         " * Separate from SYMBEX_ROM, so the binary model can go to RAM while the INT8 one stays in flash. */",
         "#ifndef SYMBEX_INT8_ROM", "#define SYMBEX_INT8_ROM const", "#endif",
         "#ifndef INT8_MARK", "#define INT8_MARK(k)", "#endif",
         f"#define INT8_N_LAYERS {len(q)}", f"#define INT8_N_INPUTS {dims[0]}",
         f"#define INT8_BYTES_BUFFERS {2 * max(dims[1:-1] + [dims[0]])}", ""]
    for k, c in enumerate(q):
        W = c["W"].ravel()
        L.append(f"static SYMBEX_INT8_ROM int8_t int8_w{k}[{len(W)}] __attribute__((aligned(16))) = {{")
        for j in range(0, len(W), 32):
            L.append("    " + ", ".join(str(int(v)) for v in W[j:j + 32]) + ",")
        L.append("};")
        L.append(f"static SYMBEX_INT8_ROM int32_t int8_b{k}[{len(c['bias'])}] = {{" + ", ".join(str(int(v)) for v in c["bias"]) + "};")
    L.append("__attribute__((unused)) static const int INT8_DIMS[] = {" + ", ".join(map(str, dims)) + "};")
    L.append("__attribute__((unused)) static const int32_t INT8_MULT[] = {" + ", ".join([str(c["mult"]) for c in q[:-1]] + ["0"]) + "};")
    L.append("__attribute__((unused)) static const int32_t INT8_SHIFT[] = {" + ", ".join([str(c["shift"]) for c in q[:-1]] + ["0"]) + "};")
    L += ["", "#ifndef SYMBEX_WITH_ESP_NN",
          "/* TFLite MultiplyByQuantizedMultiplier, the same arithmetic as ESP-NN */",
          "__attribute__((unused)) static inline int32_t int8_mbqm(int32_t x, int32_t mult, int shift) {",
          "    int left = shift > 0 ? shift : 0, right = shift > 0 ? 0 : -shift;",
          "    int64_t ab = (int64_t)x * (int64_t)(1 << left) * (int64_t)mult;",
          "    int64_t r = ab + (ab >= 0 ? (1LL << 30) : (1 - (1LL << 30)));",
          "    int64_t high = r >= 0 ? (r >> 31) : -((-r) >> 31);",
          "    if (high > INT32_MAX) high = INT32_MAX;",
          "    if (high < INT32_MIN) high = INT32_MIN;",
          "    if (right == 0) return (int32_t)high;",
          "    int32_t a = (int32_t)high, m = (int32_t)((1LL << right) - 1);",
          "    return (a >> right) + ((a & m) > ((m >> 1) + (a < 0)));",
          "}",
          "__attribute__((unused)) static void int8_fc(const int8_t* a, int n_in, const int8_t* w, const int32_t* b, int8_t* o, int n_out,",
          "                    int32_t mult, int shift) {",
          "    for (int j = 0; j < n_out; j++) {",
          "        int32_t acc = b[j];",
          "        const int8_t* wj = &w[j * n_in];",
          "        for (int i = 0; i < n_in; i++) acc += ((int32_t)a[i] + 128) * wj[i];",
          "        int32_t v = int8_mbqm(acc, mult, shift) - 128;",
          "        o[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));",
          "    }",
          "}",
          "#endif", ""]
    maxd = max(dims[:-1])
    L += [f"static inline int symbex_int8_infer(const uint8_t* x) {{",
          f"    static int8_t a[{maxd}] __attribute__((aligned(16))), b[{maxd}] __attribute__((aligned(16)));",
          f"    INT8_MARK(0);",
          f"    for (int i = 0; i < {dims[0]}; i++) a[i] = (int8_t)((int)x[i] - 128);"]
    for k in range(len(q) - 1):
        L += ["#ifdef SYMBEX_WITH_ESP_NN",
              f"    esp_nn_fully_connected_s8(a, 128, {dims[k]}, int8_w{k}, 0, int8_b{k}, b, {dims[k + 1]}, -128,",
              f"                              INT8_SHIFT[{k}], INT8_MULT[{k}], -128, 127);",
              "#else",
              f"    int8_fc(a, {dims[k]}, int8_w{k}, int8_b{k}, b, {dims[k + 1]}, INT8_MULT[{k}], INT8_SHIFT[{k}]);",
              "#endif",
              f"    memcpy(a, b, {dims[k + 1]});",
              f"    INT8_MARK({k + 1});"]
    k = len(q) - 1
    L += ["    (void)b;", "    int best = 0;", "    int32_t max_acc = 0;",
          f"    for (int j = 0; j < {dims[k + 1]}; j++) {{",
          f"        int32_t acc = int8_b{k}[j];",
          f"        const int8_t* w = &int8_w{k}[j * {dims[k]}];",
          f"        for (int i = 0; i < {dims[k]}; i++) acc += ((int32_t)a[i] + 128) * w[i];",
          "        if (j == 0 || acc > max_acc) { max_acc = acc; best = j; }",
          "    }",
          f"    INT8_MARK({len(q)});",
          "    return best;", "}", "", "#endif /* SYMBEX_INT8_H */", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    return sum(c["W"].size + 4 * len(c["bias"]) for c in q)


def export_int8(original_model, calib, val_xs, x_u8, ys, out_dir, lib_path, mean=(), std=(), verbose=True):
    """Builds the INT8 reference of the original model, writes model_int8.h + validation_int8.u8, and checks
    its plain C version against the numpy simulation on this PC. Returns a dict with its numbers."""
    q = prepare_int8(original_model, calib, mean, std)
    path = os.path.join(out_dir, "model_int8.h")
    flash = write_header_int8(q, path)
    pred = simulate_int8(q, x_u8)
    np.asarray(pred, dtype=np.uint8).tofile(os.path.join(out_dir, "validation_int8.u8"))
    with torch.no_grad():
        pred_fp = original_model.cpu().eval()(val_xs).argmax(1).numpy()
    exact = None
    if shutil.which("gcc"):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(path, os.path.join(d, "model_int8.h"))
            with open(os.path.join(d, "main.c"), "w") as f:
                f.write('#include <stdio.h>\n#include <stdlib.h>\n#include "model_int8.h"\n'
                        "int main(int argc, char** argv) { (void)argc; FILE* f = fopen(argv[1], \"rb\");\n"
                        "  static uint8_t x[INT8_N_INPUTS]; int n = atoi(argv[2]);\n"
                        "  for (int i = 0; i < n; i++) { if (fread(x, 1, INT8_N_INPUTS, f) != INT8_N_INPUTS) return 1;\n"
                        "    printf(\"%d\\n\", symbex_int8_infer(x)); } return 0; }\n")
            exe = os.path.join(d, "p")
            r = subprocess.run(["gcc", "-O2", "-std=c99", "-I", d, "-o", exe, os.path.join(d, "main.c")],
                               capture_output=True, text=True)
            if r.returncode == 0:
                np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(os.path.join(d, "x.u8"))
                out = subprocess.run([exe, os.path.join(d, "x.u8"), str(len(x_u8))], capture_output=True, text=True).stdout
                exact = 100 * float((np.array([int(v) for v in out.split()]) == pred).mean())
    res = dict(acc_int8=100 * float((pred == ys).mean()), fp32_agreement=100 * float((pred == pred_fp).mean()),
               flash_bytes=flash, c_exactness=exact, header=path)
    if verbose:
        print(f"    INT8 reference (original model, post-training like TFLite / ESP-NN): accuracy {res['acc_int8']:.2f}%"
              f" | same class as FP32 {res['fp32_agreement']:.2f}% | {flash / 1024:.1f} KB"
              + ("" if exact is None else f" | C vs. simulation {exact:.2f}%"))
    return res
