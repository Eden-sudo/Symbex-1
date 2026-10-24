"""
ternary.py -- export + validation of a ternary MLP (method "ternary", every layer "bin") to lib_symbex.

Same structure and folding as dense.py (the plain 1-bit MLP), with ternary weights:
    fold()               the trained model -> integers: {-1, 0, +1} codes, one int32 threshold per hidden neuron,
                         int64 scale + int32 shift per output class
    write_header()       model_symbex.h: sign plane + nonzero-mask plane per layer, and symbex_mlp_infer()
                         using symbex_ternary.c (same entry point and macros as the plain header)
    validate_c()         compiles the header with lib_symbex on this PC and checks that the C gives EXACTLY the
                         predictions of the integer simulation (must be 100%)
    export_and_validate()  all of it, plus the bit-level check of validate/ternary.py (packed-word simulator vs.
                         numpy integer reference, layer by layer)

Supported structure: DenseTernary -> [BatchNorm1d] -> [ReLU] -> ... -> DenseTernary -> [BatchNorm1d].

Folding, per neuron j (raw_j = sum_i t_ij * x_i, an integer; x_i = +-1, or the uint8 input in the first layer):
    z_j = A_j * raw_j + B_j       A_j = alpha_j (mean |W| over the nonzero weights) x BatchNorm (x input scale)
    bit_j = [ReLU(z_j) >= beta_j] = [raw_j >= T_j], codes of row j negated when A_j < 0    (dense._threshold)
Weight bytes: 2 bits per weight (two planes), each row padded to whole 32-bit words.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from .dense import Q_ACT, Q_MULT, INT32_MIN, _pack, _threshold, to_uint8, analyze, save_validation
from ..methods.dense.ternary import DenseTernary, ternary_codes, ternary_delta


# ---------------------------------------------------------------------------------------------------
# 1. folding to integers
# ---------------------------------------------------------------------------------------------------
def _codes_and_affine(layer, bn):
    """({-1, 0, +1} codes as int64, A, B before the input fold). The codes come from the SAME function the
    forward pass uses (float32 comparison), so no weight on the threshold flips between PyTorch and C."""
    with torch.no_grad():
        t = ternary_codes(layer.weight.detach().cpu(), layer.delta_ratio.detach().cpu())
    codes = t.numpy().astype(np.int64)
    w = layer.weight.detach().double().cpu().numpy()
    mask = codes != 0
    count = mask.sum(1)
    a = np.where(count > 0, (np.abs(w) * mask).sum(1) / np.maximum(count, 1), 0.0)
    b = layer.bias.detach().double().cpu().numpy() if layer.bias is not None else np.zeros(w.shape[0])
    if bn is not None:
        g = (bn.weight.detach().double().cpu() / torch.sqrt(bn.running_var.detach().double().cpu() + bn.eps)).numpy()
        b = g * (b - bn.running_mean.detach().double().cpu().numpy()) + bn.bias.detach().double().cpu().numpy()
        a = g * a
    return codes, a, b


def fold(groups, mean=(), std=()):
    """Folds the trained ternary layers into integers. Returns a list of dicts, one per layer:
    codes (out, in) int64 in {-1, 0, +1}, u8_input, last, and threshold (hidden) or scale + shift (output)."""
    if std and len(set(std)) != 1:
        raise ValueError("ternary export: the input normalization must use ONE std for every channel")
    s = std[0] if std else 1.0
    layers = []
    for i, (layer, bn, relu) in enumerate(groups):
        codes, a, b = _codes_and_affine(layer, bn)
        if i == 0:
            # x_i = (p_i / 255 - m_c) / s  ->  sum t_ij x_i = raw_j / (255 s) - sum_i m_c(i) t_ij / s
            n_in = codes.shape[1]
            if mean:
                if n_in % len(mean):
                    raise ValueError(f"ternary export: {n_in} inputs cannot be split into {len(mean)} channels")
                m_per_input = np.repeat(np.asarray(mean, dtype=np.float64), n_in // len(mean))
            else:
                m_per_input = np.zeros(n_in)
            D = (codes * m_per_input[None, :]).sum(1) / s
            A, B = a / (255.0 * s), b - a * D
        else:
            A, B = a, b
        d = dict(codes=codes, A=A, B=B, u8_input=(i == 0), last=(i == len(groups) - 1),
                 delta=float(ternary_delta(layer.weight.detach().cpu(), layer.delta_ratio.detach().cpu())),
                 zeros=float((codes == 0).mean()))
        if not d["last"]:
            next_layer = groups[i + 1][0]
            if not hasattr(next_layer.rsign, "beta"):
                raise ValueError(f"ternary export: layer {i + 1} does not binarize its input (binarize_input=False); "
                                 "only the first layer may take real inputs")
            betas = next_layer.rsign.beta.detach().double().cpu().numpy().ravel()
            T = np.zeros(len(A), dtype=np.int64)
            for j in range(len(A)):
                neg, T[j] = _threshold(A[j], B[j], betas[j], relu)
                if neg:
                    d["codes"][j] = -d["codes"][j]
            d["threshold"] = T
        else:
            d["scale"] = np.round(A * 2 ** Q_MULT).astype(np.int64)
            d["shift"] = np.round(B * 2 ** Q_ACT).astype(np.int64)
            if np.abs(d["shift"]).max() >= 2 ** 31:
                raise ValueError("ternary export: an output bias does not fit in int32 with Q_ACT = 16")
        layers.append(d)
    return layers


def planes(codes):
    """(out, in) {-1, 0, +1} -> (sign words, mask words, words per row). Same packing as dense._pack:
    ceil(in / 32) words per row, bit i of word i / 32 = input i. Sign bit = 1 for +1; mask bit = 1 for nonzero.
    Zero weights get sign bit 0 (never read: the mask hides them)."""
    signs, words = _pack(np.where(codes > 0, 1, -1))
    mask, _ = _pack(np.where(codes != 0, 1, -1))
    return signs, mask, words


def weight_bytes(layers):
    """Bytes of the two weight planes (what the flash holds for the weights alone)."""
    return sum(2 * planes(d["codes"])[0].size * 4 for d in layers)


def flash_bytes(layers):
    """Weights (2 planes) + int32 thresholds per hidden neuron + int64 scale and int32 shift per class."""
    return weight_bytes(layers) + sum(d["codes"].shape[0] * (12 if d["last"] else 4) for d in layers)


# ---------------------------------------------------------------------------------------------------
# 2. C header (same entry point and macros as dense.write_header, so the same test harness runs it)
# ---------------------------------------------------------------------------------------------------
def _words(lines, ctype, name, matrix):
    lines.append(f"static SYMBEX_ROM {ctype} {name}[{matrix.size}] = {{")
    for row in matrix:
        lines.append("    " + ", ".join(f"0x{int(v):08X}u" for v in row) + ",")
    lines.append("};")


def write_header(layers, path, name="symbex_mlp"):
    N = name.upper()
    n_in0 = layers[0]["codes"].shape[1]
    max_n = max(d["codes"].shape[0] for d in layers)
    steps = [f"layer {i}: {d['codes'].shape[1]} -> {d['codes'].shape[0]} "
             f"({'uint8' if d['u8_input'] else '1 bit'} x ternary) + {'argmax' if d['last'] else 'thresholds'}"
             for i, d in enumerate(layers)]
    lines = ["/* Generated by the SYMBEX-1 CDT (export/ternary.py, method \"ternary\"). Do not edit by hand. */",
             f"#ifndef {N}_H", f"#define {N}_H", "",
             "#include <stdint.h>", '#include "core/symbex_core.h"', '#include "layers/dense/symbex_dense.h"',
             '#include "layers/dense/symbex_ternary.h"', "",
             f"#define {N}_TERNARY 1", f"#define {N}_Q_ACT {Q_ACT}", f"#define {N}_Q_MULT {Q_MULT}",
             f"#define {N}_N_INPUTS {n_in0}", f"#define {N}_N_CLASSES {layers[-1]['codes'].shape[0]}", "",
             "/* Where the weights live: const = flash (default). -DSYMBEX_ROM= (empty) puts them in RAM. */",
             "#ifndef SYMBEX_ROM", "#define SYMBEX_ROM const", "#endif",
             "/* Per-layer profiling: define SYMBEX_MARK(k) before including this header (k = 0 before the first",
             " * layer, k = i + 1 after layer i). Empty by default: no cost. */",
             "#ifndef SYMBEX_MARK", "#define SYMBEX_MARK(k)", "#endif",
             f"#define {N}_N_STEPS {len(layers)}",
             f"#define {N}_BYTES_BUFFERS {4 * max_n + 2 * 4 * ((max_n + 31) // 32)}   /* static RAM of {name}_infer */",
             "#ifdef SYMBEX_PROFILE",
             f"static const char* const {name}_steps[{len(layers)}] = {{" + ", ".join(f'"{p}"' for p in steps) + "};",
             "#endif", ""]
    for i, d in enumerate(layers):
        out, n_in = d["codes"].shape
        s, m, _ = planes(d["codes"])
        lines.append(f"/* layer {i}: {n_in} -> {out}, ternary ({100 * d['zeros']:.1f}% zeros)"
                     f"{' (uint8 input)' if d['u8_input'] else ''} */")
        _words(lines, "uint32_t", f"{name}_s{i}", s)
        _words(lines, "uint32_t", f"{name}_m{i}", m)
        lines.append(f"static const struct SymbexTernaryLayerData {name}_layer{i} = "
                     f"{{ {n_in}, {out}, {name}_s{i}, {name}_m{i} }};")
        if not d["last"]:
            lines.append(f"static SYMBEX_ROM int32_t {name}_t{i}[{out}] = {{"
                         + ", ".join("INT32_MIN" if t == INT32_MIN else str(int(t)) for t in d["threshold"]) + "};")
        else:
            lines.append(f"static SYMBEX_ROM int64_t {name}_scale[{out}] = {{"
                         + ", ".join(f"{int(v)}LL" for v in d["scale"]) + "};")
            lines.append(f"static SYMBEX_ROM int32_t {name}_shift[{out}] = {{"
                         + ", ".join(str(int(v)) for v in d["shift"]) + "};")
        lines.append("")
    lines += [f"/* Full inference: x = {n_in0} uint8 inputs (training tensor order, flattened). Returns the class.",
              " * Not reentrant (static buffers). */",
              f"static inline int {name}_infer(const uint8_t* x) {{",
              f"    static int32_t raw[{max_n}];",
              f"    static uint32_t bits_a[{(max_n + 31) // 32}], bits_b[{(max_n + 31) // 32}];",
              "    uint32_t* in_bits = bits_a;", "    uint32_t* out_bits = bits_b;", "    SYMBEX_MARK(0);"]
    for i, d in enumerate(layers):
        if d["u8_input"]:
            lines.append(f"    symbex_ternary_input_u8_mask(x, raw, &{name}_layer{i});")
        else:
            lines.append(f"    symbex_ternary_forward(in_bits, raw, &{name}_layer{i});")
        if not d["last"]:
            lines.append(f"    symbex_dense_threshold(raw, {name}_t{i}, out_bits, {d['codes'].shape[0]});")
            lines.append("    { uint32_t* t = in_bits; in_bits = out_bits; out_bits = t; }")
            lines.append(f"    SYMBEX_MARK({i + 1});")
    sh = Q_MULT - Q_ACT
    lines += ["    int best = 0;", "    int64_t max_logit = INT64_MIN;",
              f"    for (int j = 0; j < {layers[-1]['codes'].shape[0]}; j++) {{",
              f"        int64_t logit = (((int64_t)raw[j] * {name}_scale[j] + ((int64_t)1 << {sh - 1})) >> {sh})"
              f" + {name}_shift[j];",
              "        if (logit > max_logit) { max_logit = logit; best = j; }",
              "    }", f"    SYMBEX_MARK({len(layers)});", "    (void)in_bits;", "    (void)out_bits;",
              "    return best;", "}", "", f"#endif /* {N}_H */", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------------------------------
# 3. exactness of the compiled C on this PC
# ---------------------------------------------------------------------------------------------------
_TEST_MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "model_symbex.h"
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    int n = atoi(argv[2]);
    static uint8_t x[SYMBEX_MLP_N_INPUTS];
    for (int i = 0; i < n; i++) {
        if (fread(x, 1, SYMBEX_MLP_N_INPUTS, f) != SYMBEX_MLP_N_INPUTS) return 1;
        printf("%d\n", symbex_mlp_infer(x));
    }
    fclose(f);
    return 0;
}
"""


def validate_c(header_path, lib_path, x_u8, expected, flags=()):
    """Compiles the header + lib_symbex (ternary + dense thresholds + generic HAL) with gcc, runs it on x_u8.
    Returns (equal, total, message). `flags`: extra compiler flags (e.g. -DSYMBEX_POPCOUNT_FORCE_SWAR)."""
    if shutil.which("gcc") is None:
        return None, 0, "gcc not found: the C exactness check was skipped"
    inc = os.path.join(lib_path, "include")
    sources = [os.path.join(lib_path, "src", "layers", "dense", "symbex_ternary.c"),
               os.path.join(lib_path, "src", "layers", "dense", "symbex_dense.c"),
               os.path.join(lib_path, "src", "ops", "symbex_and.c"),
               os.path.join(lib_path, "src", "hal", "generic", "symbex_hal_impl.c")]
    for r in [inc] + sources:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing): set [output] lib_path"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header_path, os.path.join(d, "model_symbex.h"))
        with open(os.path.join(d, "main.c"), "w") as f:
            f.write(_TEST_MAIN)
        exe = os.path.join(d, "check")
        r = subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", *flags, "-I", inc, "-I", d, "-o", exe,
                            os.path.join(d, "main.c")] + sources, capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C did not compile:\n" + r.stderr[-2000:]
        data = os.path.join(d, "x.u8")
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(data)
        r = subprocess.run([exe, data, str(len(x_u8))], capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C program failed while running"
        pred_c = np.array([int(v) for v in r.stdout.split()])
    return int((pred_c == expected).sum()), len(expected), ""


# ---------------------------------------------------------------------------------------------------
# 4. everything together
# ---------------------------------------------------------------------------------------------------
def export_and_validate(model, val_dataset, out_dir, lib_path, mean=(), std=(), n_validation=2000, verbose=True,
                        **_unused):
    """Exports `model` (all layers 'bin', method ternary) and validates it. Returns a dict with the numbers
    (same keys as dense.export_and_validate, plus bit_exact, zeros and weight_bytes). `_unused`: the plain
    exporter's arithmetic / input_bits options, which do not apply here."""
    from ..validate.ternary import verify_bit_exact
    n = min(n_validation, len(val_dataset))
    xs = torch.stack([val_dataset[i][0] for i in range(n)])
    ys = np.array([int(val_dataset[i][1]) for i in range(n)])
    model = model.cpu().eval()
    groups = analyze(model, xs[:2], cls=DenseTernary)
    layers = fold(groups, mean, std)
    header_path = os.path.join(out_dir, "model_symbex.h")
    write_header(layers, header_path)
    x_u8 = to_uint8(xs, mean, std)
    captured, hooks = {}, []
    for i, (layer, _, _) in enumerate(groups[1:]):
        hooks.append(layer.rsign.register_forward_hook(lambda m, e, s_, i=i: captured.__setitem__(i, s_.detach())))
    with torch.no_grad():
        pred_pt = model(xs).argmax(1).numpy()
    for h in hooks:
        h.remove()
    exact = verify_bit_exact(layers, x_u8)
    pred_sim, bits_sim = exact["_pred"], exact["_bits"]
    bit_agreement = [100 * float((captured[i].numpy() == bits_sim[i]).mean()) for i in range(len(bits_sim))]
    equal_c, total_c, msg = validate_c(header_path, lib_path, x_u8, pred_sim)
    save_validation(out_dir, x_u8, ys, pred_sim)
    flash, weights = flash_bytes(layers), weight_bytes(layers)
    res = dict(n=n, method="ternary", acc_pytorch=100 * float((pred_pt == ys).mean()),
               acc_integer=100 * float((pred_sim == ys).mean()),
               fidelity=100 * float((pred_pt == pred_sim).mean()), bit_agreement=bit_agreement,
               bit_exact={k: v for k, v in exact.items() if not k.startswith("_")},
               zeros=[d["zeros"] for d in layers], delta=[d["delta"] for d in layers],
               weight_bytes=weights, flash_bytes=flash, header=header_path,
               c_exactness=None if equal_c is None else 100 * equal_c / total_c, c_note=msg,
               _x_u8=x_u8, _y=ys, _pred_sim=pred_sim, _xs=xs, _layers=layers)
    if verbose:
        print("    layers      : " + " -> ".join(str(d["codes"].shape[1]) for d in layers)
              + f" -> {layers[-1]['codes'].shape[0]}   (ternary, first layer: uint8 input)")
        print("    zeros       : " + ", ".join(f"layer {i} {100 * d['zeros']:.1f}%" for i, d in enumerate(layers)))
        print(f"    header      : {header_path}  ({weights / 1024:.1f} KB of weights (2 bit planes), "
              f"{flash / 1024:.1f} KB with thresholds / scales)")
        print(f"    validation  : {n} samples")
        print(f"    accuracy    : PyTorch {res['acc_pytorch']:.2f}%   integer simulation {res['acc_integer']:.2f}%")
        print(f"    FIDELITY    : {res['fidelity']:.2f}%  (same prediction PyTorch vs. integers: what folding costs)")
        for i, a in enumerate(bit_agreement):
            print(f"      hidden layer {i}: {a:.3f}% of the bits equal PyTorch vs. integers"
                  + ("" if a >= 99.5 else "   [!] low: a bug in the folding, not rounding"))
        be = res["bit_exact"]
        print(f"    BIT-EXACT   : packed-word simulator vs. numpy integer reference: "
              f"{'OK' if be['equal'] else '[X] MISMATCH'} ({be['layers']} layers, raw scores + logits, {n} samples)")
        if res["c_exactness"] is None:
            print(f"    EXACTNESS   : not checked -- {msg}")
        else:
            status = "OK" if res["c_exactness"] == 100 else "[X] MISMATCH: bug in the export or the C engine"
            print(f"    EXACTNESS   : {res['c_exactness']:.2f}%  (compiled C vs. integer simulation, must be 100%) {status}")
    return res
