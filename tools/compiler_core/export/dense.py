"""
dense.py -- export + validation of a 1-bit MLP (method plain, every layer "bin") to lib_symbex.

Export and validation live together because they must use the SAME folded integers:
    fold()            the trained model -> integers: 1-bit weights, one int32 threshold per hidden neuron,
                        int64 scale + int32 shift per output class
    simulate()           exact integer simulation of the C engine in numpy (the reference for the C)
    write_header()   model_symbex.h: arrays + symbex_mlp_infer(const uint8_t* x) using lib_symbex
    validate_c()         compiles the header with lib_symbex on this PC (gcc) and checks that the C gives
                        EXACTLY the predictions of simulate() (must be 100%)
    export_and_validate()  all of it, with a report
The test on the board lives in compiler_core/chip/ (measure.py); the INT8 reference in export/int8_reference.py.

Supported structure (in forward order; Flatten / Dropout / Identity are skipped):
    DensePlain -> [BatchNorm1d] -> [ReLU] -> DensePlain -> ... -> DensePlain -> [BatchNorm1d]
The first layer keeps a real input (as trained): on the chip it receives uint8 values (e.g. pixels),
and the input normalization ((x / 255 - mean) / std, one std for all channels) is folded into its thresholds.

Folding, per hidden neuron j (raw_j = sum of +-input x +-1 weight, an integer):
    z_j = A_j * raw_j + B_j          (weight scale alpha, bias, input normalization, BatchNorm)
    bit_j = [ReLU(z_j) >= beta_j]    (beta_j = binarization threshold of the next layer's input)
          = [raw_j >= T_j]           with the weight row negated when A_j < 0 (so the C only does >=)
Output: logit_j = A_j * raw_j + B_j in fixed point (int64 scale with Q_MULT bits, int32 shift with Q_ACT).
Two numbers are reported: FIDELITY (PyTorch float model vs. integer simulation: what folding to integers
costs) and EXACTNESS (integer simulation vs. compiled C: must be 100%, otherwise there is a bug).
"""
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn

Q_ACT, Q_MULT = 16, 30
INT32_MIN, INT32_MAX = -2 ** 31, 2 ** 31 - 1


# ---------------------------------------------------------------------------------------------------
# 1. structure
# ---------------------------------------------------------------------------------------------------
def _leaves_in_order(model, x, cls=None):
    """Modules in the order the forward pass runs them: every DensePlain (or `cls`) as ONE unit (its inner
    RSign is part of it) and every other leaf module that is not inside one."""
    from ..methods.dense.plain import DensePlain
    cls = cls or DensePlain
    units = [n for n, m in model.named_modules() if isinstance(m, cls)]
    order, hooks = [], []
    for n, m in model.named_modules():
        inside = any(n.startswith(u + ".") for u in units)
        if isinstance(m, cls) or (len(list(m.children())) == 0 and not inside):
            hooks.append(m.register_forward_hook(lambda mod, e, s: order.append(mod)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(x)
    model.train(was_training)
    for h in hooks:
        h.remove()
    seen, res = set(), []
    for m in order:
        if id(m) not in seen:
            seen.add(id(m))
            res.append(m)
    return res


def analyze(model, x, cls=None):
    """Returns the layer groups [(DensePlain, BatchNorm1d or None, has ReLU)], or raises ValueError
    explaining what is not supported. `cls`: the layer class to look for (DensePlain by default)."""
    from ..methods.dense.plain import DensePlain
    cls = cls or DensePlain
    groups, current = [], None
    for m in _leaves_in_order(model, x, cls):
        name = type(m).__name__
        if isinstance(m, (nn.Flatten, nn.Dropout, nn.Identity)):
            continue
        if isinstance(m, cls):
            current = [m, None, False]
            groups.append(current)
        elif isinstance(m, nn.BatchNorm1d) and current is not None and current[1] is None and not current[2]:
            current[1] = m
        elif isinstance(m, nn.ReLU) and current is not None and not current[2]:
            current[2] = True
        else:
            raise ValueError(f"dense export: unsupported module {name} (supported: {cls.__name__} -> "
                             f"[BatchNorm1d] -> [ReLU], repeated). k-bit layers cannot be exported yet.")
    if not groups:
        raise ValueError(f"dense export: no {cls.__name__} layer found (is every layer 'bin'?)")
    if groups[-1][2]:
        raise ValueError("dense export: the last layer cannot have a ReLU (it produces the logits)")
    return [tuple(g) for g in groups]


# ---------------------------------------------------------------------------------------------------
# 2. folding to integers
# ---------------------------------------------------------------------------------------------------
def _affine(layer, bn):
    """(alpha, bias) of the layer combined with its BatchNorm: y = a * (sum of +-x * s) + b, as numpy float64."""
    w = layer.weight.detach().double().cpu()
    a = w.abs().mean(dim=1)
    b = layer.bias.detach().double().cpu() if layer.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    if bn is not None:
        g = bn.weight.detach().double().cpu() / torch.sqrt(bn.running_var.detach().double().cpu() + bn.eps)
        b = g * (b - bn.running_mean.detach().double().cpu()) + bn.bias.detach().double().cpu()
        a = g * a
    return a.numpy(), b.numpy()


def _threshold(A, B, beta, relu):
    """Integer threshold for bit = [act(A * raw + B) >= beta]. Returns (negate the row?, T)."""
    if relu and beta <= 0:
        return False, INT32_MIN                         # ReLU(z) >= 0 >= beta: always 1
    if A == 0:
        return False, (INT32_MIN if B >= beta else INT32_MAX)
    v = (beta - B) / A
    if A > 0:                                           # raw >= v
        t, neg = math.ceil(v), False
    else:                                               # raw <= v  <=>  -raw >= -floor(v)
        t, neg = -math.floor(v), True
    return neg, int(min(max(t, INT32_MIN), INT32_MAX))


def fold(groups, mean=(), std=()):
    """Folds the trained layers into integers. Returns a list of dicts, one per layer."""
    if std and len(set(std)) != 1:
        raise ValueError("dense export: the input normalization must use ONE std for every channel")
    s = std[0] if std else 1.0
    layers = []
    for i, (layer, bn, relu) in enumerate(groups):
        w = layer.weight.detach().cpu().numpy()
        signs = np.where(w >= 0, 1, -1).astype(np.int64)          # 0 -> +1 (same as the C engine)
        a, b = _affine(layer, bn)
        if i == 0:
            # input x_i = (p_i / 255 - m_c) / s  ->  sum x_i s_ij = raw_j / (255 s) - sum_i m_c(i) s_ij / s
            # (the flattened input is channel-major: feature i belongs to channel i // (n_in / C))
            n_in = w.shape[1]
            if mean:
                if n_in % len(mean):
                    raise ValueError(f"dense export: {n_in} inputs cannot be split into {len(mean)} channels")
                m_per_input = np.repeat(np.asarray(mean, dtype=np.float64), n_in // len(mean))
            else:
                m_per_input = np.zeros(n_in)
            D = (signs * m_per_input[None, :]).sum(1) / s
            A, B = a / (255.0 * s), b - a * D
        else:
            A, B = a, b
        d = dict(signs=signs, A=A, B=B, u8_input=(i == 0), last=(i == len(groups) - 1))
        if not d["last"]:
            next_layer = groups[i + 1][0]
            if not hasattr(next_layer.rsign, "beta"):
                raise ValueError(f"dense export: layer {i + 1} does not binarize its input (binarize_input=False); "
                                 "only the first layer may take real inputs")
            betas = next_layer.rsign.beta.detach().double().cpu().numpy().ravel()
            T = np.zeros(len(A), dtype=np.int64)
            for j in range(len(A)):
                neg, T[j] = _threshold(A[j], B[j], betas[j], relu)
                if neg:
                    d["signs"][j] = -d["signs"][j]
            d["threshold"] = T
        else:
            d["scale"] = np.round(A * 2 ** Q_MULT).astype(np.int64)
            d["shift"] = np.round(B * 2 ** Q_ACT).astype(np.int64)
            if np.abs(d["shift"]).max() >= 2 ** 31:
                raise ValueError("dense export: an output bias does not fit in int32 with Q_ACT = 16")
        layers.append(d)
    return layers


# ---------------------------------------------------------------------------------------------------
# 3. exact integer simulation (what the C must reproduce bit for bit)
# ---------------------------------------------------------------------------------------------------
def simulate(layers, x_u8, return_bits=False):
    """x_u8: (N, n_inputs) uint8. Returns the predicted classes (N,), and with return_bits=True also the
    +-1 activations of every hidden layer (as the next layer receives them, BEFORE any row negation is undone:
    a negated row produces the same bit, so they compare directly with PyTorch)."""
    act = x_u8.astype(np.int64)
    bits = []
    for d in layers:
        raw = act @ d["signs"].T                          # +-1 inputs (or uint8 for the first layer) x +-1 weights
        if not d["last"]:
            act = np.where(raw >= d["threshold"][None, :], 1, -1)
            bits.append(act)
        else:
            sh = Q_MULT - Q_ACT
            logit = ((raw * d["scale"][None, :] + (1 << (sh - 1))) >> sh) + d["shift"][None, :]
            return (logit.argmax(1), bits) if return_bits else logit.argmax(1)


# ---------------------------------------------------------------------------------------------------
# 4. C header
# ---------------------------------------------------------------------------------------------------
def _pack(signs):
    """(out, in) +-1 -> uint32 words, ceil(in / 32) per row, bit i of word i / 32 = input i (+1 -> 1)."""
    out, n = signs.shape
    pal = (n + 31) // 32
    bits = np.zeros((out, pal * 32), dtype=np.uint64)
    bits[:, :n] = (signs > 0)
    packed = (bits.reshape(out, pal, 32) << np.arange(32, dtype=np.uint64)).sum(2)
    return packed.astype(np.uint64), pal


def write_header(layers, path, name="symbex_mlp", arithmetic="and", input_bits=8):
    """arithmetic = "and" (default): AND arithmetic of ops/symbex_and (rows padded to a multiple of 4 words and
    aligned to 16 bytes, count of +1 weights per neuron; the HAL uses SIMD where the chip has it).
    "xnor": the previous layout and kernels (symbex_dense). Both give the same raw scores, bit for bit.
    input_bits < 8 (XNOR): the inputs carry only their top input_bits bits ([data] input_bits), so the first layer
    uses only those bit planes (symbex_dense_input_u8_planes_bits: input_bits AND + popcount per weight word)."""
    if arithmetic == "and":
        return _write_header_and(layers, path, name)
    if arithmetic != "xnor":
        raise ValueError("arithmetic must be 'and' or 'xnor'")
    lines = ["/* Generated by the SYMBEX-1 CDT (export/dense.py). Do not edit by hand. */",
              f"#ifndef {name.upper()}_H", f"#define {name.upper()}_H", "",
              "#include <stdint.h>", '#include "core/symbex_core.h"', '#include "layers/dense/symbex_dense.h"', "",
              f"#define {name.upper()}_Q_ACT {Q_ACT}", f"#define {name.upper()}_Q_MULT {Q_MULT}",
              f"#define {name.upper()}_N_INPUTS {layers[0]['signs'].shape[1]}",
              f"#define {name.upper()}_N_CLASSES {layers[-1]['signs'].shape[0]}", "",
              "/* Where the weights live: const = flash (default). Compile with -DSYMBEX_ROM= (empty) to put them",
              " * in RAM (faster on chips whose flash cache is smaller than the model, costs that much RAM).",
              " * AVR: PROGMEM (flash), read by lib_symbex with SYMBEX_READ_W32 = pgm_read_dword. */",
              "#if defined(__AVR__)", "#include <avr/pgmspace.h>", "#endif",
              "#ifndef SYMBEX_ROM", "#if defined(__AVR__)", "#define SYMBEX_ROM const PROGMEM", "#else",
              "#define SYMBEX_ROM const", "#endif", "#endif",
              "/* an int64 constant that may live in flash (AVR) */",
              "static inline int64_t symbex_rom_i64(const int64_t* p) {",
              "#if defined(__AVR__)", "    int64_t v; memcpy_P(&v, p, sizeof v); return v;", "#else", "    return *p;", "#endif",
              "}",
              "/* Hidden layers write each neuron's bit directly (symbex_dense_*_bits): no raw-score array, so the",
              " * same header fits chips with 2 KB of RAM. */",
              ""]
    max_n = max(d["signs"].shape[0] for d in layers)
    steps = []
    for i, d in enumerate(layers):
        out, n_in = d["signs"].shape
        kind = ("uint8 x 1 bit" if input_bits == 8 else f"{input_bits}-bit input x 1 bit") if d["u8_input"] else "1 bit x 1 bit"
        end = "argmax" if d["last"] else "thresholds"
        steps.append(f"layer {i}: {n_in} -> {out} ({kind}) + {end}")
    lines += ["/* Per-layer profiling: define SYMBEX_MARK(k) before including this header to time each step",
               " * (k = 0 before the first layer, k = i + 1 after layer i). Empty by default: no cost. */",
               "#ifndef SYMBEX_MARK", "#define SYMBEX_MARK(k)", "#endif",
               f"#define {name.upper()}_N_STEPS {len(layers)}",
               f"#define {name.upper()}_INPUT_BITS {input_bits}",
               f"#define {name.upper()}_BYTES_BUFFERS {4 * layers[-1]['signs'].shape[0] + 2 * 4 * ((max_n + 31) // 32) + (4 * input_bits * ((layers[0]['signs'].shape[1] + 31) // 32) if input_bits < 8 else 0)}"
               "   /* static RAM of symbex_mlp_infer */",
               "#ifdef SYMBEX_PROFILE",
               f"static const char* const {name}_steps[{len(layers)}] = {{"
               + ", ".join(f'"{p}"' for p in steps) + "};",
               "#endif", ""]
    for i, d in enumerate(layers):
        out, n_in = d["signs"].shape
        packed, pal = _pack(d["signs"])
        lines.append(f"/* layer {i}: {n_in} -> {out}{' (uint8 input)' if d['u8_input'] else ''} */")
        lines.append(f"static SYMBEX_ROM uint32_t {name}_w{i}[{out * pal}] = {{")
        for row in packed:
            lines.append("    " + ", ".join(f"0x{int(v):08X}u" for v in row) + ",")
        lines.append("};")
        lines.append(f"static const struct SymbexLayerData {name}_layer{i} = {{ {n_in}, {out}, 1, {name}_w{i} }};")
        if not d["last"]:
            lines.append(f"static SYMBEX_ROM int32_t {name}_t{i}[{out}] = {{"
                          + ", ".join("INT32_MIN" if t == INT32_MIN else str(int(t)) for t in d["threshold"]) + "};")
        else:
            lines.append(f"static SYMBEX_ROM int64_t {name}_scale[{out}] = {{"
                          + ", ".join(f"{int(v)}LL" for v in d["scale"]) + "};")
            lines.append(f"static SYMBEX_ROM int32_t {name}_shift[{out}] = {{"
                          + ", ".join(str(int(v)) for v in d["shift"]) + "};")
        lines.append("")
    lines += [f"/* Full inference: x = {layers[0]['signs'].shape[1]} uint8 inputs (same order as the training",
               " * tensor, flattened). Returns the predicted class. Not reentrant (static buffers). */",
               f"static inline int {name}_infer(const uint8_t* x) {{",
               f"    static int32_t raw[{layers[-1]['signs'].shape[0]}];   /* last layer only: one per class */",
               f"    static uint32_t bits_a[{(max_n + 31) // 32}], bits_b[{(max_n + 31) // 32}];",
               *([f"    static uint32_t planes[{input_bits * ((layers[0]['signs'].shape[1] + 31) // 32)}];   /* top {input_bits} bit planes of the input */"]
                 if input_bits < 8 else []),
               "    uint32_t* in_bits = bits_a;", "    uint32_t* out_bits = bits_b;", "    SYMBEX_MARK(0);"]
    for i, d in enumerate(layers):
        if not d["last"]:
            if d["u8_input"] and input_bits < 8:
                lines.append(f"    symbex_dense_input_u8_planes_bits(x, {input_bits}, planes, {name}_t{i}, out_bits, &{name}_layer{i});")
            elif d["u8_input"]:
                lines.append(f"    symbex_dense_input_u8_bits(x, {name}_t{i}, out_bits, &{name}_layer{i});")
            else:
                lines.append(f"    symbex_dense_forward_bits(in_bits, {name}_t{i}, out_bits, &{name}_layer{i});")
            lines.append("    { uint32_t* t = in_bits; in_bits = out_bits; out_bits = t; }")
            lines.append(f"    SYMBEX_MARK({i + 1});")
        elif d["u8_input"]:
            lines.append(f"    symbex_dense_input_u8_mask(x, raw, &{name}_layer{i});")
        else:
            lines.append(f"    symbex_dense_forward(in_bits, raw, &{name}_layer{i});")
    sh = Q_MULT - Q_ACT
    lines += ["    int best = 0;", "    int64_t max_logit = INT64_MIN;",
               f"    for (int j = 0; j < {layers[-1]['signs'].shape[0]}; j++) {{",
               f"        int64_t logit = (((int64_t)raw[j] * symbex_rom_i64(&{name}_scale[j]) + ((int64_t)1 << {sh - 1})) >> {sh})"
               f" + (int32_t)SYMBEX_READ_W32((const uint32_t*)&{name}_shift[j]);",
               "        if (logit > max_logit) { max_logit = logit; best = j; }",
               "    }", f"    SYMBEX_MARK({len(layers)});", "    (void)in_bits;", "    (void)out_bits;", "    return best;", "}", "", f"#endif /* {name.upper()}_H */", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_header_and(layers, path, name):
    N = name.upper()
    strides = [-(-((d["signs"].shape[1] + 31) // 32) // 4) * 4 for d in layers]     # words per row, multiple of 4
    max_n = max(d["signs"].shape[0] for d in layers)
    max_z = max(strides + [-(-((max_n + 31) // 32) // 4) * 4])
    n0 = layers[0]["signs"].shape[1]
    steps = []
    for i, d in enumerate(layers):
        out, n_in = d["signs"].shape
        kind = "uint8 x 1 bit" if d["u8_input"] else "1 bit x 1 bit"
        steps.append(f"layer {i}: {n_in} -> {out} ({kind}, AND) + {'argmax' if d['last'] else 'thresholds'}")
    L = ["/* Generated by the SYMBEX-1 CDT (export/dense.py). Do not edit by hand.", " * AND arithmetic (ops/symbex_and): "
         "the same raw scores as the XNOR path, with the counts done by the HAL (SIMD multiplier where available). */",
         f"#ifndef {N}_H", f"#define {N}_H", "", "#include <stdint.h>", '#include "layers/dense/symbex_dense.h"',
         '#include "ops/symbex_and.h"', "",
         f"#define {N}_ARITHMETIC_AND 1",
         f"#define {N}_Q_ACT {Q_ACT}", f"#define {N}_Q_MULT {Q_MULT}",
         f"#define {N}_N_INPUTS {n0}", f"#define {N}_N_CLASSES {layers[-1]['signs'].shape[0]}", "",
         "/* Where the weights live: const = flash (default). Compile with -DSYMBEX_ROM= (empty) to put them",
         " * in RAM (faster on chips whose flash cache is smaller than the model, costs that much RAM). */",
         "#ifndef SYMBEX_ROM", "#define SYMBEX_ROM const", "#endif",
         "/* Per-layer profiling: define SYMBEX_MARK(k) before including this header to time each step",
         " * (k = 0 before the first layer, k = i + 1 after layer i). Empty by default: no cost. */",
         "#ifndef SYMBEX_MARK", "#define SYMBEX_MARK(k)", "#endif",
         f"#define {N}_N_STEPS {len(layers)}",
         f"#define {N}_BYTES_BUFFERS {4 * max_n + 2 * 4 * max_z + 32 * strides[0]}   /* static RAM of {name}_infer */",
         "#ifdef SYMBEX_PROFILE",
         f"static const char* const {name}_steps[{len(layers)}] = {{" + ", ".join(f'"{p}"' for p in steps) + "};",
         "#endif", ""]
    for i, d in enumerate(layers):
        out, n_in = d["signs"].shape
        packed, pal = _pack(d["signs"])
        z = strides[i]
        rows = np.zeros((out, z), dtype=np.uint64)
        rows[:, :pal] = np.asarray(packed, dtype=np.uint64)
        L.append(f"/* layer {i}: {n_in} -> {out}{' (uint8 input)' if d['u8_input'] else ''}, {z} words per row */")
        L.append(f"static SYMBEX_ROM uint32_t {name}_w{i}[{out * z}] __attribute__((aligned(16))) = {{")
        for row in rows:
            L.append("    " + ", ".join(f"0x{int(v):08X}u" for v in row) + ",")
        L.append("};")
        if not d["u8_input"]:
            cw = (d["signs"] > 0).sum(1)
            L.append(f"static SYMBEX_ROM int32_t {name}_cw{i}[{out}] = {{" + ", ".join(str(int(v)) for v in cw) + "};")
        if not d["last"]:
            L.append(f"static SYMBEX_ROM int32_t {name}_t{i}[{out}] = {{"
                     + ", ".join("INT32_MIN" if t == INT32_MIN else str(int(t)) for t in d["threshold"]) + "};")
        else:
            L.append(f"static SYMBEX_ROM int64_t {name}_scale[{out}] = {{" + ", ".join(f"{int(v)}LL" for v in d["scale"]) + "};")
            L.append(f"static SYMBEX_ROM int32_t {name}_shift[{out}] = {{" + ", ".join(str(int(v)) for v in d["shift"]) + "};")
        L.append("")
    L += [f"/* Full inference: x = {n0} uint8 inputs (same order as the training tensor, flattened).",
          " * Returns the predicted class. Not reentrant (static buffers). */",
          f"static inline int {name}_infer(const uint8_t* x) {{",
          f"    static int32_t raw[{max_n}];",
          f"    static uint32_t bits_a[{max_z}] __attribute__((aligned(16))), bits_b[{max_z}] __attribute__((aligned(16)));",
          f"    static uint8_t work[{32 * strides[0]}] __attribute__((aligned(16)));   /* prepared input of layer 0 */",
          "    uint32_t* in_bits = bits_a;", "    uint32_t* out_bits = bits_b;", "    SYMBEX_MARK(0);"]
    for i, d in enumerate(layers):
        out, n_in = d["signs"].shape
        if d["u8_input"]:
            L.append(f"    symbex_and_input_u8(x, {n_in}, {name}_w{i}, {strides[i]}, {out}, work, raw);")
        else:
            L.append(f"    symbex_and_raw(in_bits, {n_in}, {name}_w{i}, {strides[i]}, {out}, {name}_cw{i}, raw);")
        if not d["last"]:
            L += [f"    symbex_dense_threshold(raw, {name}_t{i}, out_bits, {out});",
                  "    { uint32_t* t = in_bits; in_bits = out_bits; out_bits = t; }", f"    SYMBEX_MARK({i + 1});"]
    sh = Q_MULT - Q_ACT
    L += ["    int best = 0;", "    int64_t max_logit = INT64_MIN;",
          f"    for (int j = 0; j < {layers[-1]['signs'].shape[0]}; j++) {{",
          f"        int64_t logit = (((int64_t)raw[j] * {name}_scale[j] + ((int64_t)1 << {sh - 1})) >> {sh}) + {name}_shift[j];",
          "        if (logit > max_logit) { max_logit = logit; best = j; }",
          "    }", f"    SYMBEX_MARK({len(layers)});", "    (void)in_bits;", "    (void)out_bits;", "    return best;", "}", "",
          f"#endif /* {N}_H */", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------------------------------
# 5. exactness of the compiled C on this PC
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


def validate_c(header_path, lib_path, x_u8, expected):
    """Compiles the header + lib_symbex with gcc and runs it on x_u8. Returns (equal, total, message)."""
    if shutil.which("gcc") is None:
        return None, 0, "gcc not found: the C exactness check was skipped"
    inc = os.path.join(lib_path, "include")
    sources = [os.path.join(lib_path, "src", "layers", "dense", "symbex_dense.c"),
               os.path.join(lib_path, "src", "ops", "symbex_and.c"),
               os.path.join(lib_path, "src", "hal", "generic", "symbex_hal_impl.c")]
    for r in [inc] + sources:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing): set [output] lib_path"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header_path, os.path.join(d, "model_symbex.h"))
        with open(os.path.join(d, "main.c"), "w") as f:
            f.write(_TEST_MAIN)
        exe = os.path.join(d, "prueba")
        r = subprocess.run(["gcc", "-O2", "-std=c99", "-I", inc, "-I", d, "-o", exe, os.path.join(d, "main.c")]
                           + sources, capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C did not compile:\n" + r.stderr[-2000:]
        data = os.path.join(d, "x.u8")
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(data)
        r = subprocess.run([exe, data, str(len(x_u8))], capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C program failed while running"
        pred_c = np.array([int(v) for v in r.stdout.split()])
    equal = int((pred_c == expected).sum())
    return equal, len(expected), ""


# ---------------------------------------------------------------------------------------------------
# 6. everything together
# ---------------------------------------------------------------------------------------------------
def to_uint8(x, mean=(), std=()):
    """Normalized tensors (N, C, ...) -> the original uint8 values. Raises if they are not 8-bit data."""
    x = x.detach().cpu().double()
    if mean:
        shape = (1, -1) + (1,) * (x.dim() - 2)
        x = x * torch.tensor(std, dtype=torch.float64).view(shape) + torch.tensor(mean, dtype=torch.float64).view(shape)
    p = x * 255.0
    q = torch.round(p)
    if (q < 0).any() or (q > 255).any() or (p - q).abs().max() > 1e-3:
        raise ValueError("dense export: the inputs are not 8-bit data (x * 255 is not an integer in 0-255 after "
                         "undoing [data] mean / std); the uint8 first layer needs 8-bit inputs")
    return q.to(torch.uint8).reshape(len(q), -1).numpy()


def _save_labels(out_dir, name, values):
    """validation_<name>.u8, or validation_<name>.u32 (int32) when a value does not fit in a byte (the 31-bit grid
    checksums of a per-cell head, FOMO); the file of the other width is removed, so load_validation never reads a
    stale one"""
    v = np.asarray(values, dtype=np.int64)
    if v.size and (v.min() < 0 or v.max() > np.iinfo(np.int32).max):
        raise ValueError(f"save_validation: validation_{name} values must be in 0 .. 2^31 - 1")
    ext, other = (".u32", ".u8") if v.size and v.max() > 255 else (".u8", ".u32")
    v.astype(np.int32 if ext == ".u32" else np.uint8).tofile(os.path.join(out_dir, f"validation_{name}{ext}"))
    stale = os.path.join(out_dir, f"validation_{name}{other}")
    if os.path.exists(stale):
        os.remove(stale)


def _load_labels(out_dir, name):
    path = os.path.join(out_dir, f"validation_{name}.u32")
    if os.path.exists(path):
        return np.fromfile(path, dtype=np.int32).astype(np.int64)
    return np.fromfile(os.path.join(out_dir, f"validation_{name}.u8"), dtype=np.uint8).astype(np.int64)


def save_validation(out_dir, x_u8, y, pred_sim):
    """validation_x.u8 (N x inputs), validation_y.u8 (labels), validation_sim.u8 (classes of the integer
    simulation): what the chip test needs, so it can run later without retraining (probar_chip.py).
    y and pred_sim go to .u32 (int32) instead when a value passes 255 (the grid checksums of a per-cell head)."""
    np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(os.path.join(out_dir, "validation_x.u8"))
    _save_labels(out_dir, "y", y)
    _save_labels(out_dir, "sim", pred_sim)


def load_validation(out_dir, n_inputs):
    """the files of save_validation: validation_y / validation_sim from their .u32 if it exists, else the .u8"""
    x = np.fromfile(os.path.join(out_dir, "validation_x.u8"), dtype=np.uint8).reshape(-1, n_inputs)
    return x, _load_labels(out_dir, "y"), _load_labels(out_dir, "sim")


def flash_bytes(layers, arithmetic="and"):
    """Bytes of weights + per-neuron parameters in the header. AND rows are padded to a multiple of 4 words and
    carry the count of +1 weights per neuron; XNOR rows use ceil(n / 32) words."""
    total = 0
    for d in layers:
        out, n_in = d["signs"].shape
        words = (n_in + 31) // 32
        if arithmetic == "and":
            words = -(-words // 4) * 4
            total += 0 if d["u8_input"] else out * 4                 # count of +1 weights
        total += out * words * 4 + out * (12 if d["last"] else 4)
    return total


def export_and_validate(model, val_dataset, out_dir, lib_path, mean=(), std=(), n_validation=2000, verbose=True,
                       arithmetic="and", aritmetica=None, input_bits=8):
    """Exports `model` (all layers 'bin', plain) and validates it. Returns a dict with the numbers.
    aritmetica: old name of arithmetic."""
    arithmetic = arithmetic if aritmetica is None else aritmetica
    n = min(n_validation, len(val_dataset))
    xs = torch.stack([val_dataset[i][0] for i in range(n)])
    ys = np.array([int(val_dataset[i][1]) for i in range(n)])
    model = model.cpu().eval()
    groups = analyze(model, xs[:2])
    layers = fold(groups, mean, std)
    header_path = os.path.join(out_dir, "model_symbex.h")
    write_header(layers, header_path, arithmetic=arithmetic, input_bits=input_bits)
    x_u8 = to_uint8(xs, mean, std)
    # PyTorch: the +-1 input each binarized layer receives (output of its RSign), to compare layer by layer
    captured, hooks = {}, []
    for i, (layer, _, _) in enumerate(groups[1:]):
        hooks.append(layer.rsign.register_forward_hook(lambda m, e, s_, i=i: captured.__setitem__(i, s_.detach())))
    with torch.no_grad():
        pred_pt = model(xs).argmax(1).numpy()
    for h in hooks:
        h.remove()
    pred_sim, bits_sim = simulate(layers, x_u8, return_bits=True)
    bit_agreement = [100 * float((captured[i].numpy() == bits_sim[i]).mean()) for i in range(len(bits_sim))]
    equal_c, total_c, msg = validate_c(header_path, lib_path, x_u8, pred_sim)
    save_validation(out_dir, x_u8, ys, pred_sim)
    flash = flash_bytes(layers, arithmetic)
    res = dict(n=n, arithmetic=arithmetic, acc_pytorch=100 * float((pred_pt == ys).mean()), acc_integer=100 * float((pred_sim == ys).mean()),
               fidelity=100 * float((pred_pt == pred_sim).mean()), bit_agreement=bit_agreement,
               flash_bytes=flash, header=header_path,
               c_exactness=None if equal_c is None else 100 * equal_c / total_c, c_note=msg,
               _x_u8=x_u8, _y=ys, _pred_sim=pred_sim, _xs=xs, _layers=layers)   # "_" keys: not saved to JSON
    if verbose:
        print(f"    layers      : " + " -> ".join(str(d["signs"].shape[1]) for d in layers)
              + f" -> {layers[-1]['signs'].shape[0]}   (first layer: uint8 input)")
        print(f"    header      : {header_path}  ({flash / 1024:.1f} KB of weights + thresholds in flash, "
              f"{arithmetic.upper()} arithmetic)")
        print(f"    validation  : {n} samples")
        print(f"    accuracy    : PyTorch {res['acc_pytorch']:.2f}%   integer simulation {res['acc_integer']:.2f}%")
        print(f"    FIDELITY    : {res['fidelity']:.2f}%  (same prediction PyTorch vs. integers: what folding costs)")
        for i, a in enumerate(bit_agreement):
            print(f"      hidden layer {i}: {a:.3f}% of the bits equal PyTorch vs. integers"
                  + ("" if a >= 99.5 else "   [!] low: a bug in the folding, not rounding"))
        if res["c_exactness"] is None:
            print(f"    EXACTNESS   : not checked -- {msg}")
        else:
            status = "OK" if res["c_exactness"] == 100 else "[X] MISMATCH: bug in the export or the C engine"
            print(f"    EXACTNESS   : {res['c_exactness']:.2f}%  (compiled C vs. integer simulation, must be 100%) {status}")
    return res
