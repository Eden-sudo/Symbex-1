"""
ternary.py -- export + validation of a ternary MLP (method "ternary", every layer "bin") to lib_symbex.

Same structure and folding as dense.py (the plain 1-bit MLP), with ternary weights:
    plegar()            the trained model -> integers: {-1, 0, +1} codes, one int32 threshold per hidden neuron,
                        int64 scale + int32 shift per output class
    escribir_header()   modelo_symbex.h: sign plane + nonzero-mask plane per layer, and symbex_mlp_inferir()
                        using symbex_ternary.c (same entry point and macros as the plain header)
    validar_c()         compiles the header with lib_symbex on this PC and checks that the C gives EXACTLY the
                        predictions of the integer simulation (must be 100%)
    exportar_y_validar()  all of it, plus the bit-level check of validate/ternary.py (packed-word simulator vs.
                        numpy integer reference, layer by layer)

Supported structure: LinearTernary -> [BatchNorm1d] -> [ReLU] -> ... -> LinearTernary -> [BatchNorm1d].

Folding, per neuron j (raw_j = sum_i t_ij * x_i, an integer; x_i = +-1, or the uint8 input in the first layer):
    z_j = A_j * raw_j + B_j       A_j = alpha_j (mean |W| over the nonzero weights) x BatchNorm (x input scale)
    bit_j = [ReLU(z_j) >= beta_j] = [raw_j >= T_j], codes of row j negated when A_j < 0    (dense._umbral)
Weight bytes: 2 bits per weight (two planes), each row padded to whole 32-bit words.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from .dense import Q_ACT, Q_MULT, INT32_MIN, _empaquetar, _umbral, a_uint8, analizar, guardar_validacion
from ..methods.linear.ternary import LinearTernary, ternary_codes, ternary_delta


# ---------------------------------------------------------------------------------------------------
# 1. folding to integers
# ---------------------------------------------------------------------------------------------------
def _codigos_y_afin(capa, bn):
    """({-1, 0, +1} codes as int64, A, B before the input fold). The codes come from the SAME function the
    forward pass uses (float32 comparison), so no weight on the threshold flips between PyTorch and C."""
    with torch.no_grad():
        t = ternary_codes(capa.weight.detach().cpu(), capa.delta_ratio.detach().cpu())
    codigos = t.numpy().astype(np.int64)
    w = capa.weight.detach().double().cpu().numpy()
    mascara = codigos != 0
    cuenta = mascara.sum(1)
    a = np.where(cuenta > 0, (np.abs(w) * mascara).sum(1) / np.maximum(cuenta, 1), 0.0)
    b = capa.bias.detach().double().cpu().numpy() if capa.bias is not None else np.zeros(w.shape[0])
    if bn is not None:
        g = (bn.weight.detach().double().cpu() / torch.sqrt(bn.running_var.detach().double().cpu() + bn.eps)).numpy()
        b = g * (b - bn.running_mean.detach().double().cpu().numpy()) + bn.bias.detach().double().cpu().numpy()
        a = g * a
    return codigos, a, b


def plegar(grupos, mean=(), std=()):
    """Folds the trained ternary layers into integers. Returns a list of dicts, one per layer:
    codigos (out, in) int64 in {-1, 0, +1}, entrada_u8, ultima, and umbral (hidden) or escala + shift (output)."""
    if std and len(set(std)) != 1:
        raise ValueError("ternary export: the input normalization must use ONE std for every channel")
    s = std[0] if std else 1.0
    capas = []
    for i, (capa, bn, relu) in enumerate(grupos):
        codigos, a, b = _codigos_y_afin(capa, bn)
        if i == 0:
            # x_i = (p_i / 255 - m_c) / s  ->  sum t_ij x_i = raw_j / (255 s) - sum_i m_c(i) t_ij / s
            n_in = codigos.shape[1]
            if mean:
                if n_in % len(mean):
                    raise ValueError(f"ternary export: {n_in} inputs cannot be split into {len(mean)} channels")
                m_por_entrada = np.repeat(np.asarray(mean, dtype=np.float64), n_in // len(mean))
            else:
                m_por_entrada = np.zeros(n_in)
            D = (codigos * m_por_entrada[None, :]).sum(1) / s
            A, B = a / (255.0 * s), b - a * D
        else:
            A, B = a, b
        d = dict(codigos=codigos, A=A, B=B, entrada_u8=(i == 0), ultima=(i == len(grupos) - 1),
                 delta=float(ternary_delta(capa.weight.detach().cpu(), capa.delta_ratio.detach().cpu())),
                 ceros=float((codigos == 0).mean()))
        if not d["ultima"]:
            betas = grupos[i + 1][0].rsign.beta.detach().double().cpu().numpy().ravel()
            T = np.zeros(len(A), dtype=np.int64)
            for j in range(len(A)):
                neg, T[j] = _umbral(A[j], B[j], betas[j], relu)
                if neg:
                    d["codigos"][j] = -d["codigos"][j]
            d["umbral"] = T
        else:
            d["escala"] = np.round(A * 2 ** Q_MULT).astype(np.int64)
            d["shift"] = np.round(B * 2 ** Q_ACT).astype(np.int64)
            if np.abs(d["shift"]).max() >= 2 ** 31:
                raise ValueError("ternary export: an output bias does not fit in int32 with Q_ACT = 16")
        capas.append(d)
    return capas


def planos(codigos):
    """(out, in) {-1, 0, +1} -> (sign words, mask words, words per row). Same packing as dense._empaquetar:
    ceil(in / 32) words per row, bit i of word i / 32 = input i. Sign bit = 1 for +1; mask bit = 1 for nonzero.
    Zero weights get sign bit 0 (never read: the mask hides them)."""
    signos, pal = _empaquetar(np.where(codigos > 0, 1, -1))
    mascara, _ = _empaquetar(np.where(codigos != 0, 1, -1))
    return signos, mascara, pal


def bytes_pesos(capas):
    """Bytes of the two weight planes (what the flash holds for the weights alone)."""
    return sum(2 * planos(d["codigos"])[0].size * 4 for d in capas)


def bytes_flash(capas):
    """Weights (2 planes) + int32 thresholds per hidden neuron + int64 scale and int32 shift per class."""
    return bytes_pesos(capas) + sum(d["codigos"].shape[0] * (12 if d["ultima"] else 4) for d in capas)


# ---------------------------------------------------------------------------------------------------
# 2. C header (same entry point and macros as dense.escribir_header, so the same test harness runs it)
# ---------------------------------------------------------------------------------------------------
def _palabras(lineas, tipo, nombre, matriz):
    lineas.append(f"static SYMBEX_ROM {tipo} {nombre}[{matriz.size}] = {{")
    for fila in matriz:
        lineas.append("    " + ", ".join(f"0x{int(v):08X}u" for v in fila) + ",")
    lineas.append("};")


def escribir_header(capas, ruta, nombre="symbex_mlp"):
    N = nombre.upper()
    n_in0 = capas[0]["codigos"].shape[1]
    max_n = max(d["codigos"].shape[0] for d in capas)
    pasos = [f"layer {i}: {d['codigos'].shape[1]} -> {d['codigos'].shape[0]} "
             f"({'uint8' if d['entrada_u8'] else '1 bit'} x ternary) + {'argmax' if d['ultima'] else 'thresholds'}"
             for i, d in enumerate(capas)]
    lineas = ["/* Generated by the SYMBEX-1 CDT (export/ternary.py, method \"ternary\"). Do not edit by hand. */",
              f"#ifndef {N}_H", f"#define {N}_H", "",
              "#include <stdint.h>", '#include "layers/vector/symbex_dense.h"',
              '#include "layers/vector/symbex_ternary.h"', "",
              f"#define {N}_TERNARIO 1", f"#define {N}_Q_ACT {Q_ACT}", f"#define {N}_Q_MULT {Q_MULT}",
              f"#define {N}_N_ENTRADAS {n_in0}", f"#define {N}_N_CLASES {capas[-1]['codigos'].shape[0]}", "",
              "/* Where the weights live: const = flash (default). -DSYMBEX_ROM= (empty) puts them in RAM. */",
              "#ifndef SYMBEX_ROM", "#define SYMBEX_ROM const", "#endif",
              "/* Per-layer profiling: define SYMBEX_MARCA(k) before including this header (k = 0 before the first",
              " * layer, k = i + 1 after layer i). Empty by default: no cost. */",
              "#ifndef SYMBEX_MARCA", "#define SYMBEX_MARCA(k)", "#endif",
              f"#define {N}_N_PASOS {len(capas)}",
              f"#define {N}_BYTES_BUFFERS {4 * max_n + 2 * 4 * ((max_n + 31) // 32)}   /* static RAM of {nombre}_inferir */",
              "#ifdef SYMBEX_PERFIL",
              f"static const char* const {nombre}_pasos[{len(capas)}] = {{" + ", ".join(f'"{p}"' for p in pasos) + "};",
              "#endif", ""]
    for i, d in enumerate(capas):
        out, n_in = d["codigos"].shape
        s, m, _ = planos(d["codigos"])
        lineas.append(f"/* layer {i}: {n_in} -> {out}, ternary ({100 * d['ceros']:.1f}% zeros)"
                      f"{' (uint8 input)' if d['entrada_u8'] else ''} */")
        _palabras(lineas, "uint32_t", f"{nombre}_s{i}", s)
        _palabras(lineas, "uint32_t", f"{nombre}_m{i}", m)
        lineas.append(f"static const struct SymbexTernaryLayerData {nombre}_capa{i} = "
                      f"{{ {n_in}, {out}, {nombre}_s{i}, {nombre}_m{i} }};")
        if not d["ultima"]:
            lineas.append(f"static SYMBEX_ROM int32_t {nombre}_t{i}[{out}] = {{"
                          + ", ".join("INT32_MIN" if t == INT32_MIN else str(int(t)) for t in d["umbral"]) + "};")
        else:
            lineas.append(f"static SYMBEX_ROM int64_t {nombre}_escala[{out}] = {{"
                          + ", ".join(f"{int(v)}LL" for v in d["escala"]) + "};")
            lineas.append(f"static SYMBEX_ROM int32_t {nombre}_shift[{out}] = {{"
                          + ", ".join(str(int(v)) for v in d["shift"]) + "};")
        lineas.append("")
    lineas += [f"/* Full inference: x = {n_in0} uint8 inputs (training tensor order, flattened). Returns the class.",
               " * Not reentrant (static buffers). */",
               f"static inline int {nombre}_inferir(const uint8_t* x) {{",
               f"    static int32_t raw[{max_n}];",
               f"    static uint32_t bits_a[{(max_n + 31) // 32}], bits_b[{(max_n + 31) // 32}];",
               "    uint32_t* entrada = bits_a;", "    uint32_t* salida = bits_b;", "    SYMBEX_MARCA(0);"]
    for i, d in enumerate(capas):
        if d["entrada_u8"]:
            lineas.append(f"    symbex_ternary_entrada_u8_mascara(x, raw, &{nombre}_capa{i});")
        else:
            lineas.append(f"    symbex_ternary_forward(entrada, raw, &{nombre}_capa{i});")
        if not d["ultima"]:
            lineas.append(f"    symbex_dense_umbral(raw, {nombre}_t{i}, salida, {d['codigos'].shape[0]});")
            lineas.append("    { uint32_t* t = entrada; entrada = salida; salida = t; }")
            lineas.append(f"    SYMBEX_MARCA({i + 1});")
    sh = Q_MULT - Q_ACT
    lineas += ["    int mejor = 0;", "    int64_t max_logit = INT64_MIN;",
               f"    for (int j = 0; j < {capas[-1]['codigos'].shape[0]}; j++) {{",
               f"        int64_t logit = (((int64_t)raw[j] * {nombre}_escala[j] + ((int64_t)1 << {sh - 1})) >> {sh})"
               f" + {nombre}_shift[j];",
               "        if (logit > max_logit) { max_logit = logit; mejor = j; }",
               "    }", f"    SYMBEX_MARCA({len(capas)});", "    (void)entrada;", "    (void)salida;",
               "    return mejor;", "}", "", f"#endif /* {N}_H */", ""]
    with open(ruta, "w") as f:
        f.write("\n".join(lineas))


# ---------------------------------------------------------------------------------------------------
# 3. exactness of the compiled C on this PC
# ---------------------------------------------------------------------------------------------------
_MAIN_PRUEBA = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "modelo_symbex.h"
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    int n = atoi(argv[2]);
    static uint8_t x[SYMBEX_MLP_N_ENTRADAS];
    for (int i = 0; i < n; i++) {
        if (fread(x, 1, SYMBEX_MLP_N_ENTRADAS, f) != SYMBEX_MLP_N_ENTRADAS) return 1;
        printf("%d\n", symbex_mlp_inferir(x));
    }
    fclose(f);
    return 0;
}
"""


def validar_c(ruta_header, lib_path, x_u8, esperado, flags=()):
    """Compiles the header + lib_symbex (ternary + dense thresholds + generic HAL) with gcc, runs it on x_u8.
    Returns (equal, total, message). `flags`: extra compiler flags (e.g. -DSYMBEX_POPCOUNT_FORZAR_SWAR)."""
    if shutil.which("gcc") is None:
        return None, 0, "gcc not found: the C exactness check was skipped"
    inc = os.path.join(lib_path, "include")
    fuentes = [os.path.join(lib_path, "src", "layers", "vector", "symbex_ternary.c"),
               os.path.join(lib_path, "src", "layers", "vector", "symbex_dense.c"),
               os.path.join(lib_path, "src", "hal", "generico", "symbex_hal_impl.c")]
    for r in [inc] + fuentes:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing): set [output] lib_path"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(ruta_header, os.path.join(d, "modelo_symbex.h"))
        with open(os.path.join(d, "main.c"), "w") as f:
            f.write(_MAIN_PRUEBA)
        exe = os.path.join(d, "prueba")
        r = subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", *flags, "-I", inc, "-I", d, "-o", exe,
                            os.path.join(d, "main.c")] + fuentes, capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C did not compile:\n" + r.stderr[-2000:]
        datos = os.path.join(d, "x.u8")
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(datos)
        r = subprocess.run([exe, datos, str(len(x_u8))], capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C program failed while running"
        pred_c = np.array([int(v) for v in r.stdout.split()])
    return int((pred_c == esperado).sum()), len(esperado), ""


# ---------------------------------------------------------------------------------------------------
# 4. everything together
# ---------------------------------------------------------------------------------------------------
def exportar_y_validar(modelo, val_dataset, salida, lib_path, mean=(), std=(), n_validacion=2000, verbose=True):
    """Exports `modelo` (all layers 'bin', method ternary) and validates it. Returns a dict with the numbers
    (same keys as dense.exportar_y_validar, plus bit_exacto, ceros and pesos_bytes)."""
    from ..validate.ternary import verificar_bit_exacto
    n = min(n_validacion, len(val_dataset))
    xs = torch.stack([val_dataset[i][0] for i in range(n)])
    ys = np.array([int(val_dataset[i][1]) for i in range(n)])
    modelo = modelo.cpu().eval()
    grupos = analizar(modelo, xs[:2], clase=LinearTernary)
    capas = plegar(grupos, mean, std)
    ruta_h = os.path.join(salida, "modelo_symbex.h")
    escribir_header(capas, ruta_h)
    x_u8 = a_uint8(xs, mean, std)
    capturas, hooks = {}, []
    for i, (capa, _, _) in enumerate(grupos[1:]):
        hooks.append(capa.rsign.register_forward_hook(lambda m, e, s_, i=i: capturas.__setitem__(i, s_.detach())))
    with torch.no_grad():
        pred_pt = modelo(xs).argmax(1).numpy()
    for h in hooks:
        h.remove()
    exacto = verificar_bit_exacto(capas, x_u8)
    pred_sim, bits_sim = exacto["_pred"], exacto["_bits"]
    acuerdo_bits = [100 * float((capturas[i].numpy() == bits_sim[i]).mean()) for i in range(len(bits_sim))]
    iguales_c, total_c, msg = validar_c(ruta_h, lib_path, x_u8, pred_sim)
    guardar_validacion(salida, x_u8, ys, pred_sim)
    flash, pesos = bytes_flash(capas), bytes_pesos(capas)
    res = dict(n=n, metodo="ternary", acc_pytorch=100 * float((pred_pt == ys).mean()),
               acc_entero=100 * float((pred_sim == ys).mean()),
               fidelidad=100 * float((pred_pt == pred_sim).mean()), acuerdo_bits=acuerdo_bits,
               bit_exacto={k: v for k, v in exacto.items() if not k.startswith("_")},
               ceros=[d["ceros"] for d in capas], delta=[d["delta"] for d in capas],
               pesos_bytes=pesos, flash_bytes=flash, header=ruta_h,
               exactitud_c=None if iguales_c is None else 100 * iguales_c / total_c, nota_c=msg,
               _x_u8=x_u8, _y=ys, _pred_sim=pred_sim, _xs=xs)
    if verbose:
        print("    layers      : " + " -> ".join(str(d["codigos"].shape[1]) for d in capas)
              + f" -> {capas[-1]['codigos'].shape[0]}   (ternary, first layer: uint8 input)")
        print("    zeros       : " + ", ".join(f"layer {i} {100 * d['ceros']:.1f}%" for i, d in enumerate(capas)))
        print(f"    header      : {ruta_h}  ({pesos / 1024:.1f} KB of weights (2 bit planes), "
              f"{flash / 1024:.1f} KB with thresholds / scales)")
        print(f"    validation  : {n} samples")
        print(f"    accuracy    : PyTorch {res['acc_pytorch']:.2f}%   integer simulation {res['acc_entero']:.2f}%")
        print(f"    FIDELITY    : {res['fidelidad']:.2f}%  (same prediction PyTorch vs. integers: what folding costs)")
        for i, a in enumerate(acuerdo_bits):
            print(f"      hidden layer {i}: {a:.3f}% of the bits equal PyTorch vs. integers"
                  + ("" if a >= 99.5 else "   [!] low: a bug in the folding, not rounding"))
        be = res["bit_exacto"]
        print(f"    BIT-EXACT   : packed-word simulator vs. numpy integer reference: "
              f"{'OK' if be['iguales'] else '[X] MISMATCH'} ({be['capas']} layers, raw scores + logits, {n} samples)")
        if res["exactitud_c"] is None:
            print(f"    EXACTNESS   : not checked -- {msg}")
        else:
            estado = "OK" if res["exactitud_c"] == 100 else "[X] MISMATCH: bug in the export or the C engine"
            print(f"    EXACTNESS   : {res['exactitud_c']:.2f}%  (compiled C vs. integer simulation, must be 100%) {estado}")
    return res
