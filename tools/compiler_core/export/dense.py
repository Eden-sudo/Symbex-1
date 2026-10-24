"""
dense.py -- export + validation of a 1-bit MLP (method plain, every layer "bin") to lib_symbex.

Export and validation live together because they must use the SAME folded integers:
    plegar()            the trained model -> integers: 1-bit weights, one int32 threshold per hidden neuron,
                        int64 scale + int32 shift per output class
    simular()           exact integer simulation of the C engine in numpy (the reference for the C)
    escribir_header()   modelo_symbex.h: arrays + symbex_mlp_inferir(const uint8_t* x) using lib_symbex
    validar_c()         compiles the header with lib_symbex on this PC (gcc) and checks that the C gives
                        EXACTLY the predictions of simular() (must be 100%)
    exportar_y_validar()  all of it, with a report
    medir_en_chip()     builds an ESP-IDF project (PlatformIO) with the header + lib_symbex, flashes the board,
                        streams validation images over serial and reports: predictions equal to the integer
                        simulation (must be 100%), accuracy, and latency measured ON the chip

Supported structure (in forward order; Flatten / Dropout / Identity are skipped):
    LinearPlain -> [BatchNorm1d] -> [ReLU] -> LinearPlain -> ... -> LinearPlain -> [BatchNorm1d]
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
def _hojas_en_orden(modelo, x, clase=None):
    """Modules in the order the forward pass runs them: every LinearPlain (or `clase`) as ONE unit (its inner
    RSign is part of it) and every other leaf module that is not inside one."""
    from ..methods.linear.plain import LinearPlain
    clase = clase or LinearPlain
    unidades = [n for n, m in modelo.named_modules() if isinstance(m, clase)]
    orden, hooks = [], []
    for n, m in modelo.named_modules():
        dentro = any(n.startswith(u + ".") for u in unidades)
        if isinstance(m, clase) or (len(list(m.children())) == 0 and not dentro):
            hooks.append(m.register_forward_hook(lambda mod, e, s: orden.append(mod)))
    estaba = modelo.training
    modelo.eval()
    with torch.no_grad():
        modelo(x)
    modelo.train(estaba)
    for h in hooks:
        h.remove()
    vistos, res = set(), []
    for m in orden:
        if id(m) not in vistos:
            vistos.add(id(m))
            res.append(m)
    return res


def analizar(modelo, x, clase=None):
    """Returns the layer groups [(LinearPlain, BatchNorm1d or None, has ReLU)], or raises ValueError
    explaining what is not supported. `clase`: the layer class to look for (LinearPlain by default)."""
    from ..methods.linear.plain import LinearPlain
    clase = clase or LinearPlain
    grupos, actual = [], None
    for m in _hojas_en_orden(modelo, x, clase):
        nombre = type(m).__name__
        if isinstance(m, (nn.Flatten, nn.Dropout, nn.Identity)):
            continue
        if isinstance(m, clase):
            actual = [m, None, False]
            grupos.append(actual)
        elif isinstance(m, nn.BatchNorm1d) and actual is not None and actual[1] is None and not actual[2]:
            actual[1] = m
        elif isinstance(m, nn.ReLU) and actual is not None and not actual[2]:
            actual[2] = True
        else:
            raise ValueError(f"dense export: unsupported module {nombre} (supported: {clase.__name__} -> "
                             f"[BatchNorm1d] -> [ReLU], repeated). k-bit layers cannot be exported yet.")
    if not grupos:
        raise ValueError(f"dense export: no {clase.__name__} layer found (is every layer 'bin'?)")
    if grupos[-1][2]:
        raise ValueError("dense export: the last layer cannot have a ReLU (it produces the logits)")
    return [tuple(g) for g in grupos]


# ---------------------------------------------------------------------------------------------------
# 2. folding to integers
# ---------------------------------------------------------------------------------------------------
def _afin(capa, bn):
    """(alpha, bias) of the layer combined with its BatchNorm: y = a * (sum of +-x * s) + b, as numpy float64."""
    w = capa.weight.detach().double().cpu()
    a = w.abs().mean(dim=1)
    b = capa.bias.detach().double().cpu() if capa.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    if bn is not None:
        g = bn.weight.detach().double().cpu() / torch.sqrt(bn.running_var.detach().double().cpu() + bn.eps)
        b = g * (b - bn.running_mean.detach().double().cpu()) + bn.bias.detach().double().cpu()
        a = g * a
    return a.numpy(), b.numpy()


def _umbral(A, B, beta, relu):
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


def plegar(grupos, mean=(), std=()):
    """Folds the trained layers into integers. Returns a list of dicts, one per layer."""
    if std and len(set(std)) != 1:
        raise ValueError("dense export: the input normalization must use ONE std for every channel")
    s = std[0] if std else 1.0
    capas = []
    for i, (capa, bn, relu) in enumerate(grupos):
        w = capa.weight.detach().cpu().numpy()
        signos = np.where(w >= 0, 1, -1).astype(np.int64)          # 0 -> +1 (same as the C engine)
        a, b = _afin(capa, bn)
        if i == 0:
            # input x_i = (p_i / 255 - m_c) / s  ->  sum x_i s_ij = raw_j / (255 s) - sum_i m_c(i) s_ij / s
            # (the flattened input is channel-major: feature i belongs to channel i // (n_in / C))
            n_in = w.shape[1]
            if mean:
                if n_in % len(mean):
                    raise ValueError(f"dense export: {n_in} inputs cannot be split into {len(mean)} channels")
                m_por_entrada = np.repeat(np.asarray(mean, dtype=np.float64), n_in // len(mean))
            else:
                m_por_entrada = np.zeros(n_in)
            D = (signos * m_por_entrada[None, :]).sum(1) / s
            A, B = a / (255.0 * s), b - a * D
        else:
            A, B = a, b
        d = dict(signos=signos, A=A, B=B, entrada_u8=(i == 0), ultima=(i == len(grupos) - 1))
        if not d["ultima"]:
            siguiente = grupos[i + 1][0]
            betas = siguiente.rsign.beta.detach().double().cpu().numpy().ravel()
            T = np.zeros(len(A), dtype=np.int64)
            for j in range(len(A)):
                neg, T[j] = _umbral(A[j], B[j], betas[j], relu)
                if neg:
                    d["signos"][j] = -d["signos"][j]
            d["umbral"] = T
        else:
            d["escala"] = np.round(A * 2 ** Q_MULT).astype(np.int64)
            d["shift"] = np.round(B * 2 ** Q_ACT).astype(np.int64)
            if np.abs(d["shift"]).max() >= 2 ** 31:
                raise ValueError("dense export: an output bias does not fit in int32 with Q_ACT = 16")
        capas.append(d)
    return capas


# ---------------------------------------------------------------------------------------------------
# 3. exact integer simulation (what the C must reproduce bit for bit)
# ---------------------------------------------------------------------------------------------------
def simular(capas, x_u8, devolver_bits=False):
    """x_u8: (N, n_inputs) uint8. Returns the predicted classes (N,), and with devolver_bits=True also the
    +-1 activations of every hidden layer (as the next layer receives them, BEFORE any row negation is undone:
    a negated row produces the same bit, so they compare directly with PyTorch)."""
    act = x_u8.astype(np.int64)
    bits = []
    for d in capas:
        raw = act @ d["signos"].T                          # +-1 inputs (or uint8 for the first layer) x +-1 weights
        if not d["ultima"]:
            act = np.where(raw >= d["umbral"][None, :], 1, -1)
            bits.append(act)
        else:
            sh = Q_MULT - Q_ACT
            logit = ((raw * d["escala"][None, :] + (1 << (sh - 1))) >> sh) + d["shift"][None, :]
            return (logit.argmax(1), bits) if devolver_bits else logit.argmax(1)


# ---------------------------------------------------------------------------------------------------
# 4. C header
# ---------------------------------------------------------------------------------------------------
def _empaquetar(signos):
    """(out, in) +-1 -> uint32 words, ceil(in / 32) per row, bit i of word i / 32 = input i (+1 -> 1)."""
    out, n = signos.shape
    pal = (n + 31) // 32
    bits = np.zeros((out, pal * 32), dtype=np.uint64)
    bits[:, :n] = (signos > 0)
    pesos = (bits.reshape(out, pal, 32) << np.arange(32, dtype=np.uint64)).sum(2)
    return pesos.astype(np.uint64), pal


def escribir_header(capas, ruta, nombre="symbex_mlp"):
    lineas = ["/* Generated by the SYMBEX-1 CDT (export/dense.py). Do not edit by hand. */",
              f"#ifndef {nombre.upper()}_H", f"#define {nombre.upper()}_H", "",
              "#include <stdint.h>", '#include "layers/vector/symbex_dense.h"', "",
              f"#define {nombre.upper()}_Q_ACT {Q_ACT}", f"#define {nombre.upper()}_Q_MULT {Q_MULT}",
              f"#define {nombre.upper()}_N_ENTRADAS {capas[0]['signos'].shape[1]}",
              f"#define {nombre.upper()}_N_CLASES {capas[-1]['signos'].shape[0]}", "",
              "/* Where the weights live: const = flash (default). Compile with -DSYMBEX_ROM= (empty) to put them",
              " * in RAM (faster on chips whose flash cache is smaller than the model, costs that much RAM). */",
              "#ifndef SYMBEX_ROM", "#define SYMBEX_ROM const", "#endif",
              "/* First layer (all give the same result): masks by default (fastest on the ESP32 and the ESP32-S3,",
              " * measured); -DSYMBEX_U8_PLANOS = bit planes, -DSYMBEX_U8_SIMPLE = the simple loop over the set bits. */",
              ""]
    max_n = max(d["signos"].shape[0] for d in capas)
    pasos = []
    for i, d in enumerate(capas):
        out, n_in = d["signos"].shape
        tipo = "uint8 x 1 bit" if d["entrada_u8"] else "1 bit x 1 bit"
        fin = "argmax" if d["ultima"] else "thresholds"
        pasos.append(f"layer {i}: {n_in} -> {out} ({tipo}) + {fin}")
    lineas += ["/* Per-layer profiling: define SYMBEX_MARCA(k) before including this header to time each step",
               " * (k = 0 before the first layer, k = i + 1 after layer i). Empty by default: no cost. */",
               "#ifndef SYMBEX_MARCA", "#define SYMBEX_MARCA(k)", "#endif",
               f"#define {nombre.upper()}_N_PASOS {len(capas)}",
               f"#define {nombre.upper()}_BYTES_BUFFERS {4 * max_n + 2 * 4 * ((max_n + 31) // 32) + 32 * ((capas[0]['signos'].shape[1] + 31) // 32)}"
               "   /* static RAM of symbex_mlp_inferir */",
               "#ifdef SYMBEX_PERFIL",
               f"static const char* const {nombre}_pasos[{len(capas)}] = {{"
               + ", ".join(f'"{p}"' for p in pasos) + "};",
               "#endif", ""]
    for i, d in enumerate(capas):
        out, n_in = d["signos"].shape
        pesos, pal = _empaquetar(d["signos"])
        lineas.append(f"/* layer {i}: {n_in} -> {out}{' (uint8 input)' if d['entrada_u8'] else ''} */")
        lineas.append(f"static SYMBEX_ROM uint32_t {nombre}_w{i}[{out * pal}] = {{")
        for fila in pesos:
            lineas.append("    " + ", ".join(f"0x{int(v):08X}u" for v in fila) + ",")
        lineas.append("};")
        lineas.append(f"static const struct SymbexLayerData {nombre}_capa{i} = {{ {n_in}, {out}, 1, {nombre}_w{i} }};")
        if not d["ultima"]:
            lineas.append(f"static SYMBEX_ROM int32_t {nombre}_t{i}[{out}] = {{"
                          + ", ".join("INT32_MIN" if t == INT32_MIN else str(int(t)) for t in d["umbral"]) + "};")
        else:
            lineas.append(f"static SYMBEX_ROM int64_t {nombre}_escala[{out}] = {{"
                          + ", ".join(f"{int(v)}LL" for v in d["escala"]) + "};")
            lineas.append(f"static SYMBEX_ROM int32_t {nombre}_shift[{out}] = {{"
                          + ", ".join(str(int(v)) for v in d["shift"]) + "};")
        lineas.append("")
    lineas += [f"/* Full inference: x = {capas[0]['signos'].shape[1]} uint8 inputs (same order as the training",
               " * tensor, flattened). Returns the predicted class. Not reentrant (static buffers). */",
               f"static inline int {nombre}_inferir(const uint8_t* x) {{",
               f"    static int32_t raw[{max_n}];",
               f"    static uint32_t bits_a[{(max_n + 31) // 32}], bits_b[{(max_n + 31) // 32}];",
               f"    static uint32_t planos[{8 * ((capas[0]['signos'].shape[1] + 31) // 32)}];   /* bit planes of the input */",
               "    uint32_t* entrada = bits_a;", "    uint32_t* salida = bits_b;", "    SYMBEX_MARCA(0);"]
    for i, d in enumerate(capas):
        if d["entrada_u8"]:
            lineas += ["#if defined(SYMBEX_U8_SIMPLE)", f"    symbex_dense_entrada_u8(x, raw, &{nombre}_capa{i});",
                       "#elif defined(SYMBEX_U8_PLANOS)", f"    symbex_dense_entrada_u8_planos(x, planos, raw, &{nombre}_capa{i});",
                       "#else", f"    symbex_dense_entrada_u8_mascara(x, raw, &{nombre}_capa{i});", "#endif"]
        else:
            lineas.append(f"    symbex_dense_forward(entrada, raw, &{nombre}_capa{i});")
        if not d["ultima"]:
            lineas.append(f"    symbex_dense_umbral(raw, {nombre}_t{i}, salida, {d['signos'].shape[0]});")
            lineas.append("    { uint32_t* t = entrada; entrada = salida; salida = t; }")
            lineas.append(f"    SYMBEX_MARCA({i + 1});")
    sh = Q_MULT - Q_ACT
    lineas += ["    int mejor = 0;", "    int64_t max_logit = INT64_MIN;",
               f"    for (int j = 0; j < {capas[-1]['signos'].shape[0]}; j++) {{",
               f"        int64_t logit = (((int64_t)raw[j] * {nombre}_escala[j] + ((int64_t)1 << {sh - 1})) >> {sh})"
               f" + {nombre}_shift[j];",
               "        if (logit > max_logit) { max_logit = logit; mejor = j; }",
               "    }", f"    SYMBEX_MARCA({len(capas)});", "    (void)entrada;", "    (void)planos;", "    (void)salida;", "    return mejor;", "}", "", f"#endif /* {nombre.upper()}_H */", ""]
    with open(ruta, "w") as f:
        f.write("\n".join(lineas))


# ---------------------------------------------------------------------------------------------------
# 5. exactness of the compiled C on this PC
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


def validar_c(ruta_header, lib_path, x_u8, esperado):
    """Compiles the header + lib_symbex with gcc and runs it on x_u8. Returns (equal, total, message)."""
    if shutil.which("gcc") is None:
        return None, 0, "gcc not found: the C exactness check was skipped"
    inc = os.path.join(lib_path, "include")
    fuentes = [os.path.join(lib_path, "src", "layers", "vector", "symbex_dense.c"),
               os.path.join(lib_path, "src", "hal", "generico", "symbex_hal_impl.c")]
    for r in [inc] + fuentes:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing): set [output] lib_path"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(ruta_header, os.path.join(d, "modelo_symbex.h"))
        with open(os.path.join(d, "main.c"), "w") as f:
            f.write(_MAIN_PRUEBA)
        exe = os.path.join(d, "prueba")
        r = subprocess.run(["gcc", "-O2", "-std=c99", "-I", inc, "-I", d, "-o", exe, os.path.join(d, "main.c")]
                           + fuentes, capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C did not compile:\n" + r.stderr[-2000:]
        datos = os.path.join(d, "x.u8")
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(datos)
        r = subprocess.run([exe, datos, str(len(x_u8))], capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C program failed while running"
        pred_c = np.array([int(v) for v in r.stdout.split()])
    iguales = int((pred_c == esperado).sum())
    return iguales, len(esperado), ""


# ---------------------------------------------------------------------------------------------------
# 6. everything together
# ---------------------------------------------------------------------------------------------------
def a_uint8(x, mean=(), std=()):
    """Normalized tensors (N, C, ...) -> the original uint8 values. Raises if they are not 8-bit data."""
    x = x.detach().cpu().double()
    if mean:
        forma = (1, -1) + (1,) * (x.dim() - 2)
        x = x * torch.tensor(std, dtype=torch.float64).view(forma) + torch.tensor(mean, dtype=torch.float64).view(forma)
    p = x * 255.0
    q = torch.round(p)
    if (q < 0).any() or (q > 255).any() or (p - q).abs().max() > 1e-3:
        raise ValueError("dense export: the inputs are not 8-bit data (x * 255 is not an integer in 0-255 after "
                         "undoing [data] mean / std); the uint8 first layer needs 8-bit inputs")
    return q.to(torch.uint8).reshape(len(q), -1).numpy()


def guardar_validacion(salida, x_u8, y, pred_sim):
    """validacion_x.u8 (N x inputs), validacion_y.u8 (labels), validacion_sim.u8 (classes of the integer
    simulation): what the chip test needs, so it can run later without retraining (probar_chip.py)."""
    np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(os.path.join(salida, "validacion_x.u8"))
    np.asarray(y, dtype=np.uint8).tofile(os.path.join(salida, "validacion_y.u8"))
    np.asarray(pred_sim, dtype=np.uint8).tofile(os.path.join(salida, "validacion_sim.u8"))


def cargar_validacion(salida, n_entradas):
    x = np.fromfile(os.path.join(salida, "validacion_x.u8"), dtype=np.uint8).reshape(-1, n_entradas)
    y = np.fromfile(os.path.join(salida, "validacion_y.u8"), dtype=np.uint8).astype(np.int64)
    sim = np.fromfile(os.path.join(salida, "validacion_sim.u8"), dtype=np.uint8).astype(np.int64)
    return x, y, sim


def exportar_y_validar(modelo, val_dataset, salida, lib_path, mean=(), std=(), n_validacion=2000, verbose=True):
    """Exports `modelo` (all layers 'bin', plain) and validates it. Returns a dict with the numbers."""
    n = min(n_validacion, len(val_dataset))
    xs = torch.stack([val_dataset[i][0] for i in range(n)])
    ys = np.array([int(val_dataset[i][1]) for i in range(n)])
    modelo = modelo.cpu().eval()
    grupos = analizar(modelo, xs[:2])
    capas = plegar(grupos, mean, std)
    ruta_h = os.path.join(salida, "modelo_symbex.h")
    escribir_header(capas, ruta_h)
    x_u8 = a_uint8(xs, mean, std)
    # PyTorch: the +-1 input each binarized layer receives (output of its RSign), to compare layer by layer
    capturas, hooks = {}, []
    for i, (capa, _, _) in enumerate(grupos[1:]):
        hooks.append(capa.rsign.register_forward_hook(lambda m, e, s_, i=i: capturas.__setitem__(i, s_.detach())))
    with torch.no_grad():
        pred_pt = modelo(xs).argmax(1).numpy()
    for h in hooks:
        h.remove()
    pred_sim, bits_sim = simular(capas, x_u8, devolver_bits=True)
    acuerdo_bits = [100 * float((capturas[i].numpy() == bits_sim[i]).mean()) for i in range(len(bits_sim))]
    iguales_c, total_c, msg = validar_c(ruta_h, lib_path, x_u8, pred_sim)
    guardar_validacion(salida, x_u8, ys, pred_sim)
    flash = sum(_empaquetar(d["signos"])[0].size * 4 + d["signos"].shape[0] * (4 if not d["ultima"] else 12)
                for d in capas)
    res = dict(n=n, acc_pytorch=100 * float((pred_pt == ys).mean()), acc_entero=100 * float((pred_sim == ys).mean()),
               fidelidad=100 * float((pred_pt == pred_sim).mean()), acuerdo_bits=acuerdo_bits,
               flash_bytes=flash, header=ruta_h,
               exactitud_c=None if iguales_c is None else 100 * iguales_c / total_c, nota_c=msg,
               _x_u8=x_u8, _y=ys, _pred_sim=pred_sim, _xs=xs)          # "_" keys: for medir_en_chip, not saved to JSON
    if verbose:
        print(f"    layers      : " + " -> ".join(str(d["signos"].shape[1]) for d in capas)
              + f" -> {capas[-1]['signos'].shape[0]}   (first layer: uint8 input)")
        print(f"    header      : {ruta_h}  ({flash / 1024:.1f} KB of weights + thresholds in flash)")
        print(f"    validation  : {n} samples")
        print(f"    accuracy    : PyTorch {res['acc_pytorch']:.2f}%   integer simulation {res['acc_entero']:.2f}%")
        print(f"    FIDELITY    : {res['fidelidad']:.2f}%  (same prediction PyTorch vs. integers: what folding costs)")
        for i, a in enumerate(acuerdo_bits):
            print(f"      hidden layer {i}: {a:.3f}% of the bits equal PyTorch vs. integers"
                  + ("" if a >= 99.5 else "   [!] low: a bug in the folding, not rounding"))
        if res["exactitud_c"] is None:
            print(f"    EXACTNESS   : not checked -- {msg}")
        else:
            estado = "OK" if res["exactitud_c"] == 100 else "[X] MISMATCH: bug in the export or the C engine"
            print(f"    EXACTNESS   : {res['exactitud_c']:.2f}%  (compiled C vs. integer simulation, must be 100%) {estado}")
    return res


# ---------------------------------------------------------------------------------------------------
# 7. on the chip (ESP32 / ESP32-S3): exactness + latency
# ---------------------------------------------------------------------------------------------------
PLACAS = {"esp32": "esp32dev", "s3": "esp32-s3-devkitc-1"}

_MAIN_CHIP = r"""
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "hal/symbex_hal.h"

#define SYMBEX_PERFIL 1
static int64_t marcas[32];
#define SYMBEX_MARCA(k) (marcas[(k)] = esp_timer_get_time())
#include "modelo_symbex.h"
#ifdef SYMBEX_CON_INT8
static int64_t marcas8[32];
#define INT8_MARCA(k) (marcas8[(k)] = esp_timer_get_time())
#include "modelo_int8.h"
#if INT8_N_ENTRADAS != SYMBEX_MLP_N_ENTRADAS
#error "modelo_int8.h and modelo_symbex.h do not have the same number of inputs: export them together"
#endif
#endif

static void leer_exacto(uint8_t* d, size_t n) {
    size_t r = 0;
    while (r < n) {
        int k = uart_read_bytes(UART_NUM_0, d + r, n - r, portMAX_DELAY);
        if (k > 0) r += (size_t)k;
    }
}

void app_main(void) {
    esp_log_level_set("*", ESP_LOG_WARN);
    uart_driver_install(UART_NUM_0, 8192, 0, 0, NULL, 0);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
    uart_set_baudrate(UART_NUM_0, BAUDIOS);
    vTaskDelay(pdMS_TO_TICKS(50));
    static uint8_t x[SYMBEX_MLP_N_ENTRADAS];
    size_t heap_antes = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    symbex_mlp_inferir(x);                              /* first call: the HAL builds its table here, not timed */
    size_t heap_despues = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    printf("\nSYMBEX listo | %s | %d inputs, %d classes\n", symbex_hal_nombre(),
           SYMBEX_MLP_N_ENTRADAS, SYMBEX_MLP_N_CLASES);
    printf("PASOS:%d\n", SYMBEX_MLP_N_PASOS);
    for (int i = 0; i < SYMBEX_MLP_N_PASOS; i++) printf("PASO:%d|%s\n", i, symbex_mlp_pasos[i]);
    printf("RAM:%u,%u,%u,%u\n", (unsigned)SYMBEX_MLP_BYTES_BUFFERS, (unsigned)sizeof x,
           (unsigned)(heap_antes - heap_despues), (unsigned)CONFIG_ESP_MAIN_TASK_STACK_SIZE);
#ifdef SYMBEX_CON_INT8
    symbex_int8_inferir(x);
    printf("INT8:%d,%u\n", INT8_N_CAPAS, (unsigned)INT8_BYTES_BUFFERS);
#endif
    fflush(stdout);
    while (1) {
        uint8_t marca = 0;
        leer_exacto(&marca, 1);
        if (marca == 0x5A) {                            /* memory request: free stack (bytes) of this task */
            printf("PILA:%u\n", (unsigned)uxTaskGetStackHighWaterMark(NULL));
            fflush(stdout);
            continue;
        }
        if (marca != 0xA5) continue;
        leer_exacto(x, sizeof x);
        int c = symbex_mlp_inferir(x);
        printf("R:%d,%lld", c, (long long)(marcas[SYMBEX_MLP_N_PASOS] - marcas[0]));
        for (int i = 0; i < SYMBEX_MLP_N_PASOS; i++) printf(",%lld", (long long)(marcas[i + 1] - marcas[i]));
        printf("\n");
#ifdef SYMBEX_CON_INT8
        int c8 = symbex_int8_inferir(x);              /* the INT8 reference, on the SAME image */
        printf("Q:%d,%lld", c8, (long long)(marcas8[INT8_N_CAPAS] - marcas8[0]));
        for (int i = 0; i < INT8_N_CAPAS; i++) printf(",%lld", (long long)(marcas8[i + 1] - marcas8[i]));
        printf("\n");
#endif
        fflush(stdout);
    }
}
"""


def _build_flags(popcount, pesos, primera):
    banderas = (["-DSYMBEX_POPCOUNT_SIN_LUT16"] if popcount == "swar" else []) + \
               (["-DSYMBEX_ROM="] if pesos == "ram" else []) + (["-DSYMBEX_U8_SIMPLE"] if primera == "simple" else []) + \
               (["-DSYMBEX_U8_PLANOS"] if primera == "planos" else [])
    return ("build_flags = " + " ".join(banderas) + "\n") if banderas else ""


def generar_proyecto_pio(dir_proyecto, ruta_header, lib_path, placa, puerto, baudios, popcount="lut16",
                         pesos="flash", primera="mascara", ruta_int8=None):
    """ESP-IDF project for PlatformIO: the exported header, the lib_symbex sources it needs, and the firmware."""
    if placa not in PLACAS:
        raise ValueError(f"chip: board must be one of {sorted(PLACAS)}")
    src = os.path.join(dir_proyecto, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    for f in os.listdir(dir_proyecto):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(dir_proyecto, f))
    shutil.copytree(os.path.join(lib_path, "include"), os.path.join(src, "include"))
    fuentes = {"symbex_dense.c": os.path.join(lib_path, "src", "layers", "vector", "symbex_dense.c"),
               "symbex_hal_generico.c": os.path.join(lib_path, "src", "hal", "generico", "symbex_hal_impl.c"),
               "symbex_hal_esp32s3.c": os.path.join(lib_path, "src", "hal", "esp32s3", "symbex_hal_impl.c")}
    for destino, origen in fuentes.items():
        if not os.path.exists(origen):
            raise ValueError(f"chip: {origen} not found (check [output] lib_path)")
        shutil.copy(origen, os.path.join(src, destino))
    shutil.copy(ruta_header, os.path.join(src, "modelo_symbex.h"))
    prefijo = f"#define BAUDIOS {baudios}\n"
    if ruta_int8:
        shutil.copy(ruta_int8, os.path.join(src, "modelo_int8.h"))
        prefijo += "#define SYMBEX_CON_INT8 1\n#define SYMBEX_CON_ESP_NN 1\n"
        with open(os.path.join(src, "idf_component.yml"), "w") as f:
            f.write('dependencies:\n  espressif/esp-nn: "*"\n')
    with open(os.path.join(src, "main.c"), "w") as f:
        f.write(prefijo + _MAIN_CHIP)
    with open(os.path.join(src, "CMakeLists.txt"), "w") as f:
        f.write('idf_component_register(SRCS "main.c" ' + " ".join(f'"{d}"' for d in fuentes)
                + ' INCLUDE_DIRS "." "include")\n')
    with open(os.path.join(dir_proyecto, "platformio.ini"), "w") as f:
        f.write(f"[env:{PLACAS[placa]}]\nplatform = espressif32\nboard = {PLACAS[placa]}\nframework = espidf\n"
                f"monitor_speed = 115200\nupload_port = {puerto}\n"
                + _build_flags(popcount, pesos, primera))
    with open(os.path.join(dir_proyecto, "sdkconfig.defaults"), "w") as f:
        f.write("CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                "CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\nCONFIG_ESP32S3_DEFAULT_CPU_FREQ_240=y\n"
                "CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192\nCONFIG_LOG_DEFAULT_LEVEL_WARN=y\nCONFIG_ESP_TASK_WDT_EN=n\n"
                "CONFIG_NN_OPTIMIZED=y\n")
    with open(os.path.join(dir_proyecto, "CMakeLists.txt"), "w") as f:
        f.write("cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                f"project({os.path.basename(os.path.abspath(dir_proyecto))})\n")


def _abrir_puerto(puerto, baudios):
    import serial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = puerto, baudios, 1
    s.dtr = False
    s.rts = False
    s.open()
    s.reset_input_buffer()
    s.rts = True                                        # reset pulse on EN
    import time
    time.sleep(0.15)
    s.rts = False
    return s


def medir_en_chip(ruta_header, x, y, esperado, dir_proyecto, lib_path, placa="esp32", puerto="/dev/ttyUSB0",
                  baudios=921600, n=1000, subir=True, popcount="lut16", flash_modelo=None, verbose=True,
                  pesos="flash", primera="mascara", int8=None):
    """int8 (optional): dict(header=..., esperado=classes of its integer simulation, flash=bytes) to run the
    INT8 reference of the original model on the same chip and the same images (Espressif's ESP-NN kernel)."""
    """Flashes the exported model and measures it on the board: equality with the integer simulation (must be
    100%), accuracy, latency (total and per layer) and RAM. x / y / esperado: validation images (uint8), labels
    and the classes of the integer simulation. Returns a dict, or raises RuntimeError with what went wrong."""
    import time
    generar_proyecto_pio(dir_proyecto, ruta_header, lib_path, placa, puerto, baudios, popcount, pesos, primera,
                         int8["header"] if int8 else None)
    if subir:
        if shutil.which("pio") is None:
            raise RuntimeError("chip: PlatformIO (pio) not found")
        if verbose:
            print(f"    building and flashing {PLACAS[placa]} on {puerto} (popcount {popcount}, weights in {pesos}, "
                  f"first layer {primera}; the first build downloads ESP-IDF)...")
        r = subprocess.run(["pio", "run", "-d", dir_proyecto, "--target", "upload"], capture_output=not verbose)
        if r.returncode != 0:
            raise RuntimeError("chip: build or upload failed (run `pio run -d " + dir_proyecto + " -t upload` to see it)")
    n = min(n, len(x))
    pred, totales, por_paso, nombres, ram, pila = [], [], [], {}, None, None
    pred8, totales8, por_paso8, info8 = [], [], [], None
    with _abrir_puerto(puerto, baudios) as s:
        t0, avisado, hal = time.time(), False, "?"
        while True:
            l = s.readline().decode(errors="replace").strip()
            if "SYMBEX listo" in l:
                hal = l.split("|")[1].strip() if "|" in l else "?"
            elif l.startswith("PASO:"):
                i, texto = l[5:].split("|", 1)
                nombres[int(i)] = texto
            elif l.startswith("RAM:"):
                ram = [int(v) for v in l[4:].split(",")]
                if not int8:
                    break
            elif l.startswith("INT8:"):
                info8 = [int(v) for v in l[5:].split(",")]
                break
            if not avisado and time.time() - t0 > 8:
                print("    >>> the board did not answer the reset: press EN/RST once")
                avisado = True
            if time.time() - t0 > 60:
                raise RuntimeError("chip: the board did not start (no 'SYMBEX listo')")
        if verbose:
            print(f"    board ready | popcount: {hal}")
        for i in range(n):
            s.write(bytes([0xA5]) + np.ascontiguousarray(x[i], dtype=np.uint8).tobytes())
            limite = time.time() + 10
            while True:
                l = s.readline().decode(errors="replace").strip()
                if l.startswith("R:"):
                    v = [int(t) for t in l[2:].split(",")]
                    pred.append(v[0])
                    totales.append(v[1])
                    por_paso.append(v[2:])
                    if not int8:
                        break
                elif l.startswith("Q:"):
                    v = [int(t) for t in l[2:].split(",")]
                    pred8.append(v[0])
                    totales8.append(v[1])
                    por_paso8.append(v[2:])
                    break
                if time.time() > limite:
                    raise RuntimeError(f"chip: no answer for image {i}")
            if verbose and (i + 1) % max(1, n // 5) == 0:
                print(f"      {i + 1}/{n} | equal to the simulation {100 * np.mean(np.array(pred) == esperado[:len(pred)]):.2f}%"
                      f" | median {np.median(totales) / 1000:.3f} ms")
        s.write(bytes([0x5A]))
        limite = time.time() + 5
        while time.time() < limite:
            l = s.readline().decode(errors="replace").strip()
            if l.startswith("PILA:"):
                pila = int(l[5:])
                break
    pred, por_paso = np.array(pred), np.array(por_paso)
    medianas = np.median(por_paso, axis=0) / 1000
    buffers, entrada, tabla, pila_total = ram
    pila_usada = None if pila is None else pila_total - pila
    out = dict(placa=placa, hal=hal, popcount=popcount, pesos=pesos, primera=primera, n=n,
               igual_simulacion=100 * float((pred == esperado[:n]).mean()),
               precision=100 * float((pred == y[:n]).mean()), mediana_ms=float(np.median(totales)) / 1000,
               min_ms=float(np.min(totales)) / 1000,
               capas=[dict(paso=nombres.get(i, f"step {i}"), mediana_ms=float(m)) for i, m in enumerate(medianas)],
               ram=dict(buffers=buffers, entrada=entrada, tabla_popcount=tabla, pila_usada=pila_usada,
                        pesos_en_ram=(flash_modelo or 0) if pesos == "ram" else 0,
                        total=buffers + entrada + tabla + (pila_usada or 0)
                        + ((flash_modelo or 0) if pesos == "ram" else 0)),
               flash_modelo=flash_modelo)
    if int8:
        p8 = np.array(pred8)
        out["int8"] = dict(igual_simulacion=100 * float((p8 == int8["esperado"][:n]).mean()),
                           precision=100 * float((p8 == y[:n]).mean()), mediana_ms=float(np.median(totales8)) / 1000,
                           capas_ms=[float(m) for m in np.median(np.array(por_paso8), axis=0) / 1000],
                           ram_buffers=info8[1] if info8 else None, flash=int8.get("flash"))
    if verbose:
        estado = "OK" if out["igual_simulacion"] == 100 else "[X] MISMATCH between the chip and the simulation"
        print(f"\n    ON THE CHIP  ({PLACAS[placa]}, 240 MHz, one core, {hal}, weights in {pesos}, first layer {primera})")
        print(f"    images        : {n} | equal to the integer simulation {out['igual_simulacion']:.2f}% {estado}")
        print(f"    accuracy      : {out['precision']:.2f}%")
        print(f"    latency       : median {out['mediana_ms']:.3f} ms | min {out['min_ms']:.3f} ms")
        print(f"\n    {'step':<52}{'ms':>9}{'%':>7}")
        print("    " + "-" * 68)
        suma = float(medianas.sum()) or 1.0
        for c in out["capas"]:
            barra = "#" * int(round(20 * c["mediana_ms"] / suma))
            print(f"    {c['paso']:<52}{c['mediana_ms']:>9.3f}{100 * c['mediana_ms'] / suma:>6.1f}%  {barra}")
        print("    " + "-" * 68)
        print(f"    {'total (sum of the steps)':<52}{suma:>9.3f}")
        r = out["ram"]
        print(f"\n    RAM (measured on the chip)")
        print(f"    model buffers (static)    : {r['buffers'] / 1024:7.2f} KB")
        print(f"    input image (static)      : {r['entrada'] / 1024:7.2f} KB")
        print(f"    popcount table (heap)     : {r['tabla_popcount'] / 1024:7.2f} KB")
        if pesos == "ram":
            print(f"    weights + thresholds (RAM): {r['pesos_en_ram'] / 1024:7.2f} KB")
        print(f"    stack used by inference   : " + ("   n/a" if pila_usada is None else f"{pila_usada / 1024:7.2f} KB")
              + f"  (of {pila_total / 1024:.0f} KB reserved)")
        print(f"    TOTAL                     : {r['total'] / 1024:7.2f} KB")
        if flash_modelo:
            print(f"    flash (weights+thresholds): {flash_modelo / 1024:7.2f} KB")
        if int8:
            r8 = out["int8"]
            estado8 = "OK" if r8["igual_simulacion"] == 100 else "[X] MISMATCH"
            print(f"\n    INT8 REFERENCE ON THE SAME CHIP  (original model, ESP-NN esp_nn_fully_connected_s8, weights in flash"
                  + (": --weights ram moves only the SYMBEX weights" if pesos == "ram" else "") + ")")
            print(f"    equal to its integer simulation {r8['igual_simulacion']:.2f}% {estado8}")
            print("    " + "  ".join(f"layer {i}: {m:.3f} ms" for i, m in enumerate(r8["capas_ms"])))
            print(f"\n    {'model':<26}{'accuracy':>9}{'latency ms':>12}{'speed vs INT8':>15}{'flash KB':>10}{'RAM KB':>9}")
            print("    " + "-" * 81)
            ram8 = (r8["ram_buffers"] or 0) + out["ram"]["entrada"]
            print(f"    {'INT8 (ESP-NN)':<26}{r8['precision']:>8.2f}%{r8['mediana_ms']:>12.3f}{1.0:>14.2f}x"
                  f"{(r8['flash'] or 0) / 1024:>10.1f}{ram8 / 1024:>9.2f}")
            print(f"    {'SYMBEX':<26}{out['precision']:>8.2f}%{out['mediana_ms']:>12.3f}"
                  f"{r8['mediana_ms'] / out['mediana_ms']:>14.2f}x{(flash_modelo or 0) / 1024:>10.1f}{out['ram']['total'] / 1024:>9.2f}")
            print("    (speed vs INT8 > 1: SYMBEX is faster. RAM: model buffers + input (+ table, stack, weights for SYMBEX))")
    return out


# ---------------------------------------------------------------------------------------------------
# 8. INT8 reference of the ORIGINAL model, for the comparison on the chip (TFLite / ESP-NN arithmetic)
# ---------------------------------------------------------------------------------------------------
def _grupos_fp32(modelo, x):
    """Original FP32 MLP: [(Linear, BatchNorm1d or None, ReLU?)] in forward order (Flatten / Dropout skipped)."""
    orden, hooks = [], []
    for m in modelo.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(lambda mod, e, s_: orden.append(mod)))
    estaba = modelo.training
    modelo.eval()
    with torch.no_grad():
        modelo(x)
    modelo.train(estaba)
    for h in hooks:
        h.remove()
    grupos, vistos, actual = [], set(), None
    for m in orden:
        if id(m) in vistos:
            continue
        vistos.add(id(m))
        if isinstance(m, (nn.Flatten, nn.Dropout, nn.Identity)):
            continue
        if isinstance(m, nn.Linear):
            actual = [m, None, False]
            grupos.append(actual)
        elif isinstance(m, nn.BatchNorm1d) and actual is not None and actual[1] is None and not actual[2]:
            actual[1] = m
        elif isinstance(m, nn.ReLU) and actual is not None:
            actual[2] = True
        else:
            raise ValueError(f"INT8 reference: unsupported module {type(m).__name__} (Linear -> [BatchNorm1d] -> [ReLU] chains)")
    return grupos


def cuantizar_multiplicador(M):
    """Real multiplier -> (Q31 multiplier, shift), TFLite convention (shift > 0: left)."""
    if M == 0:
        return 0, 0
    q, shift = np.frexp(M)
    qf = int(round(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    return qf, int(shift)


def mult_cuantizado(x, mult, shift):
    """TFLite MultiplyByQuantizedMultiplier in exact integers (vectorized): what ESP-NN computes."""
    x = x.astype(np.int64)
    izq, der = max(shift, 0), max(-shift, 0)
    ab = (x * (1 << izq)) * mult
    alto = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    alto = np.where(alto >= 0, alto >> 31, -((-alto) >> 31))            # division truncated toward zero
    alto = np.clip(alto, -(1 << 31), (1 << 31) - 1)
    if der == 0:
        return alto
    mascara = (1 << der) - 1
    return (alto >> der) + ((alto & mascara) > ((mascara >> 1) + (alto < 0)))


def preparar_int8(modelo, calib, mean=(), std=()):
    """Post-training INT8 of the ORIGINAL model, like TFLite: per-tensor weights (symmetric, +-127), uint8
    input as int8 with zero point -128 (x / 255), every hidden output calibrated on `calib` (normalized
    tensors) as int8 with zero point -128, bias int32, multiplier + shift per layer. BatchNorm and the input
    normalization are folded into the Linear layers. Returns a list of layer dicts."""
    if std and len(set(std)) != 1:
        raise ValueError("INT8 reference: the input normalization must use ONE std for every channel")
    modelo = modelo.cpu().eval()
    grupos = _grupos_fp32(modelo, calib[:2])
    capas_f = []
    for i, (lin, bn, relu) in enumerate(grupos):
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
        capas_f.append((W, b, relu))
    # calibration of every hidden output range (after its ReLU), in float, from the folded layers
    a = (calib.reshape(len(calib), -1).double().numpy() * (std[0] if std else 1.0)
         + (np.repeat(np.asarray(mean), calib[0].numel() // len(mean)) if mean else 0.0))          # = p / 255
    maximos = []
    for W, b, relu in capas_f[:-1]:
        a = a @ W.T + b
        if relu:
            a = np.maximum(a, 0)
        maximos.append(max(float(np.percentile(a, 99.99)), 1e-6))
    s_in = 1.0 / 255.0
    out = []
    for i, (W, b, relu) in enumerate(capas_f):
        s_w = max(np.abs(W).max(), 1e-12) / 127.0
        Wq = np.clip(np.rint(W / s_w), -127, 127).astype(np.int8)
        bias = np.rint(b / (s_in * s_w)).astype(np.int64)
        if i == len(capas_f) - 1:
            out.append(dict(W=Wq, bias=bias, mult=None, shift=None))
        else:
            if not relu:
                raise ValueError("INT8 reference: hidden layers must end in ReLU (unsigned activations)")
            s_out = maximos[i] / 255.0
            mult, shift = cuantizar_multiplicador(s_in * s_w / s_out)
            out.append(dict(W=Wq, bias=bias, mult=mult, shift=shift))
            s_in = s_out
    return out


def simular_int8(q, x_u8):
    """Exact integer simulation of the INT8 reference (ESP-NN / TFLite arithmetic). Returns the classes."""
    a = x_u8.astype(np.int64) - 128                                       # int8, zero point -128
    for c in q:
        acc = (a + 128) @ c["W"].astype(np.int64).T + c["bias"]          # input_offset = +128
        if c["mult"] is None:
            return acc.argmax(1)                                          # last layer: argmax in int32
        a = np.clip(mult_cuantizado(acc, c["mult"], c["shift"]) - 128, -128, 127)


def escribir_header_int8(q, ruta):
    """modelo_int8.h: symbex_int8_inferir(const uint8_t* x). With SYMBEX_CON_ESP_NN it uses
    esp_nn_fully_connected_s8 (Espressif's optimized kernel); otherwise a plain C version of the same
    arithmetic (used to check the header on the PC)."""
    dims = [q[0]["W"].shape[1]] + [c["W"].shape[0] for c in q]
    L = [f"/* Generated by the SYMBEX-1 CDT: INT8 reference (post-training, TFLite / ESP-NN arithmetic). */",
         "#ifndef SYMBEX_INT8_H", "#define SYMBEX_INT8_H", "", "#include <stdint.h>", "#include <string.h>",
         "#ifdef SYMBEX_CON_ESP_NN", '#include "esp_nn.h"', "#endif",
         "/* Where the INT8 weights live: const = flash (default); -DSYMBEX_INT8_ROM= (empty) puts them in RAM.",
         " * Separate from SYMBEX_ROM, so the binary model can go to RAM while the INT8 one stays in flash. */",
         "#ifndef SYMBEX_INT8_ROM", "#define SYMBEX_INT8_ROM const", "#endif",
         "#ifndef INT8_MARCA", "#define INT8_MARCA(k)", "#endif",
         f"#define INT8_N_CAPAS {len(q)}", f"#define INT8_N_ENTRADAS {dims[0]}",
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
    L += ["", "#ifndef SYMBEX_CON_ESP_NN",
          "/* TFLite MultiplyByQuantizedMultiplier, the same arithmetic as ESP-NN */",
          "__attribute__((unused)) static inline int32_t int8_mbqm(int32_t x, int32_t mult, int shift) {",
          "    int izq = shift > 0 ? shift : 0, der = shift > 0 ? 0 : -shift;",
          "    int64_t ab = (int64_t)x * (int64_t)(1 << izq) * (int64_t)mult;",
          "    int64_t r = ab + (ab >= 0 ? (1LL << 30) : (1 - (1LL << 30)));",
          "    int64_t alto = r >= 0 ? (r >> 31) : -((-r) >> 31);",
          "    if (alto > INT32_MAX) alto = INT32_MAX;",
          "    if (alto < INT32_MIN) alto = INT32_MIN;",
          "    if (der == 0) return (int32_t)alto;",
          "    int32_t a = (int32_t)alto, m = (int32_t)((1LL << der) - 1);",
          "    return (a >> der) + ((a & m) > ((m >> 1) + (a < 0)));",
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
    L += [f"static inline int symbex_int8_inferir(const uint8_t* x) {{",
          f"    static int8_t a[{maxd}] __attribute__((aligned(16))), b[{maxd}] __attribute__((aligned(16)));",
          f"    INT8_MARCA(0);",
          f"    for (int i = 0; i < {dims[0]}; i++) a[i] = (int8_t)((int)x[i] - 128);"]
    for k in range(len(q) - 1):
        L += ["#ifdef SYMBEX_CON_ESP_NN",
              f"    esp_nn_fully_connected_s8(a, 128, {dims[k]}, int8_w{k}, 0, int8_b{k}, b, {dims[k + 1]}, -128,",
              f"                              INT8_SHIFT[{k}], INT8_MULT[{k}], -128, 127);",
              "#else",
              f"    int8_fc(a, {dims[k]}, int8_w{k}, int8_b{k}, b, {dims[k + 1]}, INT8_MULT[{k}], INT8_SHIFT[{k}]);",
              "#endif",
              f"    memcpy(a, b, {dims[k + 1]});",
              f"    INT8_MARCA({k + 1});"]
    k = len(q) - 1
    L += ["    (void)b;", "    int mejor = 0;", "    int32_t max_acc = 0;",
          f"    for (int j = 0; j < {dims[k + 1]}; j++) {{",
          f"        int32_t acc = int8_b{k}[j];",
          f"        const int8_t* w = &int8_w{k}[j * {dims[k]}];",
          f"        for (int i = 0; i < {dims[k]}; i++) acc += ((int32_t)a[i] + 128) * w[i];",
          "        if (j == 0 || acc > max_acc) { max_acc = acc; mejor = j; }",
          "    }",
          f"    INT8_MARCA({len(q)});",
          "    return mejor;", "}", "", "#endif /* SYMBEX_INT8_H */", ""]
    with open(ruta, "w") as f:
        f.write("\n".join(L))
    return sum(c["W"].size + 4 * len(c["bias"]) for c in q)


def exportar_int8(modelo_original, calib, val_xs, x_u8, ys, salida, lib_path, mean=(), std=(), verbose=True):
    """Builds the INT8 reference of the original model, writes modelo_int8.h + validacion_int8.u8, and checks
    its plain C version against the numpy simulation on this PC. Returns a dict with its numbers."""
    q = preparar_int8(modelo_original, calib, mean, std)
    ruta = os.path.join(salida, "modelo_int8.h")
    flash = escribir_header_int8(q, ruta)
    pred = simular_int8(q, x_u8)
    np.asarray(pred, dtype=np.uint8).tofile(os.path.join(salida, "validacion_int8.u8"))
    with torch.no_grad():
        pred_fp = modelo_original.cpu().eval()(val_xs).argmax(1).numpy()
    exact = None
    if shutil.which("gcc"):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(ruta, os.path.join(d, "modelo_int8.h"))
            with open(os.path.join(d, "main.c"), "w") as f:
                f.write('#include <stdio.h>\n#include <stdlib.h>\n#include "modelo_int8.h"\n'
                        "int main(int argc, char** argv) { (void)argc; FILE* f = fopen(argv[1], \"rb\");\n"
                        "  static uint8_t x[INT8_N_ENTRADAS]; int n = atoi(argv[2]);\n"
                        "  for (int i = 0; i < n; i++) { if (fread(x, 1, INT8_N_ENTRADAS, f) != INT8_N_ENTRADAS) return 1;\n"
                        "    printf(\"%d\\n\", symbex_int8_inferir(x)); } return 0; }\n")
            exe = os.path.join(d, "p")
            r = subprocess.run(["gcc", "-O2", "-std=c99", "-I", d, "-o", exe, os.path.join(d, "main.c")],
                               capture_output=True, text=True)
            if r.returncode == 0:
                np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(os.path.join(d, "x.u8"))
                out = subprocess.run([exe, os.path.join(d, "x.u8"), str(len(x_u8))], capture_output=True, text=True).stdout
                exact = 100 * float((np.array([int(v) for v in out.split()]) == pred).mean())
    res = dict(acc_int8=100 * float((pred == ys).mean()), acuerdo_fp32=100 * float((pred == pred_fp).mean()),
               flash_bytes=flash, exactitud_c=exact, header=ruta)
    if verbose:
        print(f"    INT8 reference (original model, post-training like TFLite / ESP-NN): accuracy {res['acc_int8']:.2f}%"
              f" | same class as FP32 {res['acuerdo_fp32']:.2f}% | {flash / 1024:.1f} KB"
              + ("" if exact is None else f" | C vs. simulation {exact:.2f}%"))
    return res
