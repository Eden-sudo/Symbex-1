"""
int_export.py -- an INT2 / INT4 model of fashion_int4.py as C for the board, in the CDT's export format.

    python int_export.py --bits 2 --seed 42         # -> runs/lowbit_chip/int2_s42/
    python ../../../tools/chip_test.py runs/lowbit_chip/int2_s42 --board s3 --port /dev/ttyUSB0

Run from benchmarks/mlp/mlp_fashion/ (needs models/int<bits>_qat_s<seed>.npz, which fashion_int4.py saves). The folder
holds what the CDT's chip stage and tools/chip_test.py read: model_symbex.h, the validation images and export.json.

The C reproduces fashion_int4.py's integer evaluation (predecir_entero) exactly:
    every layer's input is an unsigned byte u = x - zero point (0..255: the pixel; then ReLU outputs, zero point -128)
    acc = sum of Wq * u + bq, with Wq split into bit planes, each plane summed by lib_symbex's
          symbex_u8_sum_bits_rows (the library's uint8 first-layer kernel: SWAR / the S3's SIMD): no multiplication
              INT2  codes -1, 0, +1: two planes, acc = S(+1 plane) - S(-1 plane)
              INT4  codes -7..7 in two's complement: four planes, acc = S0 + 2 S1 + 4 S2 - 8 S3
    x = clip(rint(acc * multiplier) + zero point, -128, 127), the multiplier a float64 as in numpy: double and
        rint (round half to even) give exactly numpy's result; on the ESP32 / S3 the double is software (one per output)
    the class: the first largest output
The compiled C is checked on this PC against that evaluation on the --images test images (must be 100 %).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mlp_fashion/ (this script lives in earlier/)
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools"))
LIB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Symbex1_PIO", "lib", "lib_symbex"))
COEF = {2: (1, -1), 4: (1, 2, 4, -8)}


def load(bits, seed):
    d = np.load(os.path.join(HERE, "models", f"int{bits}_qat_s{seed}.npz"))
    layers = [dict(W=d[f"W{i}"].astype(np.int64), b=d[f"b{i}"].astype(np.int64), m=d[f"m{i}"].astype(np.float64),
                   zin=int(d[f"zin{i}"]), zout=int(d[f"zout{i}"])) for i in range(int(d["n_layers"]))]
    for i, L in enumerate(layers[1:], 1):
        if L["zin"] != -128:
            sys.exit(f"[X] layer {i}: input zero point {L['zin']}, not -128: its input is not an unsigned byte")
    if layers[0]["zin"] != -128:
        sys.exit("[X] the first layer's input zero point is not -128 (the pixel - 128)")
    return layers


def simulate(layers, x_u8):
    """fashion_int4.py's predecir_entero, the same arithmetic"""
    x = x_u8.astype(np.int64) - 128
    for L in layers:
        acc = (x - L["zin"]) @ L["W"].T + L["b"]
        x = np.clip(np.round(acc * L["m"]) + L["zout"], -128, 127).astype(np.int64)
    return x.argmax(1)


def planes(W, bits):
    """(n_out, n_in) codes -> list of 0/1 planes such that W = sum of COEF[bits][p] * plane[p]"""
    if bits == 2:
        return [(W == 1).astype(np.uint8), (W == -1).astype(np.uint8)]
    v = W & 0xF                                                          # two's complement, 4 bits
    return [((v >> p) & 1).astype(np.uint8) for p in range(4)]


def words(bits_rows, rw):
    n_out, n_in = bits_rows.shape
    pad = np.zeros((n_out, rw * 32), dtype=np.uint8)
    pad[:, :n_in] = bits_rows
    return np.packbits(pad, axis=1, bitorder="little").view("<u4")


def c_array(f, ctype, name, values, attr=""):
    values = np.asarray(values).ravel()
    f.write(f"static SYMBEX_ROM {ctype} {name}[{len(values)}]{attr} = {{")
    for i, v in enumerate(values):
        lit = (f"0x{int(v):08X}u" if ctype == "uint32_t" else float(v).hex() if ctype == "double" else str(int(v)))
        f.write(("\n    " if i % 8 == 0 else " ") + lit + ",")
    f.write("\n};\n")


def write_header(layers, bits, path):
    coef = COEF[bits]
    flash, max_rw, sizes = 0, 0, []
    with open(path, "w") as f:
        f.write(f"/* INT{bits} MLP (fashion_int4.py, QAT) for the board, written by int_export.py: bit planes of the weight "
                f"codes summed\n   by lib_symbex's symbex_u8_sum_bits_rows, float64 requantization as in numpy. */\n"
                "#ifndef SYMBEX_INT_MODEL_H\n#define SYMBEX_INT_MODEL_H\n#include <stdint.h>\n#include <math.h>\n"
                "#include \"hal/symbex_hal.h\"\n#ifndef SYMBEX_ROM\n#define SYMBEX_ROM const\n#endif\n"
                "#ifndef SYMBEX_MARK\n#define SYMBEX_MARK(k) ((void)0)\n#endif\n\n")
        for i, L in enumerate(layers):
            n_out, n_in = L["W"].shape
            rw = -(-(-(-n_in // 32)) // 4) * 4
            max_rw = max(max_rw, rw)
            sizes.append((n_in, n_out, rw))
            for p, pl in enumerate(planes(L["W"], bits)):
                c_array(f, "uint32_t", f"L{i}_P{p}", words(pl, rw), " __attribute__((aligned(16)))")
                flash += n_out * rw * 4
            c_array(f, "int32_t", f"L{i}_B", L["b"])
            c_array(f, "double", f"L{i}_M", L["m"])
            flash += n_out * 12
        widest = max(n for s in sizes for n in s[:2])
        f.write(f"static uint8_t INT_U[2][{widest}] __attribute__((aligned(16)));\n"
                f"static uint8_t INT_WS[{max_rw * 32}] __attribute__((aligned(16)));\n"
                f"static int32_t INT_ACC[{widest}], INT_SUM[{widest}];\n\n"
                "static void int_layer(const uint8_t* u, int n_in, int n_out, int rw, const uint32_t* const* pl,\n"
                "                      const int* coef, int n_planes, const int32_t* b, const double* m, int zout,\n"
                "                      uint8_t* u_next, int32_t* x_out) {\n"
                "    symbex_u8_prepare(u, (uint16_t)n_in, INT_WS, (uint16_t)rw);\n"
                "    for (int o = 0; o < n_out; o++) INT_ACC[o] = b[o];\n"
                "    for (int p = 0; p < n_planes; p++) {\n"
                "        symbex_u8_sum_bits_rows(INT_WS, (uint16_t)n_in, pl[p], (uint16_t)rw, (uint16_t)n_out, INT_SUM);\n"
                "        for (int o = 0; o < n_out; o++) INT_ACC[o] += coef[p] * INT_SUM[o];\n"
                "    }\n"
                "    for (int o = 0; o < n_out; o++) {\n"
                "        double r = rint((double)INT_ACC[o] * m[o]) + zout;\n"
                "        int32_t x = r < -128 ? -128 : (r > 127 ? 127 : (int32_t)r);\n"
                "        if (u_next) u_next[o] = (uint8_t)(x + 128); else x_out[o] = x;\n"
                "    }\n}\n\n"
                f"static const int INT_COEF[{len(coef)}] = {{{', '.join(map(str, coef))}}};\n")
        for i in range(len(layers)):
            f.write(f"static const uint32_t* const L{i}_PL[{len(coef)}] = {{"
                    + ", ".join(f"L{i}_P{p}" for p in range(len(coef))) + "};\n")
        steps = [f"layer {i}: {n_in} -> {n_out}, INT{bits} ({len(coef)} planes)" for i, (n_in, n_out, _) in enumerate(sizes)]
        ram = 2 * widest + max_rw * 32 + 8 * widest
        f.write(f"\n#define SYMBEX_MLP_N_INPUTS {sizes[0][0]}\n#define SYMBEX_MLP_N_CLASSES {sizes[-1][1]}\n"
                f"#define SYMBEX_MLP_N_STEPS {len(layers)}\n#define SYMBEX_MLP_BYTES_BUFFERS {ram}u\n"
                "static const char* const symbex_mlp_steps[SYMBEX_MLP_N_STEPS] = {\n"
                + "".join(f"    \"{s}\",\n" for s in steps) + "};\n\n"
                "static inline int symbex_mlp_infer(const uint8_t* x) {\n"
                "    int32_t out[SYMBEX_MLP_N_CLASSES];\n    const uint8_t* u = x;      /* the pixel: x - (-128) - 128 */\n"
                "    SYMBEX_MARK(0);\n")
        for i, (n_in, n_out, rw) in enumerate(sizes):
            last = i == len(sizes) - 1
            nxt = "NULL" if last else f"INT_U[{i % 2}]"
            f.write(f"    int_layer(u, {n_in}, {n_out}, {rw}, L{i}_PL, INT_COEF, {len(coef)}, L{i}_B, L{i}_M, "
                    f"{layers[i]['zout']}, {nxt}, out);\n    SYMBEX_MARK({i + 1});\n" + ("" if last else f"    u = INT_U[{i % 2}];\n"))
        f.write("    int best = 0;\n    for (int c = 1; c < SYMBEX_MLP_N_CLASSES; c++) if (out[c] > out[best]) best = c;\n"
                "    return best;\n}\n#endif\n")
    return flash, ram, steps


def check_c(header, x_u8, expected):
    sys.path.insert(0, TOOLS)
    from compiler_core.export.dense import _TEST_MAIN
    srcs = [os.path.join(LIB, "src", "hal", "generic", "symbex_hal_impl.c")]
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header, os.path.join(d, "model_symbex.h"))
        open(os.path.join(d, "main.c"), "w").write(_TEST_MAIN)
        exe = os.path.join(d, "check")
        r = subprocess.run(["gcc", "-O2", "-std=gnu11", "-I", os.path.join(LIB, "include"), "-I", d, "-o", exe,
                            os.path.join(d, "main.c")] + srcs + ["-lm"], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("[X] the C did not compile:\n" + r.stderr[-2000:])
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(os.path.join(d, "x.u8"))
        r = subprocess.run([exe, os.path.join(d, "x.u8"), str(len(x_u8))], capture_output=True, text=True)
        got = np.array([int(v) for v in r.stdout.split()])
    return int((got == expected).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=[2, 4], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images", type=int, default=2000, help="test images saved for the board and the C check")
    args = ap.parse_args()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    sys.path.insert(1, TOOLS)
    from model import validation_u8
    from compiler_core.export.dense import save_validation
    layers = load(args.bits, args.seed)
    x, y = validation_u8(args.images)
    x = x.reshape(len(x), -1)
    sim = simulate(layers, x)
    out = os.path.join("runs", "lowbit_chip", f"int{args.bits}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    header = os.path.join(out, "model_symbex.h")
    flash, ram, steps = write_header(layers, args.bits, header)
    equal = check_c(header, x, sim)
    save_validation(out, x, y, sim)
    acc = 100 * float((sim == y).mean())
    res = dict(method=f"int{args.bits}", seed=args.seed, n=len(x), acc_integer=acc, flash_bytes=flash, ram_buffers=ram,
               steps=steps, header=header, c_exactness=100 * equal / len(x),
               arithmetic=f"INT{args.bits}: bit planes summed by symbex_u8_sum_bits_rows, float64 requantization")
    json.dump(res, open(os.path.join(out, "export.json"), "w"), indent=1)
    print(f"    INT{args.bits} s{args.seed}: integer accuracy {acc:.2f}% ({len(x)} test images) | {flash / 1024:.1f} KB in flash | "
          f"C == simulation {100 * equal / len(x):.2f}% " + ("OK" if equal == len(x) else "[X]") + f"\n    -> {out}/")


if __name__ == "__main__":
    main()
