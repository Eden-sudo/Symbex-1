"""
export_c.py -- an integer model of quantize.py (INT8 / INT4 / INT2) as C for the board, running on Espressif's ESP-NN
kernels: ordinary multiply-accumulate, no SYMBEX arithmetic. The folder it writes is what tools/chip_test.py measures.

    folder = export_folder(layers, x_float, y, "runs/lowbit_chip/int4_qat_s42")
    python tools/chip_test.py <folder> --board s3 --port /dev/ttyACM0 --usb

The header (model_symbex.h, the interface chip_test's firmware expects) holds:
  weights   INT8 as int8; INT4 two codes per byte, INT2 four, low bits first, two's complement; every output
            channel starts on a byte. Flash by default; with the chip test's RAM option (-DSYMBEX_ROM=) the codes are
            unpacked to int8 in RAM once, at the first call
  layers    Linear: esp_nn_fully_connected_per_ch_s8; Conv1d / Conv2d: esp_nn_conv_s8 (a Conv1d is a conv of height 1);
            MaxPool: esp_nn_max_pool_s8. Activations int8, NHWC. With the weights in flash, INT4 / INT2 rows are
            unpacked in groups of output channels into a scratch buffer of SCRATCH bytes before each kernel call
  input     the int8 image (quantize.quantize_input), NHWC, sent as bytes; output: the first largest class
The compiled C is checked on this PC, with ESP-NN's reference (ANSI) kernels, against the integer simulation: 100 %.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from lowbit.quantize import quantize_input, run_integer

SCRATCH = 8192                     # bytes of unpacked int8 weights per kernel call (weights in flash, INT4 / INT2)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def find_esp_nn():
    """ESP-NN sources for the PC check: $ESP_NN_PATH, or a copy ESP-IDF downloaded for an earlier chip project"""
    cands = [os.environ.get("ESP_NN_PATH", "")]
    for root in (REPO, os.path.expanduser("~/.platformio"), os.path.expanduser("~/esp")):
        cands += glob.glob(os.path.join(root, "**", "managed_components", "espressif__esp-nn"), recursive=True)
        cands += glob.glob(os.path.join(root, "**", "esp-nn"), recursive=True)
    for c in cands:
        if c and os.path.exists(os.path.join(c, "include", "esp_nn.h")):
            return c
    return None


def _pack(codes, bits):
    """one output channel's codes -> bytes (low bits first, two's complement)"""
    if bits == 8:
        return codes.astype(np.int8).view(np.uint8)
    per = 8 // bits
    c = (codes.astype(np.int64) & ((1 << bits) - 1)).astype(np.uint8)
    c = np.concatenate([c, np.zeros((-len(c)) % per, dtype=np.uint8)]).reshape(-1, per)
    return np.bitwise_or.reduce(c << (np.arange(per, dtype=np.uint8) * bits), axis=1).astype(np.uint8)


def _c_array(f, ctype, name, values, attr="", rom="SYMBEX_ROM"):
    v = np.asarray(values).reshape(-1)
    f.write(f"static {rom} {ctype} {name}[{len(v)}]{attr} = {{\n")
    for i in range(0, len(v), 24):
        f.write("    " + ", ".join(str(int(t)) for t in v[i:i + 24]) + ",\n")
    f.write("};\n")


def plan(layers, in_shape):
    """the C layers with their shapes (NHWC) and weights in C order; in_shape: one input as the model takes it"""
    shape = tuple(in_shape)                         # (F,) or (C, L) or (C, H, W)
    chw = (shape[0], 1, shape[1]) if len(shape) == 2 else (shape if len(shape) == 3 else None)
    flat_from = None                                # (C, H, W) of the tensor a Flatten turned into a vector
    steps = []
    for c in layers:
        if c["op"] == "input":
            continue
        if c["op"] == "flatten":
            flat_from = chw
            continue
        if c["op"] == "maxpool":
            C, H, W = chw
            (kh, kw), (sh, sw) = c["k"], c["s"]
            Ho, Wo = (H - kh) // sh + 1, (W - kw) // sw + 1
            steps.append(dict(op="maxpool", C=C, H=H, W=W, Ho=Ho, Wo=Wo, k=(kh, kw), s=(sh, sw)))
            chw = (C, Ho, Wo)
        elif c["op"] == "conv":
            C, H, W = chw
            co, ci, kh, kw = c["W"].shape
            (sh, sw), (ph, pw) = c["stride"], c["padding"]
            Ho, Wo = (H + 2 * ph - kh) // sh + 1, (W + 2 * pw - kw) // sw + 1
            steps.append(dict(op="conv", C=C, H=H, W=W, Co=co, Ho=Ho, Wo=Wo, k=(kh, kw), s=(sh, sw), p=(ph, pw),
                              Wc=c["W"].transpose(0, 2, 3, 1).reshape(co, -1), L=c))
            chw = (co, Ho, Wo)
        elif c["op"] == "linear":
            Wl = c["W"]
            if flat_from is not None and flat_from[1] * flat_from[2] > 1:    # PyTorch flattens C,H,W; C holds H,W,C
                C, H, W = flat_from
                Wl = Wl.reshape(len(Wl), C, H, W).transpose(0, 2, 3, 1).reshape(len(Wl), -1)
            flat_from = None
            steps.append(dict(op="linear", n_in=Wl.shape[1], n_out=Wl.shape[0], Wc=Wl, L=c))
            chw = (Wl.shape[0], 1, 1)
    return steps


def _out_size(s):
    return s["n_out"] if s["op"] == "linear" else (s["Co"] * s["Ho"] * s["Wo"] if s["op"] == "conv"
                                                   else s["C"] * s["Ho"] * s["Wo"])


def _lut_c(bits):
    """INT2: LB_LUT[byte] = its four codes as int8, in RAM (1 KB: a flash table competes with the weights for the
    cache); INT4 decodes with two shifts"""
    if bits != 2:
        return ""
    per = 4
    rows = [[((b >> (k * bits)) & ((1 << bits) - 1)) - ((1 << bits) if (b >> (k * bits + bits - 1)) & 1 else 0)
             for k in range(per)] for b in range(256)]
    return (f"static int8_t LB_LUT[256][{per}] __attribute__((aligned(4))) = {{\n"
            + "".join("    {" + ", ".join(map(str, r)) + "},\n" for r in rows) + "};\n\n")


def write_header(layers, in_shape, path, bits):
    steps = plan(layers, in_shape)
    n_in = int(np.prod(in_shape))
    widest = max([n_in] + [_out_size(s) for s in steps])
    flash, names = 0, []
    with open(path, "w") as f:
        f.write(f"/* INT{bits} network for the board, written by tools/lowbit/export_c.py: ESP-NN kernels "
                "(multiply-accumulate), int8 activations NHWC. */\n#ifndef SYMBEX_INT_MODEL_H\n#define SYMBEX_INT_MODEL_H\n"
                "/* SYMBEX_NEEDS_ESP_NN: chip/measure.py adds the ESP-NN component for this header */\n"
                "#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n#include \"esp_nn.h\"\n"
                "#ifdef SYMBEX_ROM\n#define LOWBIT_WEIGHTS_RAM 1\n#else\n#define SYMBEX_ROM const\n#endif\n"
                "#ifndef SYMBEX_MARK\n#define SYMBEX_MARK(k) ((void)0)\n#endif\n"
                f"#define LOWBIT_BITS {bits}\n#define LOWBIT_SCRATCH {SCRATCH}\n"
                "/* packed INT4 / INT2 codes stay in flash in both modes (RAM mode: unpacked copies in RAM) */\n"
                "#if LOWBIT_BITS == 8\n#define LOWBIT_WROM SYMBEX_ROM\n#else\n#define LOWBIT_WROM const\n#endif\n\n")
        chunk_max, tmp_max = 1, 1
        for i, s in enumerate(steps):
            if s["op"] == "maxpool":
                continue
            L, Wc = s["L"], s["Wc"]
            rows = [_pack(r, bits) for r in Wc]
            row_bytes = len(rows[0])
            _c_array(f, "uint8_t", f"W{i}", np.concatenate(rows), " __attribute__((aligned(16)))", "LOWBIT_WROM")
            _c_array(f, "int32_t", f"B{i}", L["bias"])
            f.write(f"static int32_t M{i}[{len(L['mult'])}] = {{{', '.join(str(int(v)) for v in L['mult'])}}};\n"
                    f"static int32_t S{i}[{len(L['shift'])}] = {{{', '.join(str(int(v)) for v in L['shift'])}}};\n")
            flash += len(rows) * row_bytes + 4 * len(L["bias"]) + 8 * len(L["mult"])
            s["row_len"], s["row_bytes"] = Wc.shape[1], row_bytes
            s["chunk"] = len(Wc) if bits == 8 else max(1, min(len(Wc), SCRATCH // Wc.shape[1]))
            f.write(f"#ifdef LOWBIT_WEIGHTS_RAM\n#define LB_CHUNK{i} {len(Wc)}     /* unpacked once: one kernel call */\n"
                    f"#else\n#define LB_CHUNK{i} {s['chunk']}\n#endif\n")
            if bits != 8:                                # INT8 is read in place: no unpacking buffer
                chunk_max = max(chunk_max, s["chunk"] * Wc.shape[1])
            if s["op"] == "conv":
                tmp_max = max(tmp_max, s["Ho"] * s["Wo"] * s["chunk"])
        f.write(f"\nstatic int8_t LB_ACT[2][{widest}] __attribute__((aligned(16)));\n"
                f"static int8_t LB_W[{chunk_max}] __attribute__((aligned(16)));\n"
                f"static int8_t LB_TMP[{tmp_max}] __attribute__((aligned(16)));\n"
                "static int lb_ready;\n#ifdef LOWBIT_WEIGHTS_RAM\n"
                f"static int8_t* LB_RAM[{len(steps)}];\n#endif\n\n"
                + _lut_c(bits) +
                "static void lb_unpack(const uint8_t* src, int n, int8_t* dst) {     /* n codes */\n"
                "#if LOWBIT_BITS == 8\n    memcpy(dst, src, (size_t)n);\n#elif LOWBIT_BITS == 4\n"
                "    for (int i = 0; i < n; i += 2) { uint8_t b = src[i >> 1]; dst[i] = (int8_t)(b << 4) >> 4;\n"
                "        if (i + 1 < n) dst[i + 1] = (int8_t)b >> 4; }\n#else\n"
                "    const int full = n / 4;                      /* four codes per byte: one table entry */\n"
                "    for (int k = 0; k < full; k++) memcpy(dst + 4 * k, LB_LUT[src[k]], 4);\n"
                "    for (int i = 4 * full; i < n; i++) dst[i] = LB_LUT[src[full]][i - 4 * full];\n#endif\n}\n\n"
                "/* weights of output channels [o0, o0 + n) of a layer as int8 */\n"
                "static const int8_t* lb_rows(int step, const uint8_t* w, int row_len, int row_bytes, int o0, int n) {\n"
                "#ifdef LOWBIT_WEIGHTS_RAM\n    return LB_RAM[step] + (size_t)o0 * row_len;\n#else\n"
                "#if LOWBIT_BITS == 8\n    (void)step; (void)row_bytes; return (const int8_t*)w + (size_t)o0 * row_len;\n#else\n"
                "    (void)step;\n    for (int r = 0; r < n; r++) lb_unpack(w + (size_t)(o0 + r) * row_bytes, row_len, LB_W + r * row_len);\n"
                "    return LB_W;\n#endif\n#endif\n}\n\n")
        # one-time setup: conv scratch, RAM copies of the weights
        f.write("static void lb_setup(void) {\n    int scratch = 0;\n")
        for i, s in enumerate(steps):
            if s["op"] == "conv":
                f.write(f"    {{ data_dims_t in = {{{s['W']}, {s['H']}, {s['C']}, 1}}, fl = {{{s['k'][1]}, {s['k'][0]}, {s['C']}, 1}},\n"
                        f"                  out = {{{s['Wo']}, {s['Ho']}, LB_CHUNK{i}, 1}};\n"
                        f"      conv_params_t p = {{0, 0, {{{s['s'][1]}, {s['s'][0]}}}, {{{s['p'][1]}, {s['p'][0]}}}, {{1, 1}}, {{-128, 127}}}};\n"
                        "      int k = esp_nn_get_conv_scratch_size(&in, &fl, &out, &p); if (k > scratch) scratch = k; }\n")
        f.write("    if (scratch > 0) esp_nn_set_conv_scratch_buf(malloc((size_t)scratch));\n#ifdef LOWBIT_WEIGHTS_RAM\n")
        for i, s in enumerate(steps):
            if s["op"] != "maxpool":
                n_out = s["n_out"] if s["op"] == "linear" else s["Co"]
                f.write(f"#if LOWBIT_BITS == 8\n    LB_RAM[{i}] = (int8_t*)W{i};                 /* already int8, in RAM */\n#else\n"
                        f"    LB_RAM[{i}] = (int8_t*)aligned_alloc(16, ((size_t){n_out} * {s['row_len']} + 15) & ~(size_t)15);\n"
                        f"    for (int r = 0; r < {n_out}; r++) lb_unpack(W{i} + (size_t)r * {s['row_bytes']}, {s['row_len']}, "
                        f"LB_RAM[{i}] + (size_t)r * {s['row_len']});\n#endif\n")
        f.write("#endif\n    lb_ready = 1;\n}\n\n")
        # the steps
        for i, s in enumerate(steps):
            f.write(f"static void lb_step{i}(const int8_t* x, int8_t* y) {{\n")
            if s["op"] == "maxpool":
                f.write(f"    esp_nn_max_pool_s8(x, {s['W']}, {s['H']}, y, {s['Wo']}, {s['Ho']}, {s['s'][1]}, {s['s'][0]}, "
                        f"{s['k'][1]}, {s['k'][0]}, 0, 0, -128, 127, {s['C']});\n}}\n")
                names.append(f"maxpool {s['k'][0]}x{s['k'][1]}: {s['C']}x{s['H']}x{s['W']} -> {s['C']}x{s['Ho']}x{s['Wo']}")
                continue
            L = s["L"]
            zin, zout = -int(L["zero_in"]), int(L["zero_out"])
            if s["op"] == "linear":
                f.write(f"    for (int o0 = 0; o0 < {s['n_out']}; o0 += LB_CHUNK{i}) {{\n"
                        f"        int n = {s['n_out']} - o0 < LB_CHUNK{i} ? {s['n_out']} - o0 : LB_CHUNK{i};\n"
                        f"        const int8_t* w = lb_rows({i}, W{i}, {s['row_len']}, {s['row_bytes']}, o0, n);\n"
                        f"        esp_nn_fully_connected_per_ch_s8(x, {zin}, {s['n_in']}, w, 0, B{i} + o0, y + o0, (uint16_t)n, "
                        f"{zout}, S{i} + o0, M{i} + o0, -128, 127);\n    }}\n}}\n")
                names.append(f"linear: {s['n_in']} -> {s['n_out']}, INT{bits}")
            else:
                hw = s["Ho"] * s["Wo"]
                f.write(f"    for (int o0 = 0; o0 < {s['Co']}; o0 += LB_CHUNK{i}) {{\n"
                        f"        int n = {s['Co']} - o0 < LB_CHUNK{i} ? {s['Co']} - o0 : LB_CHUNK{i};\n"
                        f"        const int8_t* w = lb_rows({i}, W{i}, {s['row_len']}, {s['row_bytes']}, o0, n);\n"
                        f"        data_dims_t in = {{{s['W']}, {s['H']}, {s['C']}, 1}}, fl = {{{s['k'][1]}, {s['k'][0]}, {s['C']}, 1}},\n"
                        f"                    out = {{{s['Wo']}, {s['Ho']}, n, 1}};\n"
                        f"        conv_params_t p = {{{zin}, {zout}, {{{s['s'][1]}, {s['s'][0]}}}, {{{s['p'][1]}, {s['p'][0]}}}, {{1, 1}}, {{-128, 127}}}};\n"
                        f"        quant_data_t q = {{S{i} + o0, M{i} + o0}};\n"
                        f"        int8_t* dst = n == {s['Co']} ? y : LB_TMP;\n"
                        f"        esp_nn_conv_s8(&in, x, &fl, w, B{i} + o0, &out, dst, &p, &q);\n"
                        f"        if (dst != y) for (int j = 0; j < {hw}; j++) memcpy(y + j * {s['Co']} + o0, LB_TMP + j * n, (size_t)n);\n"
                        "    }\n}\n")
                names.append(f"conv {s['k'][0]}x{s['k'][1]}: {s['C']}x{s['H']}x{s['W']} -> {s['Co']}x{s['Ho']}x{s['Wo']}, INT{bits}")
        n_cls = _out_size(steps[-1])
        f.write(f"\n#define SYMBEX_MLP_N_INPUTS {n_in}\n#define SYMBEX_MLP_N_CLASSES {n_cls}\n"
                f"#define SYMBEX_MLP_N_STEPS {len(steps)}\n#define SYMBEX_MLP_BYTES_BUFFERS {2 * widest + chunk_max + tmp_max}u\n"
                "static const char* const symbex_mlp_steps[SYMBEX_MLP_N_STEPS] = {\n"
                + "".join(f"    \"{n}\",\n" for n in names) + "};\n\n"
                "static inline int symbex_mlp_infer(const uint8_t* x_bytes) {\n"
                "    if (!lb_ready) lb_setup();\n    const int8_t* x = (const int8_t*)x_bytes;\n    SYMBEX_MARK(0);\n")
        for i in range(len(steps)):
            f.write(f"    lb_step{i}(x, LB_ACT[{i % 2}]);\n    x = LB_ACT[{i % 2}];\n    SYMBEX_MARK({i + 1});\n")
        f.write("    int best = 0;\n    for (int c = 1; c < SYMBEX_MLP_N_CLASSES; c++) if (x[c] > x[best]) best = c;\n"
                "    return best;\n}\n#endif\n")
    return dict(flash_bytes=flash, ram_buffers=2 * widest + chunk_max + tmp_max, steps=names)


def input_bytes(layers, x_float):
    """the int8 input of every image as the chip receives it: NHWC, as bytes"""
    xq = quantize_input(layers, x_float)
    if xq.ndim == 3:
        xq = xq[:, :, None, :]
    if xq.ndim == 4:
        xq = xq.transpose(0, 2, 3, 1)
    return xq.reshape(len(xq), -1).astype(np.int8).view(np.uint8)


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


def check_c(header, x_bytes, expected, extra_flags=()):
    """compile the header with ESP-NN's reference kernels (gcc) and compare with the simulation: (equal, message)"""
    esp_nn = find_esp_nn()
    if shutil.which("gcc") is None or esp_nn is None:
        return None, "gcc or the ESP-NN sources not found (set ESP_NN_PATH): the PC check was skipped"
    srcs = [os.path.join(esp_nn, "src", p) for p in ("convolution/esp_nn_conv_ansi.c",
                                                      "fully_connected/esp_nn_fully_connected_ansi.c",
                                                      "pooling/esp_nn_max_pool_ansi.c")]
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header, os.path.join(d, "model_symbex.h"))
        open(os.path.join(d, "main.c"), "w").write(_TEST_MAIN)
        exe = os.path.join(d, "check")
        r = subprocess.run(["gcc", "-O2", "-std=gnu11", *extra_flags, "-I", d, "-I", os.path.join(esp_nn, "include"),
                            "-I", os.path.join(esp_nn, "src", "common"), "-o", exe, os.path.join(d, "main.c"), *srcs],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("[X] lowbit: the C did not compile:\n" + r.stderr[-3000:])
        np.ascontiguousarray(x_bytes).tofile(os.path.join(d, "x.u8"))
        r = subprocess.run([exe, os.path.join(d, "x.u8"), str(len(x_bytes))], capture_output=True, text=True)
        got = np.array([int(v) for v in r.stdout.split()])
    equal = int((got == expected).sum()) if len(got) == len(expected) else 0
    return equal, f"C (ESP-NN reference kernels) == integer simulation: {equal}/{len(expected)}"


def export_folder(layers, x_float, y, out_dir, meta=None):
    """header + validation files + export.json in out_dir; the C checked on the PC (weights in flash and in RAM)"""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from compiler_core.export.dense import save_validation
    os.makedirs(out_dir, exist_ok=True)
    x_float = np.asarray(x_float)
    bits = next(c["bits"] for c in layers if "bits" in c)
    sim = run_integer(layers, quantize_input(layers, x_float)).reshape(len(x_float), -1).argmax(1)
    xb = input_bytes(layers, x_float)
    header = os.path.join(out_dir, "model_symbex.h")
    info = write_header(layers, x_float.shape[1:], header, bits)
    eq_flash, msg = check_c(header, xb, sim)
    eq_ram, msg_ram = check_c(header, xb, sim, ["-DSYMBEX_ROM="])
    save_validation(out_dir, xb, y, sim)
    res = dict(meta or {}, bits=bits, n=len(xb), acc_integer=100 * float((sim == np.asarray(y)).mean()),
               c_exactness=None if eq_flash is None else 100 * eq_flash / len(xb),
               c_exactness_ram=None if eq_ram is None else 100 * eq_ram / len(xb),
               arithmetic=f"INT{bits}: ESP-NN multiply-accumulate (int8 activations, per-channel fixed-point "
                          "requantization)", header=header, **info)
    json.dump(res, open(os.path.join(out_dir, "export.json"), "w"), indent=1)
    ok = eq_flash == len(xb) and eq_ram == len(xb)
    print(f"    INT{bits}: integer accuracy {res['acc_integer']:.2f}% ({len(xb)} images) | {info['flash_bytes'] / 1024:.1f} KB "
          f"of weights | {msg} (RAM variant: {eq_ram}) " + ("OK" if ok else "[X]") + f"\n    -> {out_dir}/")
    return res


# ---------------------------------------------------------------------------------------------------
# On the board: tools/chip_test.py (the CDT's chip stage), the same firmware as every other model
# ---------------------------------------------------------------------------------------------------
CHIP_TEST = os.path.join(REPO, "tools", "chip_test.py")


def _find(d, names):
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return None


def chip_result(folder, board, weights):
    """the newest chip_test result of an export folder: latency (ms), board == simulation (%), file"""
    files = sorted(glob.glob(os.path.join(folder, f"chip_{board}_*_{weights}_*.json")), key=os.path.getmtime)
    if not files:
        return None
    d = json.load(open(files[-1]))
    return dict(file=files[-1], latency_ms=_find(d, ["latency_median_ms", "median_ms", "latency_ms", "ms"]),
                equal=_find(d, ["equal_simulation", "equal_to_simulation", "equal"]),
                accuracy=_find(d, ["accuracy", "acc", "accuracy_pct"]))


def measure(folder, board, port, usb=False, weights="flash"):
    """runs chip_test.py on an export folder (no INT8 side by side: the folder IS an integer model)"""
    cmd = [sys.executable, CHIP_TEST, os.path.abspath(folder), "--board", board, "--port", port, "--weights", weights,
           "--no-int8"] + (["--usb"] if usb else [])
    if subprocess.run(cmd, cwd=os.path.dirname(CHIP_TEST)).returncode != 0:
        raise SystemExit(f"[X] chip_test.py failed on {folder} (the output above says why)")
    return chip_result(folder, board, weights)


def chip_study(model_dir, x_float, y, bits, tag="", out_root="runs/lowbit_chip", board=None, port=None, usb=False,
               weights="flash", seed=42):
    """the seed-`seed` QAT model of every bit width (study.run saved it in model_dir) exported to C and, with a board,
    measured on it (once: an existing result is reused). Returns the table rows."""
    rows = []
    for b in bits:
        model = os.path.join(model_dir, f"int{b}_qat{tag}_s{seed}.npy")
        if not os.path.exists(model):
            print(f"[!] {model}: missing (run the study first): INT{b} skipped")
            continue
        folder = os.path.join(out_root, f"int{b}_qat{tag}_s{seed}")
        if not os.path.exists(os.path.join(folder, "export.json")):
            export_folder(list(np.load(model, allow_pickle=True)), x_float, y, folder,
                          meta=dict(model=os.path.relpath(model), seed=seed))
        exp = json.load(open(os.path.join(folder, "export.json")))
        chip = chip_result(folder, board, weights) if board else None
        if board and chip is None:
            print(f"\n[*] INT{b} on the {board} (weights in {weights}): {folder}")
            chip = measure(folder, board, port, usb, weights)
        rows.append(dict(model=f"INT{b} QAT s{seed}", bits=b, weight_kb=exp["flash_bytes"] / 1024,
                         acc_integer=exp["acc_integer"], c_exact=exp["c_exactness"], folder=folder,
                         latency_ms=(chip or {}).get("latency_ms"), equal=(chip or {}).get("equal"),
                         file=(chip or {}).get("file")))
    print(f"\n    {'model':<16}{'weights KB':>11}{'int acc':>9}{'PC C = sim':>12}{'latency ms':>12}{'board = sim':>13}")
    for r in rows:
        lat = f"{r['latency_ms']:.3f}" if r["latency_ms"] else "-"
        eq = f"{r['equal']:.0f}%" if isinstance(r["equal"], (int, float)) else "-"
        ce = f"{r['c_exact']:.0f}%" if r["c_exact"] is not None else "skipped"
        print(f"    {r['model']:<16}{r['weight_kb']:>11.1f}{r['acc_integer']:>8.2f}%{ce:>12}{lat:>12}{eq:>13}")
    return rows
