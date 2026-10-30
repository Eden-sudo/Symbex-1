"""
har_gvsoc.py -- whole HAR networks through both libraries: ultracompactBNN (Daghero et al., CF 2021) and lib_symbex.

    python har_gvsoc.py --repo ~/Proyectos/ultracompactBNN --pc runs/max_int8_e100_s42 runs/max_binary_e100_s42
        (this PC, gcc: both C programs on every exported validation window, against their integer simulations)
    python3 har_gvsoc.py --repo /work/ultracompactBNN runs/max_int8_e100_s42 runs/max_binary_e100_s42
        (inside the symbex-pulp container: GVSoC, RI5CY, PULP GCC -O3 -flto: instructions and cycles per layer)

Run from benchmarks/conv1d/har_ultracompact/ after har_export.py on both runs. Each library runs its own network, as
designed (same architecture, same training recipe):
    ultracompactBNN   the run with first = int8: their kernels unchanged -- 8-bit-to-1-bit convolution (int8 input x
                      int8 weights, sdotsp4), 1-bit convolutions (fused max pool where the architecture pools: their
                      *_pooling kernels), binarized linear layer (int32 scores); their layout: MSB-first bit streams,
                      time-major, int16 thresholds on the matches of the 1-bit layers
    lib_symbex        the run with first = binary: model_symbex.h of har_export.py (symbex_bool1d layers, the
                      classifier of symbex_bool.h), bits counted with p.cnt (-DSYMBEX_POPCOUNT_FORCE_NATIVE)
Their kernels OR into their outputs, so every output buffer is cleared inside the measured time (lib_symbex clears
its own). Both report the class (the first largest score). Results: results/har_gvsoc_<arch>_s<seed>.json.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Symbex1_PIO", "lib", "lib_symbex"))
LENGTH, AXES = 151, 3
KERNELS = {                                     # template folder -> sources used
    "template_convolution1d_8bit_to1bit": ["conv1bit_8bit.c", "matmul4x2.c", "matmul4x1.c"],
    "template_convolution1d_1bit": ["convolution1bit_NxM.c", "xnorpop2x2g.c", "xnorpop4x1g.c"],
    "template_convolutionMaxpool1d_1bit": ["convolution1bit_NxM_pooling.c", "xnorpop2x2_pooling.c", "xnorpop4x1_pooling.c"],
    "template_binarized_linear": ["linear.c", "xnorpop4x1_linear.c"],
}
U8_PULP = os.path.join(HERE, "pulp", "symbex_u8_pulp.c")   # lib_symbex's first layer with PULP's dot products
N_GVSOC = 4                                     # windows measured on the simulator (mean); all of them checked
# Their 8-bit first layer keeps its thresholds in int16, but int8 inputs x int8 weights over 15 x 3 taps reach
# +-58,000 in the trained networks (max). The ONE change to their code: those thresholds become int32 (in the copied
# sources of that template only). On RV32 an int32 load costs the same instruction as an int16 load.
THRESHOLDS_32 = ("int16_t *thresholds", "int32_t *thresholds")


def msb_words(bits, extra=1):
    """0/1 vector -> uint32 words, MSB first, zero padded, plus `extra` spare words (their kernels read past the end)"""
    n = -(-len(bits) // 32) + extra
    pad = np.zeros(n * 32, dtype=np.uint8)
    pad[:len(bits)] = bits
    return np.packbits(pad, bitorder="big").view(">u4").astype(np.uint32)


def load_net(run):
    d = np.load(os.path.join(run, "net.npz"))
    layers, i = [], 0
    while f"w{i}" in d:
        layers.append(dict(w=d[f"w{i}"], thr=d[f"thr{i}"], k=int(d[f"k{i}"]), stride=int(d[f"stride{i}"]),
                           pool=int(d[f"pool{i}"])))
        i += 1
    return dict(first=str(d["first"]), layers=layers, length=int(d["length"]), fc_w=d["fc_w"], fc_scale=d["fc_scale"],
                fc_bias=d["fc_bias"])


def windows(run, n=None):
    """the exported validation windows (int8, N x 151 x 3) and the classes of the integer simulation"""
    x = np.fromfile(os.path.join(run, "validation_x.u8"), dtype=np.uint8).reshape(-1, LENGTH, AXES)
    sim = np.fromfile(os.path.join(run, "validation_sim.u8"), dtype=np.uint8).astype(np.int64)
    x = (x.astype(np.int16) - 128).astype(np.int8)
    return (x, sim) if n is None else (x[:n], sim[:n])


# ---------------------------------------------------------------------------------------------------
# ultracompactBNN: the network in their layout, and the program that chains their kernels
# ---------------------------------------------------------------------------------------------------
def ucbnn_data(net):
    """data_net.h: every layer's arrays and sizes in their layout; returns (text, layer list for the program)"""
    if net["first"] != "int8":
        raise ValueError("the ultracompactBNN side runs the network with int8 first-layer weights (first = int8)")
    out, plan = [], []
    length, ch = LENGTH, AXES
    arr = lambda v: "{" + ", ".join(str(int(a)) for a in np.ravel(v)) + "}"
    hexa = lambda v: "{" + ", ".join(f"0x{int(a) & 0xFFFFFFFF:08X}u" for a in np.ravel(v)) + "}"
    for i, c in enumerate(net["layers"]):
        w = c["w"].astype(np.int64)                                     # (O, C, k)
        o, k = w.shape[0], c["k"]
        if c["stride"] != 1:
            raise ValueError("their kernels with pooling run stride 1")
        l_conv = length - k + 1
        l_out = l_conv // c["pool"]
        n = k * ch
        if i == 0:                                                      # int8 x int8: threshold on the raw sum
            thr = c["thr"].astype(np.int64)            # int32 here: see THRESHOLDS_32 (their int16 overflows)
            out.append(f"static const int8_t W{i}[{o * n + 4}] = {arr(np.concatenate([w.transpose(0, 2, 1).ravel(), [0] * 4]))};")
            out.append(f"static const int32_t T{i}[{o}] = {arr(thr)};")
        else:                                                           # bits: threshold on the matches m, raw = 2m - n
            thr = -(-(c["thr"].astype(np.int64) + n) // 2)
            thr = np.clip(thr, -32768, 32767)
            out.append(f"static const uint32_t W{i}[] = {hexa(msb_words((w.transpose(0, 2, 1) > 0).ravel().astype(np.uint8)))};")
            out.append(f"static const int16_t T{i}[{o}] = {arr(thr)};")
        out.append(f"static uint32_t M{i}[{-(-l_out * o // 32) + 2}];")
        plan.append(dict(i=i, cin=ch, cout=o, k=k, lin=length, lconv=l_conv, lout=l_out, pool=c["pool"]))
        length, ch = l_out, o
    fcw = (net["fc_w"] > 0).astype(np.uint8)                            # (classes, L * C), time-major
    n_cls, n_in = fcw.shape
    a, b = 2 * net["fc_scale"].astype(np.int64), net["fc_bias"].astype(np.int64) - net["fc_scale"].astype(np.int64) * n_in
    worst = np.maximum(np.abs(b), np.abs(a * n_in + b)).max()
    if worst >= 2 ** 31 or np.abs(a).max() >= 2 ** 31:
        raise ValueError("the classifier scores do not fit their int32 output")
    out.append(f"static const uint32_t FC_W[] = {hexa(msb_words(fcw.ravel()))};")
    out.append(f"static const int32_t FC_T[{2 * n_cls}] = {arr(np.stack([a, b], 1))};")
    out.append(f"static int32_t FC_OUT[{n_cls}];")
    out.append(f"#define N_CLASSES {n_cls}\n#define FC_IN {n_in}\n#define N_LAYERS {len(plan)}")
    return "\n".join(out) + "\n", plan


def ucbnn_infer(plan):
    """int ucbnn_infer(const int8_t* x): their kernels, layer after layer, MARK(k) after each"""
    s = ["static int ucbnn_infer(const int8_t* x) {", "    MARK(0);"]
    for p in plan:
        i, words = p["i"], -(-p["lout"] * p["cout"] // 32) + 2
        s.append(f"    for (int j = 0; j < {words}; j++) M{i}[j] = 0;          /* their kernels OR into the output */")
        src = "x" if i == 0 else f"(const uint32_t*)M{i - 1}"
        if i == 0 and p["pool"] == 1:
            s.append(f"    conv_8x1bits_1D_nopad_nodilation(x, {p['lin']}, {p['cin']}, W0, {p['cout']}, {p['k']}, 1, M0, "
                     f"{p['lconv']}, T0);")
        elif i == 0:
            raise ValueError("a pooled first layer is not in Table 1 (their *_8bit_to1bit pooling kernel is not wired)")
        elif p["pool"] == 1:
            s.append(f"    conv_bin_1D_nopad_nodilation({src}, {p['lin']}, {p['cin']}, W{i}, {p['cout']}, {p['k']}, 1, "
                     f"M{i}, {p['lconv']}, T{i});")
        else:                                  # dim_out = the steps the pool uses (their leftover step would land past it)
            s.append(f"    conv_bin_1D_nopad_nodilation_pooling({src}, {p['lin']}, {p['cin']}, W{i}, {p['cout']}, {p['k']}, 1, "
                     f"M{i}, {p['lout'] * p['pool']}, T{i}, {p['lout']}, {p['pool']}, {p['pool']});")
        s.append(f"    MARK({i + 1});")
    last = plan[-1]["i"]
    s += [f"    linear((const uint32_t*)M{last}, FC_IN, FC_W, N_CLASSES, FC_OUT, FC_T);",
          "    int best = 0;",
          "    for (int c = 1; c < N_CLASSES; c++) if (FC_OUT[c] > FC_OUT[best]) best = c;",
          f"    MARK({len(plan) + 1});", "    return best;", "}"]
    return "\n".join(s) + "\n"


KERNELS_H = """/* every ultracompactBNN kernel this program chains (their kernels.h files, guards renamed) */
#ifndef UCBNN_ALL_KERNELS_H
#define UCBNN_ALL_KERNELS_H
#include <stdint.h>
""" + "".join(f'#include "kernels_{i}.h"\n' for i in range(len(KERNELS))) + "#endif\n"

PC_RT = """/* this PC: the PULP builtins their kernels use, in plain C */
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
typedef signed char v4s __attribute__((vector_size(4)));
static inline int pc_sdotsp4(v4s a, v4s b, int c) { return c + a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }
#define __builtin_pulp_sdotsp4(a, b, c) pc_sdotsp4(a, b, c)
#define __builtin_pulp_cnt(x) __builtin_popcount(x)
#define RT_FC_SHARED_DATA
"""
GVSOC_RT = """#include "pmsis.h"
#ifndef RT_FC_SHARED_DATA
#define RT_FC_SHARED_DATA
#endif
typedef signed char v4s __attribute__((vector_size(4)));
"""

PC_MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define MARK(k)
#include "kernels.h"
#include "data_net.h"
%(INFER)s
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    int n = atoi(argv[2]);
    static int8_t x[%(N_IN)d + 4];
    for (int i = 0; i < n; i++) {
        if (fread(x, 1, %(N_IN)d, f) != %(N_IN)d) return 1;
        printf("%%d\n", ucbnn_infer(x));
    }
    return 0;
}
"""

GVSOC_MAIN = r"""
/* whole-network measurement on GVSoC: HOTTING warm-up runs, then the mean of REPEAT runs per window (their stats.h);
   instructions and cycles of every layer from the counters read after it */
#include "pmsis.h"
#include <stdint.h>
#define HOTTING 2
#define REPEAT 3
static unsigned long mark_i[16], mark_c[16];
#define MARK(k) do { mark_i[k] = pi_perf_read(PI_PERF_INSTR); mark_c[k] = pi_perf_read(PI_PERF_CYCLES); } while (0)
%(INCLUDES)s
%(WINDOWS)s
int main(void) {
    unsigned long li[16] = {0}, lc[16] = {0};
    int errors = 0;
    pi_perf_conf((1 << PI_PERF_CYCLES) | (1 << PI_PERF_INSTR));
    for (int w = 0; w < N_WIN; w++) {
        int c = -1;
        for (int k = 0; k < HOTTING + REPEAT; k++) {
            pi_perf_reset();
            pi_perf_start();
            c = %(CALL)s;
            pi_perf_stop();
            if (k >= HOTTING)
                for (int s = 1; s <= N_STEPS; s++) { li[s] += mark_i[s] - mark_i[s - 1]; lc[s] += mark_c[s] - mark_c[s - 1]; }
        }
        errors += c != EXPECTED[w];
    }
    for (int s = 1; s <= N_STEPS; s++)
        printf("STEP %%d instr=%%lu cycles=%%lu\n", s, li[s] / (REPEAT * N_WIN), lc[s] / (REPEAT * N_WIN));
    printf("NET errors=%%d windows=%%d\n", errors, N_WIN);
    return 0;
}
"""


def ucbnn_project(work, repo, net, rt):
    if os.path.exists(work):
        shutil.rmtree(work)
    for d in ("src", "include", "rt"):
        os.makedirs(os.path.join(work, d))
    open(os.path.join(work, "rt", "rt_api.h"), "w").write(rt)
    srcs = []
    for i, (tpl, files) in enumerate(KERNELS.items()):
        base = os.path.join(repo, tpl, "gvsoc")
        fix = (lambda t: t.replace(*THRESHOLDS_32)) if tpl == "template_convolution1d_8bit_to1bit" else (lambda t: t)
        h = open(os.path.join(base, "include", "kernels.h")).read().replace("BIN_CONVOLUTION_KERNELS_H", f"UCBNN_K{i}_H")
        open(os.path.join(work, "include", f"kernels_{i}.h"), "w").write(fix(h))
        for f in files:
            open(os.path.join(work, "src", f), "w").write(fix(open(os.path.join(base, "src", f)).read()))
            srcs.append(f"src/{f}")
    open(os.path.join(work, "include", "kernels.h"), "w").write(KERNELS_H)
    data, plan = ucbnn_data(net)
    open(os.path.join(work, "src", "data_net.h"), "w").write(data)
    return srcs, plan


def c_windows(x, expected):
    arr = lambda v: "{" + ", ".join(str(int(a)) for a in np.ravel(v)) + "}"
    rows = ",\n    ".join(arr(w) for w in x)
    return (f"#define N_WIN {len(x)}\n"
            f"static const %(T)s WINDOWS[N_WIN][{LENGTH * AXES + 4}] __attribute__((aligned(4))) = {{\n    {rows}\n}};\n"
            f"static const int EXPECTED[N_WIN] = {arr(expected)};\n")


# ---------------------------------------------------------------------------------------------------
# this PC: both C programs on every exported window
# ---------------------------------------------------------------------------------------------------
def pc_check(repo, run_int8, run_bin, work):
    import tempfile
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools")))
    from compiler_core.export.bool1d import validate_c
    res = {}
    x8, sim8 = windows(run_int8)
    srcs, plan = ucbnn_project(work, repo, load_net(run_int8), PC_RT)
    open(os.path.join(work, "src", "main.c"), "w").write(PC_MAIN % dict(INFER=ucbnn_infer(plan), N_IN=LENGTH * AXES))
    exe = os.path.join(work, "net")
    r = subprocess.run(["gcc", "-O2", "-w", "-std=gnu11", "-fno-strict-aliasing", "-I", work, "-I",
                        os.path.join(work, "include"), "-I", os.path.join(work, "src"), "-o", exe, "src/main.c"] + srcs,
                       cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("[X] their kernels did not compile with gcc:\n" + r.stderr[-3000:])
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "x.i8")
        x8.tofile(data)
        out = subprocess.run([exe, data, str(len(x8))], capture_output=True, text=True).stdout.split()
    pred = np.array([int(v) for v in out])
    res["ucbnn"] = dict(equal=int((pred == sim8).sum()), total=len(sim8))
    xb, simb = windows(run_bin)
    x_u8 = (xb.astype(np.int16) + 128).astype(np.uint8).reshape(len(xb), -1)
    eq, tot, msg = validate_c(os.path.join(run_bin, "model_symbex.h"), LIB, x_u8, simb, native_popcount=True)
    res["symbex"] = dict(equal=eq, total=tot)
    eq, tot, msg = validate_c(os.path.join(run_bin, "model_symbex.h"), LIB, x_u8, simb, native_popcount=True,
                              extra_sources=[U8_PULP], defines=["SYMBEX_U8_EXTERNAL"])
    res["symbex_pulp_u8"] = dict(equal=eq, total=tot)
    eq, tot, msg = validate_c(os.path.join(run_bin, "model_symbex.h"), LIB, x_u8, simb, native_popcount=True,
                              extra_sources=[U8_PULP], defines=["SYMBEX_BOOL1D_U8_EXTERNAL"])
    res["symbex_pulp_layer"] = dict(equal=eq, total=tot)
    x8_u8 = (x8.astype(np.int16) + 128).astype(np.uint8).reshape(len(x8), -1)
    for name, extra, defs in (("symbex_i8_first", [], []), ("symbex_i8_first_pulp", [U8_PULP], ["SYMBEX_BOOL1D_U8_EXTERNAL"])):
        eq, tot, msg = validate_c(os.path.join(run_int8, "model_symbex.h"), LIB, x8_u8, sim8, native_popcount=True,
                                  extra_sources=extra, defines=defs)
        res[name] = dict(equal=eq, total=tot)
    for name, r_ in res.items():
        ok = r_["equal"] == r_["total"]
        print(f"    {name:<20}: {r_['equal']}/{r_['total']} windows equal to its integer simulation "
              + ("OK" if ok else "[X]"))
    return res


# ---------------------------------------------------------------------------------------------------
# GVSoC (inside the symbex-pulp container)
# ---------------------------------------------------------------------------------------------------
def gvsoc_run(work, cmd_cflags, srcs):
    open(os.path.join(work, "Makefile"), "w").write(
        "APP = test\nAPP_SRCS = src/main.c " + " ".join(srcs) + "\n"
        f"APP_CFLAGS += -O3 -w -flto -I$(CURDIR) -I$(CURDIR)/include -I$(CURDIR)/src {cmd_cflags}\n"
        "APP_LDFLAGS += -flto\ninclude $(RULES_DIR)/pmsis_rules.mk\n")
    r = subprocess.run("make clean all run", shell=True, cwd=work, capture_output=True, text=True)
    steps = [tuple(int(v) for v in m) for m in re.findall(r"STEP (\d+) instr=(\d+) cycles=(\d+)", r.stdout)]
    m = re.search(r"NET errors=(\d+) windows=(\d+)", r.stdout)
    if not m or not steps:
        print((r.stdout + r.stderr)[-3000:])
        sys.exit(f"[X] {work}: no result from the simulator (the output above says why)")
    return dict(errors=int(m.group(1)), windows=int(m.group(2)),
                steps=[dict(instr=i, cycles=c) for _, i, c in steps],
                instr=sum(i for _, i, _ in steps), cycles=sum(c for _, _, c in steps))


def gvsoc(repo, run_int8, run_bin, work, n, cflags=""):
    x8, sim8 = windows(run_int8, n)
    srcs, plan = ucbnn_project(os.path.join(work, "ucbnn"), repo, load_net(run_int8), GVSOC_RT)
    main = GVSOC_MAIN % dict(INCLUDES='#include "kernels.h"\n#include "data_net.h"\n' + ucbnn_infer(plan),
                             WINDOWS=c_windows(x8, sim8) % dict(T="int8_t") + f"#define N_STEPS {len(plan) + 1}\n",
                             CALL="ucbnn_infer(WINDOWS[w])")
    open(os.path.join(work, "ucbnn", "src", "main.c"), "w").write(main)
    uc = gvsoc_run(os.path.join(work, "ucbnn"), "", srcs)
    n_steps = len(load_net(run_bin)["layers"]) + 1
    lib_srcs = [os.path.join(LIB, "src", "layers", "conv", f) for f in ("symbex_bool1d.c", "symbex_bool.c")]
    lib_srcs.append(os.path.join(LIB, "src", "hal", "generic", "symbex_hal_impl.c"))
    flags = f"-I{os.path.join(LIB, 'include')} -DSYMBEX_POPCOUNT_FORCE_NATIVE {cflags}"
    sx = {}
    layer = " -DSYMBEX_BOOL1D_U8_EXTERNAL -DSYMBEX_U8_PULP_DOTP"
    for name, run, extra, more in (("symbex", run_bin, [], ""),      # the generic first layer (SWAR), PULP's dot product
                                   ("symbex_pulp_u8", run_bin, [U8_PULP], " -DSYMBEX_U8_EXTERNAL -DSYMBEX_U8_PULP_DOTP"),
                                   # the whole first layer: 3 steps x 4 channels per pass, p.sdotusp4 on +1 / -1 bytes
                                   ("symbex_pulp_layer", run_bin, [U8_PULP], layer),
                                   # ultracompactBNN's own network (int8 first layer) on lib_symbex: the same kernel
                                   # on its int8 weights, then lib_symbex's binary layers
                                   ("symbex_i8_first", run_int8, [U8_PULP], layer)):
        xb, simb = windows(run, n)
        x_u8 = (xb.astype(np.int16) + 128).astype(np.uint8)
        sw = os.path.join(work, name)
        if os.path.exists(sw):
            shutil.rmtree(sw)
        os.makedirs(os.path.join(sw, "src"))
        shutil.copy(os.path.join(run, "model_symbex.h"), os.path.join(sw, "src", "model_symbex.h"))
        main = GVSOC_MAIN % dict(INCLUDES="#define SYMBEX_MARK(k) MARK(k)\n#include \"model_symbex.h\"\n",
                                 WINDOWS=c_windows(x_u8, simb) % dict(T="uint8_t") + f"#define N_STEPS {n_steps}\n",
                                 CALL="symbex_mlp_infer(WINDOWS[w])")
        open(os.path.join(sw, "src", "main.c"), "w").write(main)
        sx[name] = gvsoc_run(sw, flags + more, lib_srcs + extra)
    return uc, sx, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=2, help="the int8 run (ultracompactBNN) and the binary run (lib_symbex), exported")
    ap.add_argument("--repo", required=True, help="the ultracompactBNN clone")
    ap.add_argument("--pc", action="store_true", help="check both C programs on this PC (gcc) instead of GVSoC")
    ap.add_argument("--windows", type=int, default=N_GVSOC, help=f"windows measured on GVSoC ({N_GVSOC})")
    ap.add_argument("--cflags", default="", help="extra C flags for the lib_symbex programs on GVSoC, e.g. "
                    "-DSYMBEX_BOOL1D_U8_PULP_STEPS=3 (the result file is written the same: keep the default for it)")
    args = ap.parse_args()
    os.chdir(HERE)
    repo = os.path.abspath(os.path.expanduser(args.repo))
    run_int8, run_bin = (os.path.normpath(r) for r in args.runs)
    i8, ib = (json.load(open(os.path.join(r, "train.json"))) for r in (run_int8, run_bin))
    if i8["first"] != "int8" or ib["first"] != "binary" or i8["arch"] != ib["arch"]:
        sys.exit("[X] give the int8 run first and the binary run second, of the same architecture")
    tag = f"{ib['arch']}_s{ib['seed']}"
    if args.pc:
        print(f"\n    {tag}: both libraries' C on this PC (gcc), every exported validation window")
        pc_check(repo, run_int8, run_bin, os.path.join(HERE, "runs", "pc_ucbnn", tag))
        return
    uc, sx, plan = gvsoc(repo, run_int8, run_bin, os.path.join(HERE, "runs", "gvsoc_net", tag), args.windows,
                       args.cflags)
    names = [f"conv {p['cin']}->{p['cout']} k{p['k']}" + (f" + pool {p['pool']}" if p["pool"] > 1 else "") for p in plan]
    names.append("classifier")
    print(f"\n    {tag} on GVSoC (RI5CY, PULP GCC -O3 -flto), mean of {args.windows} windows\n")
    a_, b_, c_, d_ = sx["symbex"], sx["symbex_pulp_u8"], sx["symbex_pulp_layer"], sx["symbex_i8_first"]
    print("    SYMBEX: the generic first layer (SWAR) | +PULP rows: the HAL row functions with p.sdotup4 | +PULP layer:")
    print("    the whole first layer with p.sdotusp4, 3 steps x 4 channels per pass (both in pulp/symbex_u8_pulp.c)")
    print("    int8 1st: ultracompactBNN's own network (int8 first layer) on lib_symbex, the +PULP layer kernel")
    print("    ratio = ultracompactBNN instructions / SYMBEX instructions (> 1: SYMBEX does less work)\n")
    print(f"    {'step':<28}{'ucbnn instr':>12}{'cycles':>10} | {'SYMBEX':>9} | {'+PULP rows':>10} | "
          f"{'+PULP layer':>11}{'cycles':>10} | {'ratio':>6} | {'int8 1st':>9}{'cycles':>10} | {'ratio':>6}")
    print("    " + "-" * 136)
    rows = list(zip(names, uc["steps"], a_["steps"], b_["steps"], c_["steps"], d_["steps"]))
    rows.append(("whole network", uc, a_, b_, c_, d_))
    for i, (name, u, g, p, q, e) in enumerate(rows):
        if i == len(rows) - 1:
            print("    " + "-" * 136)
        print(f"    {name:<28}{u['instr']:>12}{u['cycles']:>10} | {g['instr']:>9} | {p['instr']:>10} | "
              f"{q['instr']:>11}{q['cycles']:>10} | {u['instr'] / max(q['instr'], 1):>5.2f}x | "
              f"{e['instr']:>9}{e['cycles']:>10} | {u['instr'] / max(e['instr'], 1):>5.2f}x")
    print(f"\n    outputs equal to their integer simulations: ultracompactBNN {uc['windows'] - uc['errors']}/{uc['windows']}"
          + "".join(f", {k} {v['windows'] - v['errors']}/{v['windows']}" for k, v in sx.items()))
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"har_gvsoc_{tag}.json")
    json.dump(dict(arch=ib["arch"], seed=ib["seed"], steps=names, ucbnn=uc, symbex=sx["symbex"],
                   symbex_pulp_u8=sx["symbex_pulp_u8"], symbex_pulp_layer=sx["symbex_pulp_layer"],
                   symbex_i8_first=sx["symbex_i8_first"], windows=args.windows,
                   accuracy=dict(ucbnn=i8.get("test"), symbex=ib.get("test"))), open(out, "w"), indent=1)
    print(f"    -> {out}")


if __name__ == "__main__":
    main()
