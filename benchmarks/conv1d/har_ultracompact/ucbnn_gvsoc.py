"""
ucbnn_gvsoc.py -- the binary 1D convolution of ultracompactBNN (Daghero et al., CF 2021) on its own processor: the
RI5CY core simulated by GVSoC, compiled by the PULP GCC. Their kernels are used unchanged.

    (inside the symbex-pulp container, from /work/Symbex1/benchmarks/conv1d/har_ultracompact)
    pip3 install numpy                                         # once per container
    python3 ucbnn_gvsoc.py --repo /work/ultracompactBNN
    python3 ucbnn_gvsoc.py --repo /work/ultracompactBNN --shapes 151x3x8x15 64x32x32x7
    python3 ucbnn_gvsoc.py --repo /work/ultracompactBNN --symbex /work/Symbex1/Symbex1_PIO/lib/lib_symbex

For every layer (L_in x C_in x C_out x K): the layer data in their layout (ucbnn_check.py: checked 60/60 against their
kernel on the PC), their convolution (convolution1bit_NxM.c + xnorpop2x2g.c + xnorpop4x1g.c) and a benchmark program for
the current PULP SDK. Their program used the old SDK (pulp-rt: rt_perf_*); this one does the same with the current API
(pi_perf_*): HOTTING = 2 warm-up runs, then the mean of REPEAT = 3 measured runs, as their stats.h. Their compiler flags:
-O3 -flto. The output is checked against the expected one, word by word. Reported as they do: cycles and MAC/cycle,
MAC = C_in x L_out x C_out x K. Results: results/ucbnn_gvsoc.json.

With --symbex, the SAME layer (the same bits, weights and thresholds) also runs through lib_symbex's 1D convolution
(layers/conv/symbex_bool1d.c) on the same core, compiled the same way, its bits counted with the same p.cnt instruction
(the generic HAL with -DSYMBEX_POPCOUNT_FORCE_NATIVE: the PULP GCC turns __builtin_popcount into p.cnt). Both outputs
are checked; instructions are the cleanest comparison (they do not depend on the memory model of the simulated chip),
cycles are reported too.
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
sys.path.insert(0, HERE)
from ucbnn_check import data_h, layer                                                          # noqa: E402

KERNELS = ("convolution1bit_NxM.c", "xnorpop2x2g.c", "xnorpop4x1g.c")
SHAPES = ["151x3x8x15", "137x8x8x15", "64x16x16x7", "64x32x32x7", "128x32x32x15", "32x64x64x3"]

MAIN = r"""
/* the benchmark program of ultracompactBNN's 1D convolution, on the current PULP SDK (pmsis): the same measurement as
   their stats.h -- HOTTING warm-up runs, then the mean of REPEAT runs -- with pi_perf_* instead of rt_perf_* */
#include "pmsis.h"
#include <stdint.h>
#include "data.h"
#include "kernels.h"
#define HOTTING 2
#define REPEAT 3
/* their data_allocation.h, plus one spare word: their kernel reads one word past the input and the weights (masked) */
static const uint32_t input_data_int1[(C_IN * L_IN + 31) / 32 + 1] = INPUT;
static const uint32_t weights[(C_OUT * K_S * C_IN + 31) / 32 + 1] = WEIGHTS;
static uint32_t output[(C_OUT * L_OUT + 31) / 32];
static const int16_t thresholds[C_OUT] = THRESHOLDS;
static const uint32_t checksum[(C_OUT * L_OUT + 31) / 32] = OUTPUT;

int main(void) {
    unsigned long cycles = 0, instr = 0;
    pi_perf_conf((1 << PI_PERF_CYCLES) | (1 << PI_PERF_INSTR));
    for (int k = 0; k < HOTTING + REPEAT; k++) {
        for (int i = 0; i < (C_OUT * L_OUT + 31) / 32; i++) output[i] = 0;   /* their kernel ORs into the output */
        pi_perf_reset();
        pi_perf_start();
        conv_bin_1D_nopad_nodilation(input_data_int1, L_IN, C_IN, weights, C_OUT, K_S, STRIDE, output, L_OUT, thresholds);
        pi_perf_stop();
        if (k >= HOTTING) {
            cycles += pi_perf_read(PI_PERF_CYCLES);
            instr += pi_perf_read(PI_PERF_INSTR);
        }
    }
    int errors = 0;
    for (int i = 0; i < (C_OUT * L_OUT + 31) / 32; i++) errors += output[i] != checksum[i];
    printf("UCBNN cycles=%lu instr=%lu errors=%d\n", cycles / REPEAT, instr / REPEAT, errors);
    return 0;
}
"""

MAKEFILE = """APP = test
APP_SRCS = src/main.c """ + " ".join(f"src/{k}" for k in KERNELS) + """
APP_CFLAGS += -O3 -w -flto -I$(CURDIR) -I$(CURDIR)/include
APP_LDFLAGS += -flto
include $(RULES_DIR)/pmsis_rules.mk
"""


SYMBEX_MAIN = r"""
/* the same layer through lib_symbex's 1D convolution, measured exactly as ultracompactBNN's program */
#include "pmsis.h"
#include <stdint.h>
#include "layers/conv/symbex_bool.h"
#include "layers/conv/symbex_bool1d.h"
#include "data_symbex.h"
#define HOTTING 2
#define REPEAT 3
static const uint32_t in[IN_WORDS] = S_INPUT;
static const uint32_t w[C_OUT * RW] __attribute__((aligned(16))) = S_WEIGHTS;
static const int32_t thr[C_OUT] = S_THRESHOLDS;
static const int32_t cnt[C_OUT] = S_COUNTS;
static const uint32_t expected[OUT_WORDS] = S_OUTPUT;
static uint32_t out[OUT_WORDS];
static uint32_t win[2 * RW] __attribute__((aligned(16)));
static int32_t scr[C_OUT];

int main(void) {
    const struct SymbexBool1D L = {C_IN, C_OUT, K_S, 1, 1, L_IN, L_OUT, RW, w, thr, cnt, NULL};
    unsigned long cycles = 0, instr = 0;
    symbex_bool_force_mode = FORM;
    pi_perf_conf((1 << PI_PERF_CYCLES) | (1 << PI_PERF_INSTR));
    for (int k = 0; k < HOTTING + REPEAT; k++) {
        pi_perf_reset();
        pi_perf_start();
        symbex_bool1d_conv(&L, in, out, win, scr);
        pi_perf_stop();
        if (k >= HOTTING) {
            cycles += pi_perf_read(PI_PERF_CYCLES);
            instr += pi_perf_read(PI_PERF_INSTR);
        }
    }
    int errors = 0;
    for (int i = 0; i < OUT_WORDS; i++) errors += out[i] != expected[i];
    printf("UCBNN cycles=%lu instr=%lu errors=%d\n", cycles / REPEAT, instr / REPEAT, errors);
    return 0;
}
"""


def symbex_data(x, w, thr_matches):
    """the same layer in lib_symbex's layout: LSB-first stream, rows padded to 4 words, thresholds on
    raw = agreements - disagreements = 2 * matches - n"""
    l_in, c_in = x.shape
    c_out, k, _ = w.shape
    n = k * c_in
    l_out = l_in - k + 1
    rw = -(-(-(-n // 32)) // 4) * 4

    def lsb(bits, n_words):
        pad = np.zeros(n_words * 32, dtype=np.uint8)
        pad[:len(bits)] = bits
        return np.packbits(pad, bitorder="little").view("<u4")

    rows = np.zeros((c_out, rw), dtype=np.uint32)
    for o in range(c_out):
        rows[o] = lsb(w[o].ravel(), rw)
    win = np.stack([x[t:t + k].ravel() for t in range(l_out)])
    matches = (win[:, None, :] == w.reshape(c_out, n)[None]).sum(2)
    out = (matches >= thr_matches[None]).astype(np.uint8)
    in_words = -(-l_in * c_in // 32) + 1
    out_words = -(-l_out * c_out // 32) + 1
    arr = lambda v: "{" + ", ".join(str(int(a)) for a in v) + "}"
    hexa = lambda v: "{" + ", ".join(f"0x{int(a) & 0xFFFFFFFF:08X}" for a in v) + "}"
    return "\n".join([
        f"#define L_IN {l_in}", f"#define C_IN {c_in}", f"#define C_OUT {c_out}", f"#define K_S {k}",
        f"#define L_OUT {l_out}", f"#define RW {rw}", f"#define IN_WORDS {in_words}", f"#define OUT_WORDS {out_words}",
        f"#define S_INPUT {hexa(lsb(x.ravel(), in_words))}", f"#define S_WEIGHTS {hexa(rows.ravel())}",
        f"#define S_THRESHOLDS {arr(2 * thr_matches.astype(np.int64) - n)}", f"#define S_COUNTS {arr(w.reshape(c_out, n).sum(1))}",
        f"#define S_OUTPUT {hexa(lsb(out.ravel(), out_words))}", ""])


def run_symbex(shape, lib, work, layer_bits, form):
    """lib_symbex's 1D convolution on the same layer"""
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "src"))
    x, w, thr = layer_bits
    open(os.path.join(work, "src", "main.c"), "w").write(SYMBEX_MAIN)
    open(os.path.join(work, "src", "data_symbex.h"), "w").write(symbex_data(x, w, thr))
    srcs = [os.path.join(lib, "src", "layers", "conv", "symbex_bool1d.c"),
            os.path.join(lib, "src", "layers", "conv", "symbex_bool.c"),
            os.path.join(lib, "src", "hal", "generic", "symbex_hal_impl.c")]
    open(os.path.join(work, "Makefile"), "w").write(
        "APP = test\nAPP_SRCS = src/main.c " + " ".join(srcs) + "\n"
        f"APP_CFLAGS += -O3 -w -flto -I$(CURDIR)/src -I{os.path.join(lib, 'include')} -DSYMBEX_POPCOUNT_FORCE_NATIVE "
        f"-DFORM={form}\nAPP_LDFLAGS += -flto\ninclude $(RULES_DIR)/pmsis_rules.mk\n")
    r = subprocess.run("make clean all run", shell=True, cwd=work, capture_output=True, text=True)
    m = re.search(r"UCBNN cycles=(\d+) instr=(\d+) errors=(\d+)", r.stdout)
    if not m:
        print((r.stdout + r.stderr)[-2500:])
        sys.exit(f"[X] SYMBEX {shape}: no result from the simulator (the output above says why)")
    asm = subprocess.run(f"riscv32-unknown-elf-objdump -d $(find {work} -name test -type f | head -1) | grep -c 'p.cnt'",
                         shell=True, capture_output=True, text=True).stdout.strip()
    return dict(zip(("cycles", "instr", "errors"), (int(v) for v in m.groups())), p_cnt=int(asm or 0))


def run(shape, repo, work, rng, layer_bits=None):
    l_in, c_in, c_out, k = (int(v) for v in shape.split("x"))
    src = os.path.join(repo, "template_convolution1d_1bit", "gvsoc")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "src"))
    os.makedirs(os.path.join(work, "include"))
    os.makedirs(os.path.join(work, "rt"))
    open(os.path.join(work, "rt", "rt_api.h"), "w").write('#include "pmsis.h"\n#define RT_FC_SHARED_DATA\n')
    shutil.copy(os.path.join(src, "include", "kernels.h"), os.path.join(work, "include", "kernels.h"))
    for f in KERNELS:
        shutil.copy(os.path.join(src, "src", f), os.path.join(work, "src", f))
    open(os.path.join(work, "src", "main.c"), "w").write(MAIN)
    if layer_bits is None:
        d = layer(rng, l_in, c_in, c_out, k)
    else:                                             # the given layer, in their layout
        from ucbnn_check import msb_words
        x, w, thr = layer_bits
        win = np.stack([x[t:t + k].ravel() for t in range(l_in - k + 1)])
        out = ((win[:, None, :] == w.reshape(c_out, k * c_in)[None]).sum(2) >= thr[None]).astype(np.uint8)
        d = dict(L_IN=l_in, C_IN=c_in, C_OUT=c_out, K_S=k, STRIDE=1, L_OUT=l_in - k + 1, INPUT=msb_words(x.ravel()),
                 WEIGHTS=msb_words(w.ravel()), THRESHOLDS=thr, OUTPUT=msb_words(out.ravel()))
    open(os.path.join(work, "src", "data.h"), "w").write(data_h(d))
    open(os.path.join(work, "Makefile"), "w").write(MAKEFILE)
    r = subprocess.run("make clean all run", shell=True, cwd=work, capture_output=True, text=True)
    m = re.search(r"UCBNN cycles=(\d+) instr=(\d+) errors=(\d+)", r.stdout)
    if not m:
        print((r.stdout + r.stderr)[-2500:])
        sys.exit(f"[X] {shape}: no result from the simulator (the output above says why)")
    cycles, instr, errors = (int(v) for v in m.groups())
    l_out = l_in - k + 1
    mac = c_in * l_out * c_out * k
    return dict(shape=shape, l_in=l_in, c_in=c_in, c_out=c_out, k=k, l_out=l_out, mac=mac, cycles=cycles,
                instr=instr, mac_per_cycle=mac / cycles, errors=errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the ultracompactBNN clone")
    ap.add_argument("--shapes", nargs="+", default=SHAPES, help="L_in x C_in x C_out x K (C_out: a power of 2)")
    ap.add_argument("--symbex", default="", help="lib_symbex folder: run the same layers through it too")
    ap.add_argument("--form", type=int, default=0, help="SYMBEX count form: 0 the HAL's choice, 1 AND, 2 XOR")
    args = ap.parse_args()
    os.chdir(HERE)
    repo = os.path.abspath(os.path.expanduser(args.repo))
    rng = np.random.default_rng(0)
    rows = []
    lib = os.path.abspath(os.path.expanduser(args.symbex)) if args.symbex else ""
    print("\n    binary 1D convolution on GVSoC (RI5CY, PULP GCC -O3 -flto): ultracompactBNN's kernels unchanged"
          + (", and lib_symbex on the same layers" if lib else "") + "\n")
    if lib:
        print(f"    {'L_in x C_in x C_out x K':<24}{'MAC':>9} | {'ucbnn instr':>11}{'cycles':>9} | {'SYMBEX instr':>12}"
              f"{'cycles':>9} | {'instr ratio':>11}  outputs")
        print("    " + "-" * 104)
    else:
        print(f"    {'L_in x C_in x C_out x K':<26}{'MAC':>10}{'cycles':>11}{'MAC/cycle':>11}{'output':>12}")
        print("    " + "-" * 70)
    for s in args.shapes:
        l_in, c_in, c_out, k = (int(v) for v in s.split("x"))
        x = rng.integers(0, 2, (l_in, c_in), dtype=np.uint8)
        w = rng.integers(0, 2, (c_out, k, c_in), dtype=np.uint8)
        n = k * c_in
        thr = rng.integers(n // 2 - n // 4, n // 2 + n // 4 + 1, c_out).astype(np.int16)
        r = run(s, repo, os.path.join(HERE, "runs", "gvsoc", s), rng, (x, w, thr))
        if lib:
            sx = run_symbex(s, lib, os.path.join(HERE, "runs", "gvsoc_symbex", s), (x, w, thr), args.form)
            r["symbex"] = sx
            ok = "both exact" if r["errors"] == 0 and sx["errors"] == 0 else f"[X] ucbnn {r['errors']}, SYMBEX {sx['errors']}"
            print(f"    {s:<24}{r['mac']:>9} | {r['instr']:>11}{r['cycles']:>9} | {sx['instr']:>12}{sx['cycles']:>9} | "
                  f"{r['instr'] / sx['instr']:>10.2f}x  {ok}" + ("" if sx["p_cnt"] else "  [!] no p.cnt in SYMBEX"))
        else:
            print(f"    {s:<26}{r['mac']:>10}{r['cycles']:>11}{r['mac_per_cycle']:>11.2f}"
                  f"{'exact' if r['errors'] == 0 else str(r['errors']) + ' words [X]':>12}")
        rows.append(r)
    if lib:
        print("\n    instr ratio = ultracompactBNN instructions / SYMBEX instructions (> 1: SYMBEX does less work)")
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(os.path.join("results", "ucbnn_gvsoc.json"), "w"), indent=1)
    print("\n    -> results/ucbnn_gvsoc.json")


if __name__ == "__main__":
    main()
