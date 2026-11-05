"""
int8_head_bench.py -- what does the int8 classifier of train_gap_study.py (--head int8) and small_gap_study.py cost on
the chip, against the binary one the networks run today? A separate test: nothing here is in the CDT or lib_symbex.

    python int8_head_bench.py --port /dev/ttyACM0                  # ESP32-S3: weights in flash, then in RAM
        (the S3's /dev/ttyACM number can change after flashing: /dev/serial/by-id/... keeps its name)
    python int8_head_bench.py --board esp32 --port /dev/ttyUSB0    # classic ESP32

The classifier reads the final map's bits (x = +-1) and gives one logit per class. Three shapes, random weights:
    resnet8   4096 bits (64 x 8 x 8) -> 10 classes, 31 % ones (measured: the trained x1 networks, 3 seeds)
    har_max   128 bits -> 17 classes, 50 % ones
    mlp       128 bits -> 10 classes, 50 % ones
Four ways:
    binary      lib_symbex's symbex_bool_classify: 1-bit weights, AND + popcount per class (what runs today)
    i8_dense    int8 weights, every bit: sum of the weights where the bit is 1 (branchless), logit = 2 * sum - sum(w)
    i8_sparse   int8 weights stored per input bit (bits x classes): only the bits that are 1 add their row of weights
    i8_planes   int8 weights as 8 bit planes: 8 AND + popcount passes (lib_symbex's symbex_and_popcount_rows), the
                planes weighted 1, 2, ..., 64 and -128 (two's complement)
Every way is checked on the chip against a plain C reference from the definition (the int8 ways: every logit equal;
binary: the same class), otherwise the times are not trusted.
The pooled int8 classifiers of train_gap_study.py --head pool2 / pool4, ResNet-8's final map (64 channels x 8 x 8, the
library's layout: per pixel 2 words of channel bits, 31 % ones):
    resnet8_pool2 / resnet8_pool4   the ones of every channel counted per 2 x 2 / 4 x 4 window (0..4 / 0..16), then
                                    int8 weights on the counts: 64 x 4 x 4 = 1,024 / 64 x 2 x 2 = 256 inputs per class
                                    (10 / 2.5 KB, against 40 KB for the int8 classifier on every bit). Two ways:
        _count      plain C: a count per bit, then one multiply-add per input
        _planes     the counts bit-sliced (32 channels per word, 3 / 5 count planes), the int8 weights as 8 bit planes:
                    AND + popcount per (count plane, weight plane) pair (symbex_and_popcount_rows), weighted 2^p * 2^q
                    (-128 for the sign plane)
    Checked: every logit equal to the ones numpy computed on the PC. Times: best of 7 runs, CPU cycles at 240 MHz.
Results: results/int8_head_bench_<board>.json (weights in flash and in RAM).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shortcut_bench import BOARD_HAL, LIB, LIB_FILES, c_array, pack  # noqa: E402
from compiler_core.chip.core import BOARDS, flash, open_port         # noqa: E402  (shortcut_bench adds tools/)

RESET = {"s3": "pico", "esp32": "rts"}   # s3: its USB port comes back after flashing; open it again until then
# (name, input bits, classes, share of ones in the input)
SHAPES = (("resnet8", 4096, 10, 0.31), ("har_max", 128, 17, 0.5), ("mlp", 128, 10, 0.5))
METHODS = ("binary", "i8_dense", "i8_sparse", "i8_planes")
# (name, window, side, channels, classes, share of ones): the pooled int8 classifiers
POOLS = (("resnet8_pool2", 2, 8, 64, 10, 0.31), ("resnet8_pool4", 4, 8, 64, 10, 0.31))

FIRMWARE = r"""
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include <string.h>
#include "layers/conv/symbex_bool.h"
#include "hal/symbex_hal.h"
#include "bench_data.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())

#define REPS 7
#define MAXC 32
struct Head {
    const char* name;
    int n_bits, n_words, classes, row_words;
    const uint32_t* x;                       /* row_words words, zero padded */
    const uint32_t* wb; const int32_t *wb_counts, *scale, *bias;     /* the binary classifier */
    const int8_t *w8, *w8t;                  /* classes x n_bits, n_bits x classes */
    const uint32_t* planes;                  /* 8 x classes x row_words: bit b of every int8 weight */
    const int32_t* w8sum;                    /* sum of row o */
};
HEADS_TABLE
struct Pool {
    const char* name;
    int window, side, channels, classes, n_in;
    const uint32_t* x;                       /* side x side pixels, channels / 32 words each */
    const int8_t* w;                         /* classes x n_in, input (window, channel) = window * channels + channel */
    const int32_t* ref;                      /* the logits, computed on the PC */
    int n_planes, row_words;                 /* count planes (counts 0..window^2), words per plane row */
    const uint32_t* wpl;                     /* 8 x classes x row_words: bit q of every int8 weight */
};
POOLS_TABLE
static const struct Pool* P;
static struct Pool pools_run[N_POOLS];       /* POOLS with the input map (and, weights in RAM, the weights) in RAM */
static const void* to_ram(const void* src, int bytes) {   /* the heap: the static DRAM is full on the classic ESP32 */
    void* d = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (d == NULL) { printf("[X] no RAM for %d bytes\n", bytes); return src; }
    memcpy(d, src, bytes);
    return d;
}
static uint8_t counts[1024];
static uint32_t cplanes[5 * 32] __attribute__((aligned(16)));   /* the S3 HAL takes its SIMD path on 16-byte aligned buffers */
static const struct Head* H;
static int32_t scratch[MAXC], logit[MAXC], cnt[MAXC], ref[MAXC];
static int cls;

static int argmax32(const int32_t* v, int n) {
    int b = 0;
    for (int o = 1; o < n; o++) if (v[o] > v[b]) b = o;
    return b;
}
static void m_binary(void) {
    cls = symbex_bool_classify(H->x, (uint16_t)H->n_words, (uint32_t)H->n_bits, H->wb, (uint16_t)H->row_words,
                               (uint16_t)H->classes, H->wb_counts, H->scale, H->bias, scratch);
}
static void m_i8_dense(void) {
    for (int o = 0; o < H->classes; o++) {
        const int8_t* w = H->w8 + o * H->n_bits;
        int32_t s = 0;
        for (int j = 0; j < H->n_words; j++) {
            const uint32_t v = H->x[j];
            const int8_t* ww = w + 32 * j;
            for (int b = 0; b < 32; b++) s += (int32_t)ww[b] & -(int32_t)((v >> b) & 1u);
        }
        logit[o] = 2 * s - H->w8sum[o];
    }
    cls = argmax32(logit, H->classes);
}
static void m_i8_sparse(void) {
    const int C = H->classes;
    int32_t acc[MAXC] = {0};
    for (int j = 0; j < H->n_words; j++) {
        uint32_t v = H->x[j];
        while (v) {
            const int8_t* r = H->w8t + (32 * j + __builtin_ctz(v)) * C;
            v &= v - 1;
            for (int o = 0; o < C; o++) acc[o] += r[o];
        }
    }
    for (int o = 0; o < C; o++) logit[o] = 2 * acc[o] - H->w8sum[o];
    cls = argmax32(logit, C);
}
static void m_i8_planes(void) {
    const int C = H->classes, plane = C * H->row_words;
    int32_t s[MAXC] = {0};
    for (int b = 0; b < 8; b++) {
        symbex_and_popcount_rows(H->x, H->planes + b * plane, H->row_words, C, cnt);
        for (int o = 0; o < C; o++) s[o] += b == 7 ? -128 * cnt[o] : cnt[o] << b;
    }
    for (int o = 0; o < C; o++) logit[o] = 2 * s[o] - H->w8sum[o];
    cls = argmax32(logit, C);
}
static void (*const FUNCS[4])(void) = {m_binary, m_i8_dense, m_i8_sparse, m_i8_planes};
static const char* const NAMES[4] = {"binary", "i8_dense", "i8_sparse", "i8_planes"};

static void m_pool(void) {
    const int k = P->window, nw = P->side / k, C = P->channels, cw = C / 32, n = P->n_in;
    for (int wy = 0; wy < nw; wy++)
        for (int wx = 0; wx < nw; wx++) {
            uint8_t* c = counts + (wy * nw + wx) * C;
            for (int i = 0; i < C; i++) c[i] = 0;
            for (int dy = 0; dy < k; dy++)
                for (int dx = 0; dx < k; dx++) {
                    const uint32_t* px = P->x + ((wy * k + dy) * P->side + wx * k + dx) * cw;
                    for (int j = 0; j < cw; j++) {
                        const uint32_t v = px[j];
                        for (int b = 0; b < 32; b++) c[32 * j + b] += (v >> b) & 1u;
                    }
                }
        }
    for (int o = 0; o < P->classes; o++) {
        const int8_t* w = P->w + o * n;
        int32_t s = 0;
        for (int i = 0; i < n; i++) s += (int32_t)w[i] * counts[i];
        logit[o] = s;
    }
    cls = argmax32(logit, P->classes);
}

static void m_pool_planes(void) {
    const int k = P->window, nw = P->side / k, cw = P->channels / 32, rw = P->row_words, np = P->n_planes;
    const int C = P->classes, plane = C * rw;
    for (int i = 0; i < np * rw; i++) cplanes[i] = 0;
    for (int wy = 0; wy < nw; wy++)
        for (int wx = 0; wx < nw; wx++)
            for (int j = 0; j < cw; j++) {
                const int at = (wy * nw + wx) * cw + j;
                for (int dy = 0; dy < k; dy++)
                    for (int dx = 0; dx < k; dx++) {
                        uint32_t carry = P->x[((wy * k + dy) * P->side + wx * k + dx) * cw + j];
                        for (int p = 0; p < np && carry; p++) {     /* add one bit to the bit-sliced counter */
                            const uint32_t c = cplanes[p * rw + at];
                            cplanes[p * rw + at] = c ^ carry;
                            carry &= c;
                        }
                    }
            }
    int32_t s[MAXC] = {0};
    for (int p = 0; p < np; p++)
        for (int q = 0; q < 8; q++) {
            symbex_and_popcount_rows(cplanes + p * rw, P->wpl + q * plane, (uint16_t)rw, (uint16_t)C, cnt);
            for (int o = 0; o < C; o++) s[o] += (q == 7 ? -128 * cnt[o] : cnt[o] << q) << p;
        }
    for (int o = 0; o < C; o++) logit[o] = s[o];
    cls = argmax32(logit, C);
}

static int bit(const uint32_t* v, int i) { return (v[i >> 5] >> (i & 31)) & 1u; }
/* from the definitions: int8 logits sum w * x (x = +-1); binary logits scale * (agreements - disagreements) + bias */
static int check(int method) {
    const int C = H->classes;
    if (method == 0) {
        int best = 0;
        int64_t best_l = 0;
        for (int o = 0; o < C; o++) {
            int32_t raw = 0;
            for (int i = 0; i < H->n_bits; i++) raw += bit(H->x, i) == bit(H->wb + o * H->row_words, i) ? 1 : -1;
            const int64_t l = (int64_t)H->scale[o] * raw + H->bias[o];
            if (o == 0 || l > best_l) { best_l = l; best = o; }
        }
        return cls == best;
    }
    for (int o = 0; o < C; o++) {
        ref[o] = 0;
        for (int i = 0; i < H->n_bits; i++) ref[o] += bit(H->x, i) ? H->w8[o * H->n_bits + i] : -H->w8[o * H->n_bits + i];
        if (ref[o] != logit[o]) return 0;
    }
    return cls == argmax32(ref, C);
}

static uint32_t best(void (*f)(void)) {
    uint32_t b = 0xFFFFFFFFu;
    for (int r = 0; r < REPS; r++) {
        const uint32_t t0 = NOW();
        f();
        const uint32_t t = NOW() - t0;
        if (t < b) b = t;
    }
    return b;
}

void app_main(void) {
    for (int p = 0; p < N_POOLS; p++) {
        const struct Pool* q = &POOLS[p];
        pools_run[p] = *q;
        pools_run[p].x = to_ram(q->x, q->side * q->side * q->channels / 8);
        if (POOLS_IN_RAM) {
            pools_run[p].w = to_ram(q->w, q->classes * q->n_in);
            pools_run[p].wpl = to_ram(q->wpl, 8 * q->classes * q->row_words * 4);
        }
    }
    for (;;) {
        printf("HEAD BENCH ready (%s)\n", WEIGHTS_WHERE);
        for (int h = 0; h < N_HEADS; h++) {
            H = &HEADS[h];
            for (int m = 0; m < 4; m++) {
                const uint32_t c = best(FUNCS[m]);
                printf("R:%s_%s|%lu|%d\n", H->name, NAMES[m], (unsigned long)c, check(m));
            }
        }
        for (int p = 0; p < N_POOLS; p++) {
            P = &pools_run[p];
            for (int m = 0; m < 2; m++) {
                for (int o = 0; o < P->classes; o++) logit[o] = 0;
                const uint32_t c = best(m ? m_pool_planes : m_pool);
                int ok = cls == argmax32(P->ref, P->classes);
                for (int o = 0; o < P->classes; o++) ok &= logit[o] == P->ref[o];
                printf("R:%s_%s|%lu|%d\n", P->name, m ? "planes" : "count", (unsigned long)c, ok);
            }
        }
        printf("DONE\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
"""


def head_arrays(rng, name, n_bits, classes, ones):
    """the C arrays of one shape and its line of the HEADS table"""
    n_words = -(-n_bits // 32)
    row_words = -(-n_words // 4) * 4                      # a multiple of 4 words, as the exporter pads rows
    x = (rng.random(n_bits) < ones).astype(np.uint8)[None]
    wb = rng.integers(0, 2, (classes, n_bits), dtype=np.uint8)
    w8 = rng.integers(-127, 128, (classes, n_bits))
    planes = np.concatenate([((w8 & 0xFF) >> b) & 1 for b in range(8)]).astype(np.uint8)
    p = name.upper()
    text = (c_array("uint32_t", f"{p}_X", pack(x, row_words), False)
            + c_array("uint32_t", f"{p}_WB", pack(wb, row_words), True)
            + c_array("int32_t", f"{p}_WBC", wb.sum(1), True)
            + c_array("int32_t", f"{p}_SC", rng.integers(1, 1000, classes), True)
            + c_array("int32_t", f"{p}_BI", rng.integers(-50000, 50000, classes), True)
            + c_array("int8_t", f"{p}_W8", w8, True)
            + c_array("int8_t", f"{p}_W8T", w8.T, True)
            + c_array("uint32_t", f"{p}_PL", pack(planes, row_words), True)
            + c_array("int32_t", f"{p}_W8S", w8.sum(1), True))
    row = (f'    {{"{name}", {n_bits}, {n_words}, {classes}, {row_words}, {p}_X, {p}_WB, {p}_WBC, {p}_SC, {p}_BI, '
           f"{p}_W8, {p}_W8T, {p}_PL, {p}_W8S}},")
    return text, row


def pool_arrays(rng, name, window, side, channels, classes, ones):
    """the C arrays of one pooled classifier, its logits computed here, and its line of the POOLS table"""
    bits = (rng.random((side, side, channels)) < ones).astype(np.int64)
    nw = side // window
    counts = bits.reshape(nw, window, nw, window, channels).sum((1, 3)).reshape(-1)   # (window, channel)
    w = rng.integers(-127, 128, (classes, len(counts)))
    p = name.upper()
    row_words = -(-(-(-len(counts) // 32)) // 4) * 4
    planes = np.concatenate([((w & 0xFF) >> b) & 1 for b in range(8)]).astype(np.uint8)
    n_planes = int(window * window).bit_length()          # counts 0..window^2
    text = (c_array("uint32_t", f"{p}_X", pack(bits.reshape(side * side, channels), channels // 32), False)
            + c_array("int8_t", f"{p}_W", w, True) + c_array("int32_t", f"{p}_REF", w @ counts, True)
            + c_array("uint32_t", f"{p}_WPL", pack(planes, row_words), True))
    row = (f'    {{"{name}", {window}, {side}, {channels}, {classes}, {len(counts)}, {p}_X, {p}_W, {p}_REF, '
           f"{n_planes}, {row_words}, {p}_WPL}},")
    return text, row


def write_data(path, weights):
    rng = np.random.default_rng(2026)
    texts, rows = zip(*(head_arrays(rng, *s) for s in SHAPES))
    rng = np.random.default_rng(2027)                      # its own generator: the heads' data stay as before
    ptexts, prows = zip(*(pool_arrays(rng, *s) for s in POOLS))
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define ROM {'const' if weights == 'flash' else ''}\n"
                f"#define WEIGHTS_WHERE \"weights in {weights}\"\n#define N_HEADS {len(SHAPES)}\n" + "".join(texts)
                + "#define HEADS_TABLE static const struct Head HEADS[N_HEADS] = { \\\n"
                + " \\\n".join(rows) + " \\\n};\n"
                + f"#define N_POOLS {len(POOLS)}\n#define POOLS_IN_RAM {int(weights == 'ram')}\n"
                + "".join(ptexts).replace("static ROM ", "static const ").replace("static uint32_t", "static const uint32_t")
                + "#define POOLS_TABLE static const struct Pool POOLS[N_POOLS] = { \\\n"
                + " \\\n".join(prows) + " \\\n};\n")


def write_project(project, port, weights, board):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)   # noqa: E731
    files = dict(LIB_FILES, **dict([BOARD_HAL[board]] if board in BOARD_HAL else []))
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    for name, rel in files.items():
        shutil.copy(os.path.join(LIB, "src", *rel), os.path.join(src, name))
    write_data(os.path.join(src, "bench_data.h"), weights)
    w("platformio.ini", f"[env:{board}]\nplatform = espressif32\nboard = {BOARDS[board]}\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    for f in os.listdir(project):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project, f))
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n"
                            + ("CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n" if board == "s3" else ""))
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(int8_head_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in files)
                            + ' INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)


def read_results(port, weights, board):
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = {}, False, time.time()
    while time.time() - t0 < 60:
        line = s.readline().decode(errors="replace").strip()
        if "HEAD BENCH ready" in line:
            if f"(weights in {weights})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected weights in {weights}")
            ready, rows = True, {}
        elif ready and line.startswith("R:"):
            name, cyc, ok = line[2:].split("|")
            rows[name] = (int(cyc), ok == "1")
        elif ready and line == "DONE":
            s.close()
            return rows
    s.close()
    sys.exit("[X] no results from the board in 60 s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(RESET), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--weights", choices=["flash", "ram", "both"], default="both")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    if a.no_upload and a.weights == "both":
        ap.error("--no-upload reads the firmware already on the board: give --weights flash or --weights ram")
    out = {}
    for weights in (["flash", "ram"] if a.weights == "both" else [a.weights]):
        project = os.path.join(HERE, "runs", f"int8_head_bench_{a.board}_{weights}")
        if not a.no_upload:
            write_project(project, a.port, weights, a.board)
            flash(project, verbose=False, what=f"weights in {weights}")
        rows = read_results(a.port, weights, a.board)
        print(f"\n{a.board}, weights in {weights} (best of 7, 240 MHz)")
        print(f"  {'shape / method':<22}{'cycles':>10}{'us':>9}  equal to the reference")
        for name, (cyc, ok) in rows.items():
            print(f"  {name:<22}{cyc:>10}{cyc / 240:>9.1f}  {'OK' if ok else 'NO'}")
        missing = sorted(({f"{shape}_{method}" for shape, *_ in SHAPES for method in METHODS}
                          | {f"{name}_{m}" for name, *_ in POOLS for m in ("count", "planes")}) - set(rows))
        if missing:                                        # a lost serial line must not make a short result file
            sys.exit(f"[X] no result for {', '.join(missing)}: the times are not trusted")
        if not all(ok for _, ok in rows.values()):
            sys.exit("[X] a method differs from its reference: the times are not trusted")
        out[weights] = {name: {"cycles": cyc, "us": round(cyc / 240, 2), "equal_reference": ok}
                        for name, (cyc, ok) in rows.items()}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"int8_head_bench_{a.board}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, mhz=240, reps="best of 7",
                       shapes=[dict(name=n, bits=b, classes=c, ones=p) for n, b, c, p in SHAPES],
                       pools=[dict(name=n, window=k, side=s, channels=ch, classes=c, ones=p)
                              for n, k, s, ch, c, p in POOLS], **out), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
