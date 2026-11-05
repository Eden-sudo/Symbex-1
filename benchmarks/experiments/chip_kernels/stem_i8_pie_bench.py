"""
stem_i8_pie_bench.py -- a faster int8 stem for the ESP32-S3: vectorized over the output channels instead of the window.

lib_symbex's int8 stem (symbex_bool_stem_i8) builds a 27-byte window per output pixel, then one PIE dot product per
output channel (symbex_s8_dot_rows: 2 x ee.vmulas.s8.accx + reading ACCX, per channel). Here the loops are turned
around, as ESP-DL's int8 convolutions do: the 16 lanes of a PIE register are 16 output channels of one pixel.
    weights      transposed once per call to [tap][16 channels] (a 16-byte vector per tap, in RAM)
    per pixel    QACC (16 lanes of 20 bits) loaded with -threshold; per tap: the input byte broadcast to 16 lanes
                 (ee.vldbc.8, straight from the image; XOR 0x80 = pixel - 128), its 16 weights, ee.vmulas.s8.qacc
    out bits     ee.srcmb.s8.qacc (saturated, the sign kept) -> bit = lane >= 0
Exact only if no lane leaves 20 bits: |threshold| + 128 * sum|w| < 2^19 per channel (channels whose bit never
changes are set directly); otherwise the kernel says so and the library's kernel must run.

    python stem_i8_pie_bench.py --port /dev/ttyACM0                  # weights in flash, then in RAM
    python stem_i8_pie_bench.py --port /dev/ttyACM0 --weights ram

Checked on the chip: the new kernel against the library's (and both against plain C) on 3 shapes: ResNet-8's stem
(32 x 32, 16 channels, stride 1; the image in RAM, as from a camera), 40 channels with stride 2 (a partial group of 16) and a 1 x 1 kernel. The first
line printed is a probe of the QACC layout (lane j of ee.vmulas.s8.qacc = 20-bit slot j % 8 of QACC_L / QACC_H).
Times: best of 7 (cycles at 240 MHz). Results: results/stem_i8_pie_bench_s3.json.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
LIB = os.path.join(ROOT, "Symbex1_PIO", "lib", "lib_symbex")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.chip.core import BOARDS, flash, open_port  # noqa: E402

# (name, in_hw, out_ch, kernel, stride): the first is cnn_cifar_resnet8's stem, the one timed in the table
SHAPES = [("resnet8", 32, 16, 3, 1), ("s2_c40", 32, 40, 3, 2), ("w48_c24", 48, 24, 3, 1)]

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "hal/symbex_hal.h"
#include "layers/conv/symbex_bool.h"
#include "bench_data.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())

/* ---- the new kernel ------------------------------------------------------------------------------------------ */

#define QLIM 524287                                  /* a QACC lane of ee.vmulas.s8.qacc: 20-bit signed */
#define STEM_PIE_MAX_W 128                           /* the ring of 3 padded rows lives on the stack */
#define RING_ROW(w) ((((w) + 2) * 3 + 15) & ~15)     /* bytes of one padded row, a whole number of 16-byte blocks */

/* channel j of a group of 16 runs in lane LANE(j): then the 4 words of the 16 saturated lanes pack into the 16 bits
   with shifts and masks only (pack16) */
#define LANE(j) (4 * ((j) & 3) + ((j) >> 2))

/* QACC as ee.ld.qacc_* read it (ESP-DL's bias layout; checked by the probe): bytes 0-19 QACC_L (lanes 0-7, 20 bits
   each, little-endian), bytes 32-51 QACC_H (lanes 8-15) */
static void qacc_pack(uint8_t* q, int lane, int32_t v) {
    uint8_t* h = q + (lane >> 3) * 32;
    const int bit = (lane & 7) * 20;
    const uint32_t u = (uint32_t)v & 0xFFFFFu;
    for (int b = 0; b < 20; b++) {
        const int p = bit + b;
        if ((u >> b) & 1u) h[p >> 3] |= (uint8_t)(1u << (p & 7));
    }
}

/* a_i = lanes 4i..4i+3 (one byte each, int8): byte j of m gets bit i = lane 4i + j is negative, = channel 4j + i */
static inline uint32_t pack16(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    const uint32_t M = 0x01010101u;
    uint32_t m = ((a0 >> 7) & M) | ((a1 >> 6) & (M << 1)) | ((a2 >> 5) & (M << 2)) | ((a3 >> 4) & (M << 3));
    m ^= 0x0F0F0F0Fu;                                /* bit = lane >= 0 */
    return (m & 0xFu) | ((m >> 4) & 0xF0u) | ((m >> 8) & 0xF00u) | ((m >> 12) & 0xF000u);
}

/* one padded row of the ring: int8 (pixel - 128), a zero pixel left and right; outside the image all zero */
static void ring_fill(int8_t* dst, const uint8_t* img, int y, int H, int W) {
    memset(dst, 0, RING_ROW(W));
    if (y < 0 || y >= H) return;
    const uint8_t* src = img + (size_t)y * W * 3;
    for (int i = 0; i < W * 3; i++) dst[3 + i] = (int8_t)(src[i] ^ 0x80u);
}

/* the 3 rows of a pixel's window: realigned from the 2 aligned blocks of its first and last byte (never past them) */
#define ROW(p, q) "addi %[a0], " p ", 8 \n\t ee.ld.128.usar.ip q0, " p ", 0 \n\t ee.vld.128.ip q1, %[a0], 0 \n\t" \
                  "ee.src.q " q ", q0, q1 \n\t"
/* tap i: its 16 weights (in q[i % 3], loaded 2 taps before) times byte sel of the row, the weights of tap i + 2
   loaded meanwhile */
#define T(wc, wn, q, sel) "ee.vsmulas.s8.qacc.ld.incp " #wn ", %[w], " #wc ", " #q ", " #sel " \n\t"
#define TL(wc, q, sel) "ee.vsmulas.s8.qacc " #wc ", " #q ", " #sel " \n\t"
#define PIXEL                                                                                                        \
    ROW("%[p0]", "q4") ROW("%[p1]", "q5") ROW("%[p2]", "q6")                                                         \
    "ee.ld.qacc_l.l.128.ip %[q], 16 \n\t ee.ld.qacc_l.h.32.ip %[q], 16 \n\t"                                          \
    "ee.ld.qacc_h.l.128.ip %[q], 16 \n\t ee.ld.qacc_h.h.32.ip %[q], 16 \n\t"                                          \
    "ee.vld.128.ip q0, %[w], 16 \n\t ee.vld.128.ip q1, %[w], 16 \n\t"                                                \
    T(q0, q2, q4, 0) T(q1, q0, q4, 1) T(q2, q1, q4, 2) T(q0, q2, q4, 3) T(q1, q0, q4, 4) T(q2, q1, q4, 5)          \
    T(q0, q2, q4, 6) T(q1, q0, q4, 7) T(q2, q1, q4, 8) T(q0, q2, q5, 0) T(q1, q0, q5, 1) T(q2, q1, q5, 2)          \
    T(q0, q2, q5, 3) T(q1, q0, q5, 4) T(q2, q1, q5, 5) T(q0, q2, q5, 6) T(q1, q0, q5, 7) T(q2, q1, q5, 8)          \
    T(q0, q2, q6, 0) T(q1, q0, q6, 1) T(q2, q1, q6, 2) T(q0, q2, q6, 3) T(q1, q0, q6, 4) T(q2, q1, q6, 5)          \
    T(q0, q2, q6, 6) TL(q1, q6, 7) TL(q2, q6, 8)                                                                    \
    "movi %[a0], 0 \n\t ee.srcmb.s8.qacc q2, %[a0], 0 \n\t"                                                         \
    "ee.movi.32.a q2, %[a0], 0 \n\t ee.movi.32.a q2, %[a1], 1 \n\t"                                                 \
    "ee.movi.32.a q2, %[a2], 2 \n\t ee.movi.32.a q2, %[a3], 3 \n\t"

/* 3 x 3 stems only (stride 1 or 2, width <= STEM_PIE_MAX_W). 0 = done; -1 = not this kernel's (run
   symbex_bool_stem_i8): another kernel size, too wide, or a lane could leave 20 bits */
int stem_i8_pie(const struct SymbexBoolStemI8* L, const uint8_t* img, uint32_t* out) {
    const int H = L->in_h, W = L->in_w, s = L->stride, ow = L->out_w, n_out = L->out_ch, pout = (n_out + 31) / 32;
    if (L->kernel != 3 || W > STEM_PIE_MAX_W) return -1;
    for (int o = 0; o < n_out; o++) {                /* the checks first: refuse before writing anything */
        int32_t b = 0;
        for (int i = 0; i < 27; i++) { const int v = L->weights[o * L->row_bytes + i]; b += 128 * (v < 0 ? -v : v); }
        const int32_t t = L->thresholds[o];
        if (t > b || t <= -b) continue;              /* a constant bit */
        if ((t < 0 ? -t : t) + b > QLIM) return -1;
    }
    const int rb = RING_ROW(W);
    int8_t ring[3 * RING_ROW(STEM_PIE_MAX_W)] __attribute__((aligned(16)));
    int8_t wt[27 * 16 + 32] __attribute__((aligned(16)));    /* + 2 vectors: the loads 2 taps ahead read past 27 */
    uint8_t qi[64] __attribute__((aligned(16)));
    for (int g = 0; g * 16 < n_out; g++) {
        const int m = n_out - g * 16 < 16 ? n_out - g * 16 : 16;
        uint32_t live = 0, ones = 0;                 /* live: computed channels; ones: channels always 1 */
        memset(wt, 0, sizeof wt);
        memset(qi, 0, sizeof qi);
        for (int j = 0; j < m; j++) {
            const int8_t* r = L->weights + (g * 16 + j) * L->row_bytes;
            int32_t b = 0;
            for (int i = 0; i < 27; i++) { wt[i * 16 + LANE(j)] = r[i]; b += 128 * (r[i] < 0 ? -r[i] : r[i]); }
            const int32_t t = L->thresholds[g * 16 + j];
            if (t > b) continue;                     /* always 0 */
            if (t <= -b) { ones |= 1u << j; continue; }
            live |= 1u << j;
            qacc_pack(qi, LANE(j), -t);
        }
        const int sh = (g & 1) * 16, wd = g >> 1;
        int have = -2;                               /* the last image row in the ring (row y in slot (y + 3) % 3) */
        for (int oy = 0; oy < L->out_h; oy++) {
            const int iy0 = oy * s - 1;
            for (int y = have + 1 > iy0 ? have + 1 : iy0; y <= iy0 + 2; y++) ring_fill(ring + ((y + 3) % 3) * rb, img, y, H, W);
            have = iy0 + 2;
            const int8_t* r0 = ring + ((iy0 + 3) % 3) * rb;
            const int8_t* r1 = ring + ((iy0 + 4) % 3) * rb;
            const int8_t* r2 = ring + ((iy0 + 5) % 3) * rb;
            uint32_t* o = out + (size_t)oy * ow * pout + wd;
            for (int ox = 0; ox < ow; ox++, o += pout) {
                const int8_t *p0 = r0 + ox * s * 3, *p1 = r1 + ox * s * 3, *p2 = r2 + ox * s * 3;
                const int8_t* wp = wt;
                const uint8_t* qp = qi;
                uint32_t a0, a1, a2, a3;
                __asm__ __volatile__ (PIXEL : [p0] "+r"(p0), [p1] "+r"(p1), [p2] "+r"(p2), [w] "+r"(wp), [q] "+r"(qp),
                                      [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3) : : "memory");
                const uint32_t bits = ((pack16(a0, a1, a2, a3) & live) | ones) << sh;
                if (sh) *o |= bits; else *o = bits;  /* the first group of a word writes it whole */
            }
        }
    }
    return 0;
}

/* ---- the bench ----------------------------------------------------------------------------------------------- */

static void probe(void) {                            /* QACC after x = 1, w = 1..16 (lane j holds j + 1) */
    static const int8_t one __attribute__((aligned(4))) = 1;
    static const int8_t w16[16] __attribute__((aligned(16))) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint8_t raw[64] __attribute__((aligned(16)));
    int8_t sat[16] __attribute__((aligned(16)));
    memset(raw, 0, sizeof raw);
    uint8_t* rp = raw;
    __asm__ __volatile__ ("ee.zero.qacc \n\t ee.vldbc.8 q0, %1 \n\t ee.vld.128.ip q1, %2, 0 \n\t"
                          "ee.vmulas.s8.qacc q0, q1 \n\t"
                          "ee.st.qacc_l.l.128.ip %0, 16 \n\t ee.st.qacc_l.h.32.ip %0, 16 \n\t"
                          "ee.st.qacc_h.l.128.ip %0, 16 \n\t ee.st.qacc_h.h.32.ip %0, 16 \n\t"
                          "movi a8, 0 \n\t ee.srcmb.s8.qacc q2, a8, 0 \n\t ee.vst.128.ip q2, %3, 0 \n\t"
                          : "+r"(rp) : "r"(&one), "r"(w16), "r"(sat) : "a8", "memory");
    printf("P:");
    for (int i = 0; i < 64; i++) printf("%02x", raw[i]);
    printf("|");
    for (int i = 0; i < 16; i++) printf("%d,", sat[i]);
    printf("\n");
}

/* plain C: the reference of both kernels */
static void stem_ref(const struct SymbexBoolStemI8* L, const uint8_t* img, uint32_t* out) {
    const int k = L->kernel, pad = k / 2, pout = (L->out_ch + 31) / 32;
    memset(out, 0, (size_t)L->out_h * L->out_w * pout * 4);
    for (int oy = 0; oy < L->out_h; oy++)
        for (int ox = 0; ox < L->out_w; ox++)
            for (int o = 0; o < L->out_ch; o++) {
                int32_t d = 0;
                for (int kh = 0; kh < k; kh++)
                    for (int kw = 0; kw < k; kw++) {
                        const int iy = oy * L->stride - pad + kh, ix = ox * L->stride - pad + kw;
                        if (iy < 0 || iy >= L->in_h || ix < 0 || ix >= L->in_w) continue;
                        for (int c = 0; c < 3; c++)
                            d += ((int32_t)img[(iy * L->in_w + ix) * 3 + c] - 128)
                                 * L->weights[o * L->row_bytes + (kh * k + kw) * 3 + c];
                    }
                if (d >= L->thresholds[o]) out[(oy * L->out_w + ox) * pout + o / 32] |= 1u << (o % 32);
            }
}

#define REPS 7
static int8_t win[SYMBEX_BOOL_STEM_I8_WINDOW] __attribute__((aligned(16)));
static int32_t scratch[64];
static uint32_t o_lib[MAX_OUT_WORDS], o_new[MAX_OUT_WORDS], o_ref[MAX_OUT_WORDS];

static void bench_once(void) {
    printf("STEM I8 PIE BENCH ready (%s)\n", WEIGHTS_WHERE);
    probe();
    for (int c = 0; c < N_CASES; c++) {
        const struct SymbexBoolStemI8* L = &CASES[c];
        const size_t nw = (size_t)L->out_h * L->out_w * ((L->out_ch + 31) / 32);
        uint32_t best_lib = UINT32_MAX, best_new = UINT32_MAX;
        int rc = 0;
        for (int r = 0; r < REPS; r++) {
            memset(o_new, 0xA5, sizeof o_new);      /* the kernel must write every word itself */
            uint32_t t0 = NOW();
            symbex_bool_stem_i8(L, CASE_IMG[c], o_lib, win, scratch);
            uint32_t t1 = NOW();
            rc |= stem_i8_pie(L, CASE_IMG[c], o_new);
            uint32_t t2 = NOW();
            if (t1 - t0 < best_lib) best_lib = t1 - t0;
            if (t2 - t1 < best_new) best_new = t2 - t1;
        }
        stem_ref(L, CASE_IMG[c], o_ref);
        const int ok_lib = memcmp(o_lib, o_ref, nw * 4) == 0, ok_new = rc == 0 && memcmp(o_new, o_ref, nw * 4) == 0;
        int ones = 0;
        for (size_t i = 0; i < nw; i++) ones += __builtin_popcount(o_ref[i]);
        printf("R:%s|%lu|%lu|%d|%d|%d|%d\n", CASE_NAMES[c], (unsigned long)best_lib, (unsigned long)best_new, ok_lib,
               ok_new, rc, ones);
    }
    printf("DONE\n");
}

void app_main(void) {
    for (;;) {
        bench_once();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
"""


def c_array(ctype, name, values, rom, per_line=24):
    vals = [str(int(v)) for v in np.asarray(values).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return (f"static {'ROM ' if rom else ''}{ctype} {name}[{len(vals)}] __attribute__((aligned(16))) = {{\n  "
            + ",\n  ".join(lines) + "\n};\n")


def write_data(path, weights):
    """random images and weights; thresholds near the middle of each channel's dot products (both bit values)"""
    rng = np.random.default_rng(2026)
    parts, cases, max_words, max_pad = [], [], 0, 0
    for c, (name, hw, ch, k, s) in enumerate(SHAPES):
        n, rb = k * k * 3, (k * k * 3 + 3) & ~3
        img = rng.integers(0, 256, (hw, hw, 3))
        w = np.zeros((ch, rb), np.int64)
        w[:, :n] = rng.integers(-60, 61, (ch, n))    # 128 * sum|w| < 2^19 - |threshold|: the fast path
        x = img.astype(np.int64) - 128
        pad = k // 2
        xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
        out_hw = hw // s
        win = np.stack([xp[ky:ky + out_hw * s:s, kx:kx + out_hw * s:s] for ky in range(k) for kx in range(k)], 2)
        dots = win.reshape(out_hw * out_hw, n) @ w[:, :n].T
        thr = np.median(dots, 0).astype(np.int64) + rng.integers(-50, 51, ch)
        thr[0] = 10 ** 6                             # a channel always 0 and one always 1 (the constant lanes)
        thr[1] = -10 ** 6
        parts += [c_array("uint8_t", f"IMG{c}", img, False), c_array("int8_t", f"W{c}", w, True),
                  c_array("int32_t", f"T{c}", thr, True)]
        cases.append(f"{{{ch}, {k}, {s}, {hw}, {hw}, {out_hw}, {out_hw}, {rb}, W{c}, T{c}}}")
        max_words = max(max_words, out_hw * out_hw * ((ch + 31) // 32))
        max_pad = max(max_pad, (hw + 2 * pad) ** 2 * 3)
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define ROM {'const' if weights == 'flash' else ''}\n"
                f"#define WEIGHTS_WHERE \"weights in {weights}\"\n#define N_CASES {len(SHAPES)}\n"
                f"#define MAX_OUT_WORDS {max_words}\n#define MAX_PAD_BYTES {max_pad}\n")
        f.write("".join(parts))
        f.write("static const struct SymbexBoolStemI8 CASES[N_CASES] = {\n  " + ",\n  ".join(cases) + "\n};\n")
        f.write("static const uint8_t* const CASE_IMG[N_CASES] = {"
                + ", ".join(f"IMG{c}" for c in range(len(SHAPES))) + "};\n")
        f.write("static const char* const CASE_NAMES[N_CASES] = {"
                + ", ".join(f'"{s[0]}"' for s in SHAPES) + "};\n")


LIB_FILES = {"symbex_and.c": ("ops", "symbex_and.c"), "symbex_bool.c": ("layers", "conv", "symbex_bool.c"),
             "hal_generic.c": ("hal", "generic", "symbex_hal_impl.c"),
             "hal_s3.c": ("hal", "esp32s3", "symbex_hal_impl.c")}


def write_project(project, port, weights):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    for name, rel in LIB_FILES.items():
        shutil.copy(os.path.join(LIB, "src", *rel), os.path.join(src, name))
    write_data(os.path.join(src, "bench_data.h"), weights)
    w("platformio.ini", f"[env:s3]\nplatform = espressif32\nboard = {BOARDS['s3']}\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    for f in os.listdir(project):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project, f))
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n"
                            "CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n")
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(stem_i8_pie_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in LIB_FILES)
                            + ' INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)


def read_results(port, weights):
    s = open_port(port, 115200, reset="usb")
    rows, probe, ready, t0 = {}, None, False, time.time()
    while time.time() - t0 < 60:
        line = s.readline().decode(errors="replace").strip()
        if "STEM I8 PIE BENCH ready" in line:
            if f"(weights in {weights})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected weights in {weights}")
            ready, rows = True, {}
        elif ready and line.startswith("P:"):
            probe = line[2:]
        elif ready and line.startswith("R:"):
            name, lib, new, ok_lib, ok_new, rc, ones = line[2:].split("|")
            rows[name] = dict(lib_cycles=int(lib), new_cycles=int(new), lib_equal=ok_lib == "1",
                              new_equal=ok_new == "1", refused=rc != "0", ones=int(ones))
        elif ready and line == "DONE":
            s.close()
            return probe, rows
    s.close()
    sys.exit("[X] no results from the board in 60 s")


def check_probe(probe):
    """lane j of ee.vmulas.s8.qacc = 20-bit slot j % 8 of QACC_L (j < 8) / QACC_H: the layout qacc_pack writes"""
    raw, sat = probe.split("|")
    b = bytes.fromhex(raw)
    lanes = []
    for j in range(16):
        h = int.from_bytes(b[(j >> 3) * 32:(j >> 3) * 32 + 20], "little")
        lanes.append((h >> ((j & 7) * 20)) & 0xFFFFF)
    return lanes, [int(v) for v in sat.strip(",").split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--weights", choices=["flash", "ram", "both"], default="both")
    a = ap.parse_args()
    out = {}
    for weights in (["flash", "ram"] if a.weights == "both" else [a.weights]):
        project = os.path.join(HERE, "runs", f"stem_i8_pie_bench_{weights}")
        write_project(project, a.port, weights)
        flash(project, verbose=False, what=f"weights in {weights}")
        probe, rows = read_results(a.port, weights)
        lanes, sat = check_probe(probe)
        layout_ok = lanes == list(range(1, 17)) and sat == list(range(1, 17))
        print(f"\nQACC probe: lanes {lanes}, srcmb {sat} -> layout {'as assumed' if layout_ok else 'NOT as assumed'}")
        print(f"ESP32-S3, weights in {weights} (best of 7, ms at 240 MHz)")
        print(f"  {'shape':<10}{'library':>10}{'new':>10}{'speedup':>9}  equal (lib / new)")
        for name, r in rows.items():
            print(f"  {name:<10}{r['lib_cycles'] / 240e3:>10.3f}{r['new_cycles'] / 240e3:>10.3f}"
                  f"{r['lib_cycles'] / r['new_cycles']:>8.2f}x  {r['lib_equal']} / {r['new_equal']}"
                  + ("  (refused)" if r["refused"] else ""))
        out[weights] = dict(probe_layout_ok=layout_ok, cases=rows)
        if not all(r["lib_equal"] and r["new_equal"] for r in rows.values()):
            print("[X] a kernel differs from the plain C reference: the times are not trusted")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "stem_i8_pie_bench_s3.json")
    with open(path, "w") as f:
        json.dump(dict(board="s3", mhz=240, reps="best of 7", shapes=SHAPES, **out), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
