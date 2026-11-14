"""
dense_vertical_bench.py -- can a binary DENSE layer run vertically (bit-sliced), as the conv layers do on the S3?
The vertical engine of lib_symbex (symbex_bool_vertical, docs/vertical.md) puts 128 output PIXELS in the 128 lanes of
a PIE register. A dense layer has no pixels; here the lanes are 128 OUTPUT NEURONS instead, the roles of data and
weights swapped:
    weights    transposed: plane (g, i) = input weight i of the 128 neurons of group g (bit lane = neuron), plus one
               zero plane per group (the padding of the list)
    list       per input vector, ONCE per layer: the inputs that are 1 (as plane deltas, built a byte at a time with
               a 256-entry table, as the conv engine builds its lists from the weights)
    count      c_n = count(x AND w_n) for 128 neurons at once: the planes of the inputs that are 1, added with
               Harley-Seal carry-save adders (the conv engine's code: hs_gather, count_finish)
    threshold  +-1 score = 4 c - 2 count(x) - 2 count(w_n) + n_in >= thr_n  <=>  2 c - count(x) >= E_n,
               E_n = ceil((thr_n - n_in + 2 count(w_n)) / 2), kept as 12 planes of 2048 - E_n per group;
               out = bit 12 of 2 c + (2048 - count(x)) + (2048 - E_n): one bit-sliced 3-operand add per group
Only the inputs that are 1 cost anything (about half of them), and each costs ~6 instructions per 128 neurons.

    python dense_vertical_bench.py --port /dev/ttyACM0                     # ESP32-S3 (PIE)
    python dense_vertical_bench.py --port /dev/ttyACM0 --weights ram
    python dense_vertical_bench.py --board esp32 --port /dev/ttyUSB0       # the same steps in portable C
    python dense_vertical_bench.py --board pico2_arm --port /dev/ttyACM1

Shapes (in -> out): the hidden / output binary layers of the mlp_fashion networks and two larger ones. Each with two
inputs: half of the bits 1 (what a trained layer sees) and 90 % 1 (the slow case of the vertical method). Methods:
    xnor       symbex_dense_forward_bits (XNOR + popcount: lut16 on every board)
    and        symbex_and_raw + symbex_dense_threshold (AND; the PIE multiplier counts on the S3)
    vertical   this kernel (list + counts + threshold, everything per call)
Every method is checked against a plain C reference on the chip (the output bits), otherwise its times are not
trusted. Random weights (flash by default). Times: best of 7 runs (cycles on the ESP32 boards; ns per call on the
Pico 2). Results: results/dense_vertical_bench_<board>[_wram].json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dense_short_bench import (BOARD_HAL, BOARDS, LIB_FILES, RESET, TICKS, c_array, copy_lib,  # noqa: E402
                               flash, open_port)

SHAPES = [(128, 512), (512, 10), (256, 128), (128, 10), (512, 512), (1024, 256)]
DENSITIES = [0.5, 0.9]
EP = 12                                   # planes of the threshold operands (2 c, 2048 - count(x), 2048 - E < 4096)

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if defined(ESP_PLATFORM)
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define OUT printf
#define INNER 1
#define SCALE 1
#elif defined(SYMBEX_HOST)                                /* the PC check (--host): exactness, ns per call */
#include <time.h>
static uint32_t host_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint32_t)(t.tv_sec * 1000000000ull + t.tv_nsec); }
#define NOW() host_ns()
#define OUT printf
#define INNER 1
#define SCALE 1
#else
#include <Arduino.h>
#define NOW() ((uint32_t)micros())
#define OUT Serial.printf
#define INNER 32
#define SCALE 1000
#endif
#include "hal/symbex_hal.h"
#include "layers/dense/symbex_dense.h"
#include "ops/symbex_and.h"

#if defined(CONFIG_IDF_TARGET_ESP32S3)
#define VERT_PIE 1
#endif

typedef struct { uint32_t w[4]; } __attribute__((aligned(16))) vplane_t;
#define EP 12
#include "bench_data.h"

/* ================= the conv engine's vertical counting (symbex_bool_vertical.c), unchanged ================= */
struct VCount {
    vplane_t *ba, *bb;
#if !defined(VERT_PIE)
    vplane_t q[3];
#endif
};
#if defined(VERT_PIE)
#define CSA(A_, B_, C_, T_) "ee.xorq " T_ ", " A_ ", " B_ " \n\t ee.andq " B_ ", " A_ ", " B_ " \n\t ee.xorq " A_ ", " T_ ", " C_ " \n\t" \
                            "ee.andq " T_ ", " T_ ", " C_ " \n\t ee.orq " B_ ", " B_ ", " T_ " \n\t"
#define HS8(L0, L1, L2, L3, L4, L5, L6, L7) L0 L1 CSA("q0","q3","q4","q5") L2 L3 CSA("q0","q4","q5","q6") CSA("q1","q3","q4","q5") \
    L4 L5 CSA("q0","q4","q5","q6") L6 L7 CSA("q0","q5","q6","q7") CSA("q1","q4","q5","q6") CSA("q2","q3","q4","q5") \
    "ee.vst.128.ip q3, %[o], 16 \n\t"
#define CL(Q) "ee.vld.128.ip " Q ", %[p], 16 \n\t"
SYMBEX_FAST_CODE static void hs_gather(struct VCount* S, const uint8_t* p, const uint16_t* l, int groups, vplane_t* o) {
    (void)S;
    uint32_t d0, d1;
    for (int g = 0; g < groups; g++)
        __asm__ __volatile__(
            "l16ui %[d0], %[l], 0 \n\t l16ui %[d1], %[l], 2 \n\t"
            "ee.vld.128.xp q3, %[p], %[d0] \n\t ee.vld.128.xp q4, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 4 \n\t l16ui %[d1], %[l], 6 \n\t"
            CSA("q0","q3","q4","q5")
            "ee.vld.128.xp q4, %[p], %[d0] \n\t ee.vld.128.xp q5, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 8 \n\t l16ui %[d1], %[l], 10 \n\t"
            CSA("q0","q4","q5","q6") CSA("q1","q3","q4","q5")
            "ee.vld.128.xp q4, %[p], %[d0] \n\t ee.vld.128.xp q5, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 12 \n\t l16ui %[d1], %[l], 14 \n\t"
            CSA("q0","q4","q5","q6")
            "ee.vld.128.xp q5, %[p], %[d0] \n\t ee.vld.128.xp q6, %[p], %[d1] \n\t"
            "addi %[l], %[l], 16 \n\t"
            CSA("q0","q5","q6","q7") CSA("q1","q4","q5","q6") CSA("q2","q3","q4","q5")
            "ee.vst.128.ip q3, %[o], 16 \n\t"
            : [p] "+r"(p), [l] "+r"(l), [o] "+r"(o), [d0] "=&r"(d0), [d1] "=&r"(d1) : : "memory");
}
SYMBEX_FAST_CODE static void hs_contig(struct VCount* S, const vplane_t* p, int groups, vplane_t* o) {
    (void)S;
    for (int g = 0; g < groups; g++)
        __asm__ __volatile__(HS8(CL("q3"), CL("q4"), CL("q4"), CL("q5"), CL("q4"), CL("q5"), CL("q5"), CL("q6"))
                             : [p] "+r"(p), [o] "+r"(o) : : "memory");
}
static inline void zero_planes(vplane_t* p, int count) {
    if (count <= 0) return;
    __asm__ __volatile__("ee.zero.q q3");
    while (count--) __asm__ __volatile__("ee.vst.128.ip q3, %0, 16" : "+r"(p) : : "memory");
}
static inline void zero3(struct VCount* S) { (void)S; __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t ee.zero.q q2"); }
static inline void store3(struct VCount* S, vplane_t* d) {
    (void)S;
    __asm__ __volatile__("ee.vst.128.ip q0, %0, 16 \n\t ee.vst.128.ip q1, %0, 16 \n\t ee.vst.128.ip q2, %0, 16" : "+r"(d) : : "memory");
}
#define E3(KIN, KOUT) "ee.vld.128.ip q3, %[a], 16 \n\t ee.vld.128.ip q4, %[b], 16 \n\t ee.vld.128.ip q5, %[x], 16 \n\t" \
    "ee.xorq q6, q3, q4 \n\t ee.andq " KOUT ", q3, q4 \n\t ee.andq q3, q6, q5 \n\t ee.xorq q6, q6, q5 \n\t" \
    "ee.orq " KOUT ", " KOUT ", q3 \n\t ee.xorq q7, q6, " KIN " \n\t ee.andq q6, q6, " KIN " \n\t" \
    "ee.andq q7, q7, q0 \n\t ee.orq q0, q6, q7 \n\t"
SYMBEX_FAST_CODE static void epilogue3(const vplane_t* a, const vplane_t* b, const vplane_t* x, vplane_t* out) {
    __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t"
                         E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1")
                         E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1")
                         "ee.xorq q2, q1, q0 \n\t ee.vst.128.ip q2, %[o], 0 \n\t"
                         : [a] "+r"(a), [b] "+r"(b), [x] "+r"(x), [o] "+r"(out) : : "memory");
}
#else
static inline void csa(vplane_t* a, vplane_t* b, const vplane_t* c) {
    for (int w = 0; w < 4; w++) {
        const uint32_t t = a->w[w] ^ b->w[w];
        b->w[w] = (a->w[w] & b->w[w]) | (t & c->w[w]);
        a->w[w] = t ^ c->w[w];
    }
}
SYMBEX_FAST_CODE static void hs8(vplane_t* Q, const vplane_t* x0, const vplane_t* x1, const vplane_t* x2, const vplane_t* x3,
                                 const vplane_t* x4, const vplane_t* x5, const vplane_t* x6, const vplane_t* x7, vplane_t* o) {
    vplane_t a = *x0, b = *x1, c, d, e, f;
    csa(&Q[0], &a, &b);
    c = *x2; d = *x3;
    csa(&Q[0], &c, &d);
    csa(&Q[1], &a, &c);
    c = *x4; d = *x5;
    csa(&Q[0], &c, &d);
    e = *x6; f = *x7;
    csa(&Q[0], &e, &f);
    csa(&Q[1], &c, &e);
    csa(&Q[2], &a, &c);
    *o = a;
}
SYMBEX_FAST_CODE static void hs_gather(struct VCount* S, const uint8_t* p, const uint16_t* l, int groups, vplane_t* o) {
    for (int g = 0; g < groups; g++, l += 8, o++) {
        const vplane_t* x[8];
        for (int i = 0; i < 8; i++) { x[i] = (const vplane_t*)p; p += l[i]; }
        hs8(S->q, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], o);
    }
}
SYMBEX_FAST_CODE static void hs_contig(struct VCount* S, const vplane_t* p, int groups, vplane_t* o) {
    for (int g = 0; g < groups; g++, p += 8, o++) hs8(S->q, p, p + 1, p + 2, p + 3, p + 4, p + 5, p + 6, p + 7, o);
}
static inline void zero_planes(vplane_t* p, int count) { if (count > 0) memset(p, 0, (size_t)count * sizeof *p); }
static inline void zero3(struct VCount* S) { memset(S->q, 0, sizeof S->q); }
static inline void store3(struct VCount* S, vplane_t* d) { d[0] = S->q[0]; d[1] = S->q[1]; d[2] = S->q[2]; }
SYMBEX_FAST_CODE static void epilogue3(const vplane_t* a, const vplane_t* b, const vplane_t* x, vplane_t* out) {
    for (int w = 0; w < 4; w++) {
        uint32_t chain = 0, kin = 0;
        for (int i = 0; i < EP; i++) {
            const uint32_t A = a[i].w[w], B = b[i].w[w], X = x[i].w[w], ab = A ^ B, s = ab ^ X;
            const uint32_t kout = (A & B) | (ab & X);
            chain = (s & kin) | ((s ^ kin) & chain);
            kin = kout;
        }
        out->w[w] = kin ^ chain;
    }
}
#endif
SYMBEX_FAST_CODE static void count_finish(struct VCount* S, int n, vplane_t* bits, int need) {
    store3(S, bits);
    int lvl = 1;
    vplane_t *src = S->ba, *dst = S->bb;
    while (n > 0) {
        const int g = (n + 7) >> 3;
        zero_planes(src + n, g * 8 - n);
        zero3(S); hs_contig(S, src, g, dst); store3(S, bits + 3 * lvl); lvl++;
        if (n < 8) break;
        vplane_t* t = src; src = dst; dst = t; n = g;
    }
    zero_planes(bits + 3 * lvl, need - 3 * lvl);
}

/* the conv engine's plane list (make_list), here built from the INPUT bits: the inputs that are 1 select the planes */
typedef struct { uint16_t d[7]; uint16_t first, last, n; } ListByte;
static ListByte LT[256];
static void list_table(ListByte* lt) {
    for (int b = 0; b < 256; b++) {
        int c = 0, prev = 0;
        memset(&lt[b], 0, sizeof lt[b]);
        for (int i = 0; i < 8; i++)
            if ((b >> i) & 1) {
                if (c == 0) lt[b].first = (uint16_t)(i * 16); else lt[b].d[c - 1] = (uint16_t)((i - prev) * 16);
                prev = i; c++;
            }
        lt[b].n = (uint16_t)c; lt[b].last = (uint16_t)(prev * 16);
    }
}
static int make_list(const uint32_t* row, int K, const ListByte* lt, uint16_t* d, uint32_t* start) {
    const uint8_t* rb = (const uint8_t*)row;
    const int bytes = K / 8;
    uint16_t* q = d - 1;
    uint32_t prev = 0;
    for (int j = 0; j < bytes; j++) {
        const ListByte* e = &lt[rb[j]];
        const uint32_t base = (uint32_t)j * 128, c = e->n;
        q[0] = (uint16_t)(base + e->first - prev);
        q[1] = e->d[0]; q[2] = e->d[1]; q[3] = e->d[2]; q[4] = e->d[3]; q[5] = e->d[4]; q[6] = e->d[5]; q[7] = e->d[6];
        q += c;
        const uint32_t last = base + e->last;
        prev = c ? last : prev;
    }
    const int n = (int)(q - (d - 1));
    *start = n ? d[-1] : 0;
    const int n8 = (n + 7) & ~7;
    if (n) {
        d[n - 1] = (uint16_t)(n8 > n ? (uint32_t)K * 16 - prev : 0);
        for (int i = n; i < n8; i++) d[i] = 0;
    }
    return n;
}

/* ================= the vertical dense layer ================= */
static uint16_t list_buf[MAX_IN + 16];
static vplane_t ba[MAX_IN / 8 + 16], bb[MAX_IN / 64 + 16], cnt[16], sc[EP];

/* x: n_in bits (n_in a multiple of 8, below 2048); wt: G groups of n_in + 1 planes; e: G x EP planes of 2048 - E;
   out: G x 4 words, bit n % 32 of word n / 32 = neuron n (the usual bit layout) */
SYMBEX_FAST_CODE static void dense_vertical(const uint32_t* x, int n_in, int groups, const vplane_t* wt,
                                            const vplane_t* e, uint32_t* out) {
    uint32_t start;
    const int n = make_list(x, n_in, LT, list_buf + 1, &start), g8 = (n + 7) >> 3;
    const uint32_t a = 2048u - (uint32_t)n;                        /* 2048 - count(x): the same in every lane */
    for (int i = 0; i < EP; i++) { const uint32_t v = ((a >> i) & 1u) ? 0xFFFFFFFFu : 0u; sc[i].w[0] = sc[i].w[1] = sc[i].w[2] = sc[i].w[3] = v; }
    struct VCount S; S.ba = ba; S.bb = bb;
    cnt[0].w[0] = cnt[0].w[1] = cnt[0].w[2] = cnt[0].w[3] = 0;     /* 2 c: the count planes one place up */
    for (int g = 0; g < groups; g++) {
        zero3(&S);
        hs_gather(&S, (const uint8_t*)(wt + (uint32_t)g * (n_in + 1)) + start, list_buf + 1, g8, ba);
        count_finish(&S, g8, cnt + 1, EP - 1);
        epilogue3(sc, e + g * EP, cnt, (vplane_t*)(out + 4 * g));
    }
}

/* ================= timing and checks ================= */
#define REPS 7
static uint32_t xin[MAX_IN / 32] __attribute__((aligned(16)));
static uint32_t bits[MAX_GROUPS * 4] __attribute__((aligned(16))), ref_bits[MAX_GROUPS * 4];
static int32_t raw[MAX_OUT];
static int cs, cx;

static void run_xnor(void) { symbex_dense_forward_bits(xin, THRS[cs], bits, &LAYERS[cs]); }
static void run_and(void) {
    const struct SymbexLayerData* L = &LAYERS[cs];
    symbex_and_raw(xin, L->in_features, L->weights, (L->in_features + 31) / 32, L->out_features, WCOUNT[cs], raw);
    symbex_dense_threshold(raw, THRS[cs], bits, L->out_features);
}
static void run_vertical(void) { dense_vertical(xin, LAYERS[cs].in_features, (LAYERS[cs].out_features + 127) / 128, WT[cs], EV[cs], bits); }

static uint32_t best(void (*f)(void)) {
    uint32_t b = 0xFFFFFFFFu;
    for (int r = 0; r < REPS; r++) {
        const uint32_t t0 = NOW();
        for (int k = 0; k < INNER; k++) f();
        const uint32_t t = (NOW() - t0) * SCALE / INNER;
        if (t < b) b = t;
    }
    return b;
}
static void reference(void) {
    const struct SymbexLayerData* L = &LAYERS[cs];
    const int words = (L->in_features + 31) / 32;
    memset(ref_bits, 0, sizeof ref_bits);
    for (int n = 0; n < L->out_features; n++) {
        int32_t s = 0;
        for (int i = 0; i < L->in_features; i++)
            s += ((((xin[i / 32] >> (i % 32)) ^ (L->weights[n * words + i / 32] >> (i % 32))) & 1u) ? -1 : 1);
        if (s >= THRS[cs][n]) ref_bits[n / 32] |= 1u << (n % 32);
    }
}
static int same(void) {
    const int n_out = LAYERS[cs].out_features;
    for (int n = 0; n < n_out; n++) if (((bits[n / 32] ^ ref_bits[n / 32]) >> (n % 32)) & 1u) return 0;
    return 1;
}

static void bench_once(void) {
    OUT("DENSE VERTICAL BENCH ready (%s)\n", WHERE);
    list_table(LT);
    for (cs = 0; cs < N_SHAPES; cs++)
        for (cx = 0; cx < N_DENS; cx++) {
            const int n_in = LAYERS[cs].in_features, n_out = LAYERS[cs].out_features;
            memcpy(xin, XS[cs][cx], (size_t)n_in / 8);
            int ones = 0;
            for (int i = 0; i < n_in; i++) ones += (xin[i / 32] >> (i % 32)) & 1u;
            reference();
            uint32_t c;
            memset(bits, 0, sizeof bits); c = best(run_xnor);
            OUT("R:%d|%d|%d|xnor|%lu|%d\n", n_in, n_out, ones, (unsigned long)c, same());
            memset(bits, 0, sizeof bits); c = best(run_and);
            OUT("R:%d|%d|%d|and|%lu|%d\n", n_in, n_out, ones, (unsigned long)c, same());
            memset(bits, 0, sizeof bits); c = best(run_vertical);
            OUT("R:%d|%d|%d|vertical|%lu|%d\n", n_in, n_out, ones, (unsigned long)c, same());
        }
    OUT("DONE\n");
}

#if defined(ESP_PLATFORM)
void app_main(void) {
    for (;;) {
        bench_once();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
#elif defined(SYMBEX_HOST)
int main(void) { bench_once(); return 0; }
#else
void setup() { Serial.begin(115200); }
void loop() {
    bench_once();
    delay(2000);
}
#endif
"""


def planes_of(bits_rows):
    """(128, n) lane bits -> n planes of 4 words (bit l % 32 of word l / 32 = lane l)"""
    b = bits_rows.astype(np.uint64).reshape(4, 32, -1)
    return (b << np.arange(32, dtype=np.uint64)[None, :, None]).sum(1).T      # (n, 4)


def pack_rows(b):
    """(rows, n) bits -> rows x ceil(n / 32) words, bit i % 32 of word i / 32"""
    words = (b.shape[1] + 31) // 32
    p = np.zeros((b.shape[0], words * 32), np.uint64)
    p[:, :b.shape[1]] = b
    return (p.reshape(b.shape[0], words, 32) << np.arange(32, dtype=np.uint64)).sum(-1)


def write_data(path, where, weights="flash"):
    rng = np.random.default_rng(2026)
    const = weights == "flash"
    max_in, max_out = max(s[0] for s in SHAPES), max(s[1] for s in SHAPES)
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define WHERE \"{where}\"\n#define N_SHAPES {len(SHAPES)}\n#define N_DENS {len(DENSITIES)}\n"
                f"#define MAX_IN {max_in}\n#define MAX_OUT {max_out}\n#define MAX_GROUPS {(max_out + 127) // 128}\n")
        for s, (n_in, n_out) in enumerate(SHAPES):
            assert n_in % 8 == 0 and n_in < 2048
            groups = (n_out + 127) // 128
            b = rng.integers(0, 2, (n_out, n_in))                             # weight bits, 1 = +1
            f.write(c_array("uint32_t", f"W{s}", pack_rows(b), const, 8))
            pw = b.sum(1)
            f.write(c_array("int32_t", f"WC{s}", pw, const))
            xs = [(rng.random(n_in) < d).astype(np.int64) for d in DENSITIES]
            for k, x in enumerate(xs):
                f.write(c_array("uint32_t", f"X{s}_{k}", pack_rows(x[None, :]), True, 8))
            # thresholds around the scores of the first input, so both bit values appear
            score = ((2 * b - 1) * (2 * xs[0] - 1)).sum(1)
            thr = score + rng.integers(-6, 7, n_out)
            f.write(c_array("int32_t", f"T{s}", thr, const))
            # vertical data: transposed weights (+ a zero plane per group) and the planes of 2048 - E
            lanes = np.zeros((groups * 128, n_in + 1), np.int64)
            lanes[:n_out, :n_in] = b
            wt = np.concatenate([planes_of(lanes[g * 128:(g + 1) * 128]) for g in range(groups)])
            f.write(c_array("uint32_t", f"WT{s}", wt, const, 8))
            e = np.full(groups * 128, n_in + 1, np.int64)                      # padding lanes: never 1
            e[:n_out] = -((-(thr - n_in + 2 * pw)) // 2)                       # ceil((thr - n_in + 2 count(w)) / 2)
            e = np.clip(e, -n_in, n_in + 1)                                    # 2 c - count(x) is in [-n_in, n_in]
            v = 2048 - e
            assert v.min() >= 0 and v.max() < 4096
            ep = np.stack([(v >> i) & 1 for i in range(EP)])                   # (EP, lanes)
            ev = np.concatenate([planes_of(ep[:, g * 128:(g + 1) * 128].T) for g in range(groups)])
            f.write(c_array("uint32_t", f"EV{s}", ev, const, 8))
        f.write("static const struct SymbexLayerData LAYERS[N_SHAPES] = {"
                + ", ".join(f"{{{i}, {o}, 1, W{s}}}" for s, (i, o) in enumerate(SHAPES)) + "};\n")
        for name in ("T", "WC"):
            f.write(f"static const int32_t* const {'THRS' if name == 'T' else 'WCOUNT'}[N_SHAPES] = {{"
                    + ", ".join(f"{name}{s}" for s in range(len(SHAPES))) + "};\n")
        f.write("static const vplane_t* const WT[N_SHAPES] = {" + ", ".join(f"(const vplane_t*)WT{s}" for s in range(len(SHAPES))) + "};\n")
        f.write("static const vplane_t* const EV[N_SHAPES] = {" + ", ".join(f"(const vplane_t*)EV{s}" for s in range(len(SHAPES))) + "};\n")
        f.write("static const uint32_t* const XS[N_SHAPES][N_DENS] = {"
                + ", ".join("{" + ", ".join(f"X{s}_{k}" for k in range(len(DENSITIES))) + "}" for s in range(len(SHAPES)))
                + "};\n")


def write_project(project, port, board, weights):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        import shutil
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    files, get = copy_lib(src, None)
    if board in BOARD_HAL:
        files.update([BOARD_HAL[board]])
    for name, rel in files.items():
        open(os.path.join(src, name), "wb").write(get(os.path.join("src", *rel)))
    where = f"weights in {weights}"
    write_data(os.path.join(src, "bench_data.h"), where, weights)
    if board.startswith("pico2"):
        w("platformio.ini", f"[env:{board}]\nplatform = raspberrypi\nboard = {BOARDS[board]}\nframework = arduino\n"
                            f"board_build.core = earlephilhower\nupload_protocol = picotool\nupload_port = {port}\n"
                            + ("board_build.mcu = rp2350-riscv\n" if board == "pico2_riscv" else "")
                            + "build_flags = -Isrc/include -O2\nbuild_unflags = -Os\n")
        w("src/main.cpp", FIRMWARE)
        return where
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
                        "project(dense_vertical_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in files)
                            + ' INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)
    return where


def read_results(port, board, where):
    import time
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = [], False, time.time()
    while time.time() - t0 < 90:
        line = s.readline().decode(errors="replace").strip()
        if "DENSE VERTICAL BENCH ready" in line:
            if f"({where})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected {where}")
            ready, rows = True, []
        elif ready and line.startswith("R:"):
            n_in, n_out, ones, method, t, ok = line[2:].split("|")
            rows.append((int(n_in), int(n_out), int(ones), method, int(t), ok == "1"))
        elif ready and line == "DONE":
            s.close()
            return rows
    s.close()
    sys.exit("[X] no results from the board in 90 s")


def host(weights):
    """the same firmware on this PC (gcc -O2, the generic HAL, portable vertical code): every method vs the reference"""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "project"))
        src = os.path.join(d, "project", "src")
        os.makedirs(src)
        files, get = copy_lib(src, None)
        for name, rel in files.items():
            open(os.path.join(src, name), "wb").write(get(os.path.join("src", *rel)))
        write_data(os.path.join(src, "bench_data.h"), f"weights in {weights}", weights)
        open(os.path.join(src, "main.c"), "w").write(FIRMWARE)
        exe = os.path.join(d, "bench")
        subprocess.run(["gcc", "-O2", "-DSYMBEX_HOST", "-I", src, "-I", os.path.join(src, "include"), "-o", exe,
                        os.path.join(src, "main.c")] + [os.path.join(src, n) for n in files], check=True)
        out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    rows = [r[2:].split("|") for r in out.splitlines() if r.startswith("R:")]
    print(f"  {'shape':<11}{'ones':>6}  {'method':<10}{'ns':>9}  equal to the reference")
    for n_in, n_out, ones, m, t, ok in rows:
        print(f"  {f'{n_in}->{n_out}':<11}{ones:>6}  {m:<10}{t:>9}  {'OK' if ok == '1' else 'NO'}")
    if not rows or not all(r[5] == "1" for r in rows):
        sys.exit("[X] a method differs from its reference")
    print("[*] every method equals the reference on this PC")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(TICKS), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="where the weights and thresholds are")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    ap.add_argument("--host", action="store_true", help="build and run on this PC (gcc, portable C): the exactness check")
    a = ap.parse_args()
    if a.host:
        return host(a.weights)
    tag = a.board + ("_wram" if a.weights == "ram" else "")
    project = os.path.join(HERE, "runs", f"dense_vertical_bench_{tag}")
    where = f"weights in {a.weights}"
    if not a.no_upload:
        where = write_project(project, a.port, a.board, a.weights)
        flash(project, verbose=False, what=where)
    rows = read_results(a.port, a.board, where)
    unit, per_us = TICKS[a.board]
    print(f"\n{a.board}, {where} (best of 7)")
    print(f"  {'shape':<11}{'ones':>6}  {'method':<10}{unit:>9}{'us':>9}{'xnor / it':>11}  equal to the reference")
    base = {(r[0], r[1], r[2]): r[4] for r in rows if r[3] == "xnor"}
    for n_in, n_out, ones, m, t, ok in rows:
        print(f"  {f'{n_in}->{n_out}':<11}{ones:>6}  {m:<10}{t:>9}{t / per_us:>9.2f}{base[(n_in, n_out, ones)] / t:>10.2f}x"
              f"  {'OK' if ok else 'NO'}")
    if not all(r[5] for r in rows):
        sys.exit("[X] a method differs from its reference: the times are not trusted")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"dense_vertical_bench_{tag}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, weights=a.weights, reps="best of 7",
                       rows=[dict(n_in=i, n_out=o, ones=k, method=m, **({unit: t}), us=round(t / per_us, 3),
                                  equal_reference=ok) for i, o, k, m, t, ok in rows]), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
