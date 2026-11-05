"""
s3_vertical_count.py -- counting bits on the ESP32-S3 without popcount: vertical (bit-sliced) sums against today's
pie_swar and against the int8 multiply-accumulate unit that ESP-DL uses. A separate microbenchmark.

    python s3_vertical_count.py --port /dev/ttyACM0
    python s3_vertical_count.py --port /dev/ttyACM0 --no-upload      # the firmware is already on the board

Every method computes the same binary layer: P positions x K input bits x O outputs, out bit = matches(x, w) >= T[o]
(matches = K - popcount(x XOR w)), and its output is compared bit by bit with a scalar C reference on the chip.

    pie_swar      today's lib_symbex S3 count: per (position, output), XOR 128 bits, count inside every byte with
                  vector SWAR, add the 16 bytes with one ee.vmulas.u8.accx (horizontal: one count at a time)
    int8          the same layer as an int8 network with weights and inputs in {-1, +1}: ee.vsmulas.s8.qacc.ld.incp,
                  16 outputs x 1 input per instruction with the next weights loaded in the same instruction (the
                  ESP-DL inner loop), then the 16 accumulators are read and compared with the thresholds
    int8_mac      the int8 loop alone (QACC stored, no thresholds): the int8 unit's ceiling; its output is not checked
    vert_out      vertical, lanes = outputs (dense layers): bit o of a 128-bit plane is output o. AND form: only the
                  weight planes of the input bits that are 1 are summed (a list built at run time, about K/2 planes),
                  with carry-save adders (Harley-Seal, logic only: and/or/xor; the list loads pipelined). Threshold compared in bit-slice, the
                  output comes out already packed. Weights take 1 bit each (transposed).
    vert_pos      vertical, lanes = positions (convolutions with few output channels): bit p of a plane is position p.
                  Per output, the input planes of the weights that are 1 are summed (a fixed list of offsets per
                  output, uint16: 16 bits per weight that is 1, about 8 bits per weight). The input is given as K
                  planes of 128 positions (im2col already transposed; how a real convolution builds them, borders
                  included, is not measured here).

Cycles are the best of 7 runs of the whole layer, including everything each method needs at run time (lists, input
popcounts, threshold planes). MAC/cycle = P * K * O / cycles.

Two more measurements of the vertical methods:
    breakdown     the same layer stopped after each stage (prep: lists / input popcounts and threshold planes; sum: the
                  Harley-Seal pass over the planes; levels: the eights counted into the binary count; threshold: the
                  bit-sliced compare), so the time of every stage is the difference of two runs
    _flash        the weights (transposed planes or offset lists, and the threshold planes) in a flash partition read
                  through the cache, as a model is usually deployed; the plain rows have them in RAM
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "runs", "s3_vertical_count")
RESULTS = os.path.join(HERE, "results", "s3_vertical_count.json")
# (kind, P, K, O): two dense layers (one input vector), three 3x3 convolutions as matrices (128 positions,
# 16 / 32 / 64 input channels x 9, 32 outputs)
SHAPES = [("dense", 1, 128, 512), ("dense", 1, 512, 256),
          ("conv", 128, 144, 32), ("conv", 128, 288, 32), ("conv", 128, 576, 32)]

FIRMWARE = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"

typedef struct { uint32_t w[4]; } __attribute__((aligned(16))) plane_t;
#define NB 15                       /* counter planes kept (bits 0..14); the epilogue uses 0..10 */
#define EB 11                       /* bits of the threshold epilogue: every value stays below 2048 */

static int P, K, O, nw, st;
static uint32_t *X, *W, *REF, *Y;   /* bits: X[P][st], W[O][st], outputs [P][O/32] (vert_pos: [O][4]) */
static int32_t *T, *T2, *PCW;
static int8_t *X8, *W8;
static uint8_t* QB;
static int MAP[16];
static plane_t *WT, *XT, *BA, *BB, *BITS, *PX, *N2, *N0, *N1, ONESQ;
static uint16_t *D, *D3;
static uint32_t *S3;
static int *NN3, *N83;
static volatile uint32_t sink;
static int STAGE = 3;               /* vertical methods: 0 prep, 1 + sum, 2 + levels, 3 + threshold (the whole layer) */
static const plane_t *WTu, *N2u;    /* the weights in use: in RAM or in the flash partition */
static const uint16_t* D3u;

static uint32_t rs = 2463534242u;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static void* A(size_t n) {
    void* p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) { printf("ERROR: no RAM for %u bytes\n", (unsigned)n); fflush(stdout); while (1) vTaskDelay(portMAX_DELAY); }
    memset(p, 0, n); return p;
}
#define LOADQ(REG, PTR) { const void* p_ = (PTR); __asm__ __volatile__("ee.vld.128.ip " REG ", %0, 0" : "+r"(p_) : : "memory"); }

/* ================= pie_swar (today) ================= */
static const uint8_t ONES8[16] __attribute__((aligned(16))) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static const uint8_t M11[16] __attribute__((aligned(16))) = {0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11};
static const uint8_t M0F[16] __attribute__((aligned(16))) = {0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F};
static inline int cnt_pie_swar(const uint32_t* a, const uint32_t* b, int words) {
    __asm__ __volatile__("ee.zero.accx");
    for (int g = 0; g < words / 4; g++)
        __asm__ __volatile__("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.xorq q2, q0, q1 \n\t"
                             "ee.andq q4, q2, q5 \n\t"
                             "ssai 1 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ssai 2 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ssai 3 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ssai 4 \n\t ee.vsr.32 q3, q4 \n\t ee.andq q3, q3, q6 \n\t ee.andq q4, q4, q6 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ee.vmulas.u8.accx q4, q7 \n\t"
                             : "+r"(a), "+r"(b) : : "memory");
    uint32_t v; __asm__ __volatile__("rur.accx_0 %0" : "=r"(v)); return (int)v;
}
static void run_pie_swar(void) {
    LOADQ("q5", M11); LOADQ("q6", M0F); LOADQ("q7", ONES8);
    const int ow = O / 32;
    memset(Y, 0, (size_t)P * ow * 4);
    for (int p = 0; p < P; p++)
        for (int o = 0; o < O; o++)
            if (K - cnt_pie_swar(&X[p * st], &W[o * st], st) >= T[o]) Y[p * ow + o / 32] |= 1u << (o % 32);
}

/* ================= int8: ee.vsmulas.s8.qacc.ld.incp (the ESP-DL inner loop) ================= */
#define VS(QW, S) "ee.vsmulas.s8.qacc.ld.incp " QW ", %1, " QW ", q0, " #S " \n\t"
#define VS16 "ee.vld.128.ip q0, %0, 16 \n\t" VS("q1",0) VS("q2",1) VS("q1",2) VS("q2",3) VS("q1",4) VS("q2",5) \
             VS("q1",6) VS("q2",7) VS("q1",8) VS("q2",9) VS("q1",10) VS("q2",11) VS("q1",12) VS("q2",13) VS("q1",14) VS("q2",15)
static inline void int8_block(const int8_t* xp, const int8_t* wp) {
    __asm__ __volatile__("ee.zero.qacc \n\t ee.vld.128.ip q1, %0, 16 \n\t ee.vld.128.ip q2, %0, 16" : "+r"(wp) : : "memory");
    for (int c = 0; c < K / 16; c++) __asm__ __volatile__(VS16 : "+r"(xp), "+r"(wp) : : "memory");
    uint8_t* q = QB;
    __asm__ __volatile__("ee.st.qacc_l.l.128.ip %0, 16 \n\t ee.st.qacc_l.h.32.ip %0, 16 \n\t"
                         "ee.st.qacc_h.l.128.ip %0, 16 \n\t ee.st.qacc_h.h.32.ip %0, 4" : "+r"(q) : : "memory");
}
static inline int32_t lane(int l) {             /* 16 lanes of 20 bits: 8 in QACC_L (bytes 0..19), 8 in QACC_H (32..51) */
    const uint8_t* b = QB + (l < 8 ? 0 : 32); int bit = 20 * (l & 7), by = bit >> 3;
    uint32_t v = ((uint32_t)b[by] | (uint32_t)b[by + 1] << 8 | (uint32_t)b[by + 2] << 16) >> (bit & 7);
    return (int32_t)(v << 12) >> 12;
}
static void run_int8(int thresholds) {
    const int ow = O / 32;
    if (thresholds) memset(Y, 0, (size_t)P * ow * 4);
    for (int p = 0; p < P; p++)
        for (int b = 0; b < O / 16; b++) {
            int8_block(&X8[p * K], &W8[(size_t)b * K * 16]);
            if (!thresholds) continue;
            uint32_t m = 0;
            for (int l = 0; l < 16; l++) { int o = MAP[l]; if (lane(l) >= T2[b * 16 + o]) m |= 1u << o; }
            Y[p * ow + b / 2] |= m << (16 * (b & 1));
        }
}
static int calibrate_qacc(void) {               /* which accumulator lane holds which output: weight j+1 on output j */
    int8_t* x = A(16); int8_t* w = A(16 * 16 + 32);
    x[0] = 1; for (int j = 0; j < 16; j++) w[j] = (int8_t)(j + 1);
    int k = K; K = 16; int8_block(x, w); K = k;
    int seen = 0;
    for (int l = 0; l < 16; l++) { int v = lane(l); if (v < 1 || v > 16) seen = -1; else { MAP[l] = v - 1; if (seen >= 0) seen |= 1 << (v - 1); } }
    heap_caps_free(x); heap_caps_free(w);
    if (seen != 0xFFFF) {
        printf("ERROR: unexpected QACC layout:"); for (int i = 0; i < 52; i++) printf(" %02x", QB[i]); printf("\n");
        return 0;
    }
    return 1;
}

/* ================= vertical counting: Harley-Seal carry-save adders on 128-bit planes =================
   q0 = ones, q1 = twos, q2 = fours of the running count (kept across calls); q3..q7 temporaries.
   CSA(a, b, c, t): a <- a ^ b ^ c (sum), b <- majority(a, b, c) (carry), 5 logic instructions. */
#define CSA(A_, B_, C_, T_) "ee.xorq " T_ ", " A_ ", " B_ " \n\t ee.andq " B_ ", " A_ ", " B_ " \n\t ee.xorq " A_ ", " T_ ", " C_ " \n\t" \
                            "ee.andq " T_ ", " T_ ", " C_ " \n\t ee.orq " B_ ", " B_ ", " T_ " \n\t"
/* 8 input planes -> ones/twos/fours updated, one "eights" plane (weight 8) stored */
#define HS8(L0, L1, L2, L3, L4, L5, L6, L7) L0 L1 CSA("q0","q3","q4","q5") L2 L3 CSA("q0","q4","q5","q6") CSA("q1","q3","q4","q5") \
    L4 L5 CSA("q0","q4","q5","q6") L6 L7 CSA("q0","q5","q6","q7") CSA("q1","q4","q5","q6") CSA("q2","q3","q4","q5") \
    "ee.vst.128.ip q3, %[o], 16 \n\t"
#define CL(Q) "ee.vld.128.ip " Q ", %[p], 16 \n\t"                                         /* contiguous load */

/* gather, pipelined (Eden's v2): the deltas are loaded two at a time, ahead of the loads that use them, so the
   scalar load latency is hidden behind the CSAs. Same loads in the same order as HS8 with GL. */
static void hs_gather(const uint8_t* p, const uint16_t* l, int groups, plane_t* o) {
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
static void hs_contig(const plane_t* p, int groups, plane_t* o) {
    for (int g = 0; g < groups; g++)
        __asm__ __volatile__(HS8(CL("q3"), CL("q4"), CL("q4"), CL("q5"), CL("q4"), CL("q5"), CL("q5"), CL("q6"))
                             : [p] "+r"(p), [o] "+r"(o) : : "memory");
}
/* zero planes with the vector unit instead of memset (Eden's v2); q3 is a temporary everywhere it is called */
static inline void zero_planes(plane_t* p, int count) {
    if (count <= 0) return;
    __asm__ __volatile__("ee.zero.q q3");
    while (count--) __asm__ __volatile__("ee.vst.128.ip q3, %0, 16" : "+r"(p) : : "memory");
}
static inline void zero3(void) { __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t ee.zero.q q2"); }
static inline void store3(plane_t* d) {
    __asm__ __volatile__("ee.vst.128.ip q0, %0, 16 \n\t ee.vst.128.ip q1, %0, 16 \n\t ee.vst.128.ip q2, %0, 16" : "+r"(d) : : "memory");
}
/* level 1 already ran (its counters in q0..q2, n eights planes in BA); the eights are counted again level after
   level, each level giving 3 more bits: bits = the count in binary, one plane per bit */
static void count_finish(int n, plane_t* bits) {
    store3(bits);
    int lvl = 1; plane_t *src = BA, *dst = BB;
    while (n > 0) {
        int g = (n + 7) >> 3;
        zero_planes(src + n, g * 8 - n);
        zero3(); hs_contig(src, g, dst); store3(bits + 3 * lvl); lvl++;
        if (n < 8) break;
        plane_t* t = src; src = dst; dst = t; n = g;
    }
    zero_planes(bits + 3 * lvl, NB - 3 * lvl);
}
/* out = (count + N >= M) per lane, N a plane per bit, M a constant: bit-sliced add, then compare from the low bit
   (ge = M_i ? s_i & ge : s_i | ge). q1 = carry, q2 = ge. */
#define EPI(OP) "ee.vld.128.ip q3, %0, 16 \n\t ee.vld.128.ip q4, %1, 16 \n\t ee.xorq q5, q3, q4 \n\t ee.xorq q6, q5, q1 \n\t" \
                "ee.andq q3, q3, q4 \n\t ee.andq q5, q5, q1 \n\t ee.orq q1, q3, q5 \n\t " OP " q2, q6, q2 \n\t"
static void epilogue(const plane_t* c, const plane_t* n, uint32_t m, plane_t* out) {
    const plane_t* po = &ONESQ;
    __asm__ __volatile__("ee.zero.q q1 \n\t ee.vld.128.ip q2, %0, 0" : "+r"(po) : : "memory");
    for (int i = 0; i < EB; i++) {
        if ((m >> i) & 1) __asm__ __volatile__(EPI("ee.andq") : "+r"(c), "+r"(n) : : "memory");
        else              __asm__ __volatile__(EPI("ee.orq")  : "+r"(c), "+r"(n) : : "memory");
    }
    __asm__ __volatile__("ee.vst.128.ip q2, %0, 0" : "+r"(out) : : "memory");
}
/* offsets (bytes) of the set bits of x, as a start offset and deltas, padded to a multiple of 8 with a zero plane */
static int build_list(const uint32_t* x, uint32_t stride, uint32_t zero_off, uint16_t* d, uint32_t* start, int* n8) {
    int n = 0; uint32_t prev = 0; *start = 0;
    for (int j = 0; j < nw; j++) {
        uint32_t v = x[j];
        while (v) {
            uint32_t off = (uint32_t)(j * 32 + __builtin_ctz(v)) * stride; v &= v - 1;
            if (n == 0) *start = off; else d[n - 1] = (uint16_t)(off - prev);
            prev = off; n++;
        }
    }
    *n8 = (n + 7) & ~7;
    if (n) { d[n - 1] = (uint16_t)(*n8 > n ? zero_off - prev : 0); for (int i = n; i < *n8; i++) d[i] = 0; }
    return n;
}
static inline void setbit(plane_t* pl, int lane) { pl->w[lane / 32] |= 1u << (lane % 32); }

/* ---- vert_out: lanes = outputs. With A = T + pcw - K = 2a + s and pcx = 2g + r (popcount of the input):
   matches >= T  <=>  and >= a + g + (s|r)  <=>  and + (1024 - a - (s|r)) >= 1024 + g.
   The planes of 1024 - a - (s|r) depend only on the weights and r: two sets per 128 outputs, made once. ---- */
static void prep_vert_out(void) {
    const int rsb = O / 8;                       /* bytes per transposed weight row */
    for (int o = 0; o < O; o++)
        for (int k = 0; k < K; k++)
            if ((W[o * st + k / 32] >> (k % 32)) & 1) WT[(k * rsb) / 16 + o / 128].w[(o % 128) / 32] |= 1u << (o % 32);
    for (int o = 0; o < O; o++) {
        int a_ = T[o] + PCW[o] - K, s = a_ & 1, a = (a_ - s) / 2;
        for (int r = 0; r < 2; r++) {
            uint32_t nv = (uint32_t)(1024 - a - (s | r));
            for (int i = 0; i < EB; i++) if ((nv >> i) & 1) setbit(&N2[((o / 128) * 2 + r) * EB + i], o % 128);
        }
    }
}
static void run_vert_out(void) {
    const int rsb = O / 8, groups = O / 128;
    for (int p = 0; p < P; p++) {
        uint32_t start; int n8;
        int n = build_list(&X[p * st], rsb, (uint32_t)K * rsb, D, &start, &n8);
        uint32_t m = 1024u + (uint32_t)(n >> 1); int r = n & 1;
        for (int g = 0; g < groups; g++) {
            if (STAGE < 1) continue;
            zero3();
            if (n8) hs_gather((const uint8_t*)WTu + g * 16 + start, D, n8 / 8, BA);
            if (STAGE < 2) continue;
            count_finish(n >= 8 ? n8 / 8 : 0, BITS);
            if (STAGE < 3) continue;
            epilogue(BITS, &N2u[(g * 2 + r) * EB], m, (plane_t*)&Y[p * (O / 32)] + g);
        }
    }
}

/* ---- vert_pos: lanes = 128 positions. Per output: A = T + pcw - K = 2a + s (constants), per lane pcx = 2g + r:
   s = 1: and + (1024 - g) >= 1025 + a;   s = 0: and + (1024 - g - r) >= 1024 + a.
   pcx is counted vertically once per tile (all K planes), and 1024 - floor(pcx/2), 1024 - ceil(pcx/2) built from it. ---- */
static void prep_vert_pos(void) {
    for (int p = 0; p < P; p++)
        for (int k = 0; k < K; k++)
            if ((X[p * st + k / 32] >> (k % 32)) & 1) setbit(&XT[k], p);
    for (int o = 0; o < O; o++) NN3[o] = build_list(&W[o * st], 16, (uint32_t)K * 16, &D3[(size_t)o * (K + 8)], &S3[o], &N83[o]);
}
static void run_vert_pos(void) {
    zero3(); hs_contig(XT, K / 8, BA); count_finish(K >= 8 ? K / 8 : 0, PX);
    for (int w = 0; w < 4; w++) {               /* N1 = ~f + 1, N0 = ~f + ~r (11 bits), then flip bit 10 */
        uint32_t c1 = ~0u, c0 = ~PX[0].w[w];
        for (int i = 0; i < EB; i++) {
            uint32_t a = ~(i < EB - 1 ? PX[i + 1].w[w] : 0u), s1 = a ^ c1, s0 = a ^ c0;
            c1 &= a; c0 &= a;
            if (i == EB - 1) { s1 = ~s1; s0 = ~s0; }
            N1[i].w[w] = s1; N0[i].w[w] = s0;
        }
    }
    for (int o = 0; o < O; o++) {
        int a_ = T[o] + PCW[o] - K, s = a_ & 1, a = (a_ - s) / 2;
        if (STAGE < 1) continue;
        zero3();
        if (N83[o]) hs_gather((const uint8_t*)XT + S3[o], &D3u[(size_t)o * (K + 8)], N83[o] / 8, BA);
        if (STAGE < 2) continue;
        count_finish(NN3[o] >= 8 ? N83[o] / 8 : 0, BITS);
        if (STAGE < 3) continue;
        epilogue(BITS, s ? N1 : N0, (uint32_t)(1024 + a + s), (plane_t*)Y + o);
    }
}

/* ================= harness ================= */
static int check(int method) {                  /* 1 = every output bit equals the reference */
    const int ow = O / 32;
    for (int p = 0; p < P; p++)
        for (int o = 0; o < O; o++) {
            int ref = (REF[p * ow + o / 32] >> (o % 32)) & 1;
            int got = method == 4 ? (int)((Y[o * 4 + p / 32] >> (p % 32)) & 1) : (int)((Y[p * ow + o / 32] >> (o % 32)) & 1);
            if (ref != got) return 0;
        }
    return 1;
}
static void run(int method) {
    if (method == 0) run_pie_swar(); else if (method == 1) run_int8(1); else if (method == 2) run_int8(0);
    else if (method == 3) run_vert_out(); else run_vert_pos();
}
static void measure_as(const char* shape, int method, const char* label) {
    memset(Y, 0, (size_t)(P > O ? P : O) * (O > 128 ? O : 128) / 8);
    run(method);
    int exact = method == 2 || (method >= 3 && STAGE < 3) ? -1 : check(method);
    int reps = 1 + 2000000 / (P * O * K / 8 + 1);
    double best = 1e30;
    for (int r = 0; r < 7; r++) {
        uint32_t t0 = (uint32_t)esp_cpu_get_cycle_count();
        for (int k = 0; k < reps; k++) run(method);
        double t = (double)(uint32_t)((uint32_t)esp_cpu_get_cycle_count() - t0) / reps;
        if (t < best) best = t;
    }
    sink = Y[0];
    printf("R:%s|%s|%.1f|%d\n", shape, label, best, exact);
    fflush(stdout);
}
static void measure(const char* shape, int method) {
    static const char* names[] = {"pie_swar", "int8", "int8_mac", "vert_out", "vert_pos"};
    measure_as(shape, method, names[method]);
}
/* every stage of a vertical method (the whole layer last, under its plain name) */
static void measure_stages(const char* shape, int method) {
    static const char* parts[2][3] = {{"vert_out_prep", "vert_out_sum", "vert_out_levels"},
                                      {"vert_pos_prep", "vert_pos_sum", "vert_pos_levels"}};
    for (STAGE = 0; STAGE < 3; STAGE++) measure_as(shape, method, parts[method - 3][STAGE]);
    STAGE = 3; measure(shape, method);
}
/* the weights into the "bench" flash partition, then mapped through the cache (read-only, like a model's const data) */
static const esp_partition_t* PART;
static esp_partition_mmap_handle_t FLH;
static const uint8_t* to_flash(const void* a, size_t na, const void* b, size_t nb) {
    size_t total = (na + 15) / 16 * 16 + nb;
    if (total > PART->size) { printf("ERROR: %u bytes do not fit the bench partition\n", (unsigned)total); return NULL; }
    const void* map;
    if (esp_partition_erase_range(PART, 0, (total + 4095) / 4096 * 4096) != ESP_OK ||
        esp_partition_write(PART, 0, a, na) != ESP_OK || (nb && esp_partition_write(PART, (na + 15) / 16 * 16, b, nb) != ESP_OK) ||
        esp_partition_mmap(PART, 0, (total + 65535) / 65536 * 65536, ESP_PARTITION_MMAP_DATA, &map, &FLH) != ESP_OK) {
        printf("ERROR: writing or mapping the bench partition failed\n"); return NULL;
    }
    return (const uint8_t*)map;
}

static const int SHAPES_C[][4] = {SHAPES_INIT};
void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(500));
    memset(&ONESQ, 0xFF, sizeof ONESQ);
    QB = A(64); BA = A(160 * 16); BB = A(160 * 16); BITS = A(NB * 16); PX = A(NB * 16); N0 = A(EB * 16); N1 = A(EB * 16);
    PART = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "bench");
    if (!PART) { printf("ERROR: no bench partition (partitions.csv)\n"); return; }
    K = 16;
    if (!calibrate_qacc()) return;
    printf("\nS3 VERTICAL BENCH ready\n");
    printf("QACC lanes:"); for (int l = 0; l < 16; l++) printf(" %d", MAP[l]); printf("\n");
    fflush(stdout);
    for (unsigned si = 0; si < sizeof SHAPES_C / sizeof SHAPES_C[0]; si++) {
        int dense = SHAPES_C[si][0];
        P = SHAPES_C[si][1]; K = SHAPES_C[si][2]; O = SHAPES_C[si][3];
        nw = (K + 31) / 32; st = (nw + 3) / 4 * 4;
        char shape[32]; snprintf(shape, sizeof shape, "%s_%dx%dx%d", dense ? "dense" : "conv", P, K, O);
        X = A((size_t)P * st * 4); W = A((size_t)O * st * 4); REF = A((size_t)P * O / 8);
        Y = A((size_t)(P > O ? P : O) * (O > 128 ? O : 128) / 8);
        T = A(O * 4); T2 = A(O * 4); PCW = A(O * 4);
        for (int p = 0; p < P; p++) for (int j = 0; j < nw; j++) X[p * st + j] = rnd() & (j == nw - 1 && K % 32 ? (1u << (K % 32)) - 1 : ~0u);
        for (int o = 0; o < O; o++) {
            for (int j = 0; j < nw; j++) { W[o * st + j] = rnd() & (j == nw - 1 && K % 32 ? (1u << (K % 32)) - 1 : ~0u); PCW[o] += __builtin_popcount(W[o * st + j]); }
            T[o] = K / 2 - 8 + (int)(rnd() % 17); T2[o] = 2 * T[o] - K;
        }
        for (int p = 0; p < P; p++)
            for (int o = 0; o < O; o++) {
                int d = 0; for (int j = 0; j < nw; j++) d += __builtin_popcount(X[p * st + j] ^ W[o * st + j]);
                if (K - d >= T[o]) REF[p * (O / 32) + o / 32] |= 1u << (o % 32);
            }
        measure(shape, 0);
        X8 = A((size_t)P * K); W8 = A((size_t)O * K + 32);
        for (int p = 0; p < P; p++) for (int k = 0; k < K; k++) X8[p * K + k] = (X[p * st + k / 32] >> (k % 32)) & 1 ? 1 : -1;
        for (int o = 0; o < O; o++) for (int k = 0; k < K; k++) W8[(size_t)(o / 16) * K * 16 + k * 16 + o % 16] = (W[o * st + k / 32] >> (k % 32)) & 1 ? 1 : -1;
        measure(shape, 1); measure(shape, 2);
        heap_caps_free(X8); heap_caps_free(W8);
        if (dense) {
            WT = A((size_t)(K + 1) * O / 8); N2 = A((size_t)(O / 128) * 2 * EB * 16); D = A((size_t)(K + 8) * 2);
            prep_vert_out(); WTu = WT; N2u = N2; measure_stages(shape, 3);
            size_t nwt = (size_t)(K + 1) * O / 8, nn2 = (size_t)(O / 128) * 2 * EB * 16;
            const uint8_t* fl = to_flash(WT, nwt, N2, nn2);
            if (!fl) return;
            WTu = (const plane_t*)fl; N2u = (const plane_t*)(fl + (nwt + 15) / 16 * 16);
            measure_as(shape, 3, "vert_out_flash");
            esp_partition_munmap(FLH);
            heap_caps_free(WT); heap_caps_free(N2); heap_caps_free(D);
        } else {
            XT = A((size_t)(K + 1) * 16); D3 = A((size_t)O * (K + 8) * 2); S3 = A(O * 4); NN3 = A(O * 4); N83 = A(O * 4);
            prep_vert_pos(); D3u = D3; measure_stages(shape, 4);
            const uint8_t* fl = to_flash(D3, (size_t)O * (K + 8) * 2, NULL, 0);
            if (!fl) return;
            D3u = (const uint16_t*)fl;
            measure_as(shape, 4, "vert_pos_flash");
            esp_partition_munmap(FLH);
            heap_caps_free(XT); heap_caps_free(D3); heap_caps_free(S3); heap_caps_free(NN3); heap_caps_free(N83);
        }
        heap_caps_free(X); heap_caps_free(W); heap_caps_free(REF); heap_caps_free(Y);
        heap_caps_free(T); heap_caps_free(T2); heap_caps_free(PCW);
    }
    printf("DONE\n");
    fflush(stdout);
    while (1) vTaskDelay(portMAX_DELAY);
}
"""


def write_project(port):
    os.makedirs(os.path.join(PROJECT, "src"), exist_ok=True)
    w = lambda name, text: open(os.path.join(PROJECT, name), "w").write(text)
    w("platformio.ini", f"[env:s3]\nplatform = espressif32\nboard = esp32-s3-devkitc-1\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\nboard_build.partitions = partitions.csv\n")
    w("partitions.csv", "# Name, Type, SubType, Offset, Size\nnvs, data, nvs, 0x9000, 0x6000\n"
                        "phy_init, data, phy, 0xf000, 0x1000\nfactory, app, factory, 0x10000, 0x100000\n"
                        "bench, data, 0x40, 0x110000, 0x80000\n")
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n")
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(s3_vertical_count)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" INCLUDE_DIRS "." REQUIRES esp_partition)\n')
    init = ", ".join("{%d, %d, %d, %d}" % (kind == "dense", p, k, o) for kind, p, k, o in SHAPES)
    w("src/main.c", FIRMWARE.replace("SHAPES_INIT", init))


def read_results(port):
    import serial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 115200, 1
    s.dtr = s.rts = False
    s.open()
    s.reset_input_buffer()
    s.rts = True
    time.sleep(0.15)
    s.rts = False
    rows, ready, t0 = [], False, time.time()
    while True:
        line = s.readline().decode(errors="replace").strip()
        if "ERROR" in line:
            sys.exit(f"[X] {line}")
        if "S3 VERTICAL BENCH ready" in line:
            ready = True
        elif ready and line.startswith("QACC lanes:"):
            print(f"      {line}")
        elif ready and line.startswith("R:"):
            shape, m, c, ex = line[2:].split("|")
            rows.append(dict(shape=shape, method=m, cycles=float(c), exact=int(ex)))
            print(f"      {shape:<22}{m:<10}{float(c):>12.0f} cycles")
        elif ready and line == "DONE":
            return rows
        if not ready and time.time() - t0 > 10:
            print("  >>> no answer to the software reset: press RST once")
            t0 = time.time() + 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    write_project(args.port)
    if not args.no_upload:
        print("[*] building and flashing the ESP32-S3 ...")
        if subprocess.run(["pio", "run", "-d", PROJECT, "-t", "upload"]).returncode != 0:
            sys.exit(f"[X] build or upload failed (pio run -d {PROJECT} -t upload)")
    rows = read_results(args.port)
    print("\n    BINARY LAYER ON THE ESP32-S3, measured  (cycles of the whole layer, best of 7; MAC = P x K x O)\n")
    print(f"    {'shape (PxKxO)':<22}{'method':<10}{'cycles':>11}{'cyc/output':>12}{'MAC/cycle':>11}"
          f"{'vs pie_swar':>13}   exact")
    for kind, p, k, o in SHAPES:
        shape = f"{kind}_{p}x{k}x{o}"
        r = {x["method"]: x for x in rows if x["shape"] == shape}
        base = r["pie_swar"]["cycles"]
        for m, x in r.items():
            if m.endswith(("_prep", "_sum", "_levels")):
                continue                        # the stages go to the breakdown table
            ex = {1: "yes", 0: "NO [X]", -1: "(not checked)"}[x["exact"]]
            print(f"    {shape:<22}{m:<10}{x['cycles']:>11.0f}{x['cycles'] / (p * o):>12.1f}"
                  f"{p * k * o / x['cycles']:>11.2f}{base / x['cycles']:>12.2f}x   {ex}")
        print()
    print("    WHERE THE TIME OF THE VERTICAL METHODS GOES  (cycles per output; each stage = difference of two runs)\n")
    print(f"    {'shape (PxKxO)':<22}{'method':<10}{'prep':>8}{'sum':>8}{'levels':>8}{'thresh':>8}{'total':>8}"
          f"   {'sum %':>6}   {'flash/RAM':>9}")
    for kind, p, k, o in SHAPES:
        shape = f"{kind}_{p}x{k}x{o}"
        r = {x["method"]: x["cycles"] for x in rows if x["shape"] == shape}
        m = "vert_out" if kind == "dense" else "vert_pos"
        if m + "_prep" not in r:
            continue
        cum = [r[m + "_prep"], r[m + "_sum"], r[m + "_levels"], r[m]]
        parts = [cum[0]] + [cum[i] - cum[i - 1] for i in (1, 2, 3)]
        n = p * o
        print(f"    {shape:<22}{m:<10}" + "".join(f"{x / n:>8.1f}" for x in parts) + f"{cum[3] / n:>8.1f}"
              f"   {100 * parts[1] / cum[3]:>5.0f}%   {r[m + '_flash'] / cum[3]:>8.2f}x")
    print()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(dict(board="esp32s3", mhz=240, shapes=SHAPES, rows=rows), f, indent=1)
    print(f"    saved {os.path.relpath(RESULTS)}")


if __name__ == "__main__":
    main()
