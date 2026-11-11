"""
s3_vertical_layers.py -- the vertical (bit-sliced) count on the real ResNet-8 layers, on the ESP32-S3: every conv of
the trained network (runs/w1_s42), with its real shape, borders and stride, against lib_symbex's own
symbex_bool_conv on the same chip and the same inputs, output bit by output bit.

    python s3_vertical_layers.py --port /dev/ttyACM0
    python s3_vertical_layers.py --port /dev/ttyACM0 --no-upload      # the firmware is already on the board
    python s3_vertical_layers.py --run runs/w1_s42 --images 8

How a layer runs vertically (lanes = 128 output pixels of one channel, row-major):
    planes     the input is kept as channel planes (bit p of plane c = pixel p, channel c: what the previous vertical
               layer writes). For every kernel pixel (kh, kw) and channel c, the plane of the input shifted by
               (kh - pad, kw - pad), zero outside the image (a word shift and a column mask). Stride 2: the channel
               planes are split first into their 4 phases (even / odd rows x columns), then shifted by 0 or -1.
    pcx        the popcount of every window, counted vertically over the K = k*k*cin planes, once per tile; kept as
               the planes of 2048 - pcx
    sum        per output channel, the planes of the weights that are 1 summed with Harley-Seal carry-save adders
               (a fixed list of plane offsets per output, uint16), then counted to binary (cnt)
    threshold  out = 2 cnt - pcx >= E, E = ceil((thr + 2 pcw_valid - nbits) / 2) per output and pixel: borders
               change pcw_valid and nbits, so E is a plane set per output and tile kind (precomputed from the
               weights, deduplicated). Bit 12 of 2 cnt + (2048 - pcx) + (2048 - E): a 3-operand bit-sliced add.
The output is the channel planes of the next layer. Not timed here: the HWC <-> planes conversion of the input
and output (a chain of vertical layers does not need it; the thermometer can write planes directly), the OR pool
and OR of the shortcuts on planes, the classifier.

The inputs are random RGB images run through the library's own network up to each layer (the timing does not depend
on the image: the lists come from the weights); every image is checked bit by bit. Cycles: best of 5 runs.
_flash: the lists and the threshold planes in a flash partition read through the cache, as deployed; the library's
weights are in flash (const) in every row.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "experiments", "chip_kernels"))
from compiler_core.chip.core import lib_sources  # noqa: E402
import s3_vertical_count as svc  # noqa: E402  (the Harley-Seal primitives, measured exact there)

PROJECT = os.path.join(HERE, "runs", "s3_vertical_layers")
LIB = os.path.join(ROOT, "Symbex1_PIO", "lib", "lib_symbex")
NAMES = ["stem 24->16 32x32", "b0 16->16 32x32", "b1 16->16 32x32", "b2 16->32 s2 32->16", "b2 shortcut 1x1 16->32",
         "b3 32->32 16x16", "b4 32->64 s2 16->8", "b4 shortcut 1x1 32->64", "b5 64->64 8x8"]


def primitives():
    """the vertical-count C primitives of s3_vertical_count.py (CSA, HS8, hs_gather, hs_contig, count_finish):
    the same code that was measured exact there, not a copy"""
    fw = svc.FIRMWARE
    a = fw.find("/* ================= vertical counting")
    b = fw.find("/* out = (count + N >= M)")
    if a < 0 or b < a:
        sys.exit("[X] s3_vertical_count.py changed: its vertical-count section was not found")
    return fw[a:b]


FIRMWARE_HEAD = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "model_symbex.h"

typedef struct { uint32_t w[4]; } __attribute__((aligned(16))) plane_t;
#define NB 15                       /* counter planes kept by count_finish */
#define EP 12                       /* planes of the threshold operands (every value < 4096) */
static plane_t *BA, *BB;            /* eights planes of count_finish */
static void* A(size_t n) {
    void* p = heap_caps_aligned_alloc(16, n ? n : 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) { printf("ERROR: no RAM for %u bytes\n", (unsigned)n); fflush(stdout); while (1) vTaskDelay(portMAX_DELAY); }
    memset(p, 0, n ? n : 16); return p;
}
"""

FIRMWARE_BODY = r"""
/* ================= one conv layer, vertical ================= */
static const struct SymbexBoolConv* LC;
static int cin, cout, kq, pad, sd, H, W, OH, OW, nout, nwo, nwi, K, K8, TL, NBG;
static uint32_t *CPI, *PH;                  /* input channel planes / their 4 phases (stride 2), 2 guard words each side */
static plane_t *XQ, *OUTP, *AP, *BG, *CB, *PXB;
static uint16_t *LST, *BIDX;
static uint32_t *LSTART, *LOFF;
static int *LN, *LN8;
static const uint16_t* LSTu;                /* the lists and threshold planes in use: RAM or the flash partition */
static const plane_t* BGu;
static size_t LST_BYTES, BG_BYTES;
static int VSTAGE = 2;                      /* 0 planes, 1 + pcx, 2 the whole layer */
static int LIST_RT = 0;                     /* 1: the lists built at run time from the weights (into LRB), bit by bit;
                                               2: the same lists, one weight byte at a time from a table (make_list_fast) */
static int THR_RT = 0;                      /* 1: the threshold planes built at run time from one value per border kind */
/* border kinds: the lanes whose window has the same in-image kernel pixels. KM[t * NK + k] = lanes of kind k in tile
   t; tiles with the same masks share a type (TT[t]; REP[type] = its first tile; TK / TKN = its non-base kinds).
   DKA[o * (NK + 1)]: B = 2048 - E of the base kind (the commonest, the interior), then per kind B_k XOR B_base. */
#define MAX_KINDS 16
static int NK, NTT, KBASE;
static plane_t *KM, *BT;
static uint8_t *TT, *REP, *TK, *TKN;
static uint16_t* DKA;
static const uint16_t* DKu;
static size_t DK_BYTES;
static uint16_t *LRB, *LRBA;                /* LRB = LRBA + 2: make_list_fast writes the start into LRB[-1] */
static int make_list(const uint32_t* row, uint16_t* d, uint32_t* start);
static int make_list_fast(const uint32_t* row, uint16_t* d, uint32_t* start);
#define PW(c) (CPI + (size_t)(c) * (nwi + 4) + 2)
#define PHW(r, c) (PH + ((size_t)(r) * cin + (c)) * (nwo + 4) + 2)

/* out bit p = src bit p + off (0 outside the plane: guard words), then the column mask; written tile by tile */
static void shift_plane(const uint32_t* s, int off, uint32_t mask, uint32_t* d) {
    const int a = off >> 5, b = off & 31, step = K8 * 4;
    s += a;
    if (b == 0) for (int i = 0; i < nwo; i++) d[(i >> 2) * step + (i & 3)] = s[i] & mask;
    else        for (int i = 0; i < nwo; i++) d[(i >> 2) * step + (i & 3)] = ((s[i] >> b) | (s[i + 1] << (32 - b))) & mask;
}
/* lanes of a word whose column is 0 (sx = -1) or OW - 1 (sx = +1) cleared: their source is outside the image */
static uint32_t colmask(int sx) {
    if (sx == 0) return ~0u;
    uint32_t p = 0;
    for (int j = 0; j < 32; j += OW) p |= 1u << j;
    return sx < 0 ? ~p : ~(p << (OW - 1));
}
/* even bits to the low half, odd bits to the high half (Hacker's Delight unshuffle) */
static inline uint32_t unshuffle(uint32_t x) {
    uint32_t t;
    t = (x ^ (x >> 1)) & 0x22222222u; x ^= t ^ (t << 1);
    t = (x ^ (x >> 2)) & 0x0C0C0C0Cu; x ^= t ^ (t << 2);
    t = (x ^ (x >> 4)) & 0x00F000F0u; x ^= t ^ (t << 4);
    t = (x ^ (x >> 8)) & 0x0000FF00u; x ^= t ^ (t << 8);
    return x;
}

/* V = 2 cnt + (2048 - pcx) + (2048 - E): CSA of the three operands bit by bit (x = cnt planes shifted by one),
   then the carry chain of sum + carries; out = bit 12 of V. q0 = chain carry, q1 / q2 = CSA carry (alternating). */
#define E3(KIN, KOUT) "ee.vld.128.ip q3, %[a], 16 \n\t ee.vld.128.ip q4, %[b], 16 \n\t ee.vld.128.ip q5, %[x], 16 \n\t" \
    "ee.xorq q6, q3, q4 \n\t ee.andq " KOUT ", q3, q4 \n\t ee.andq q3, q6, q5 \n\t ee.xorq q6, q6, q5 \n\t" \
    "ee.orq " KOUT ", " KOUT ", q3 \n\t ee.xorq q7, q6, " KIN " \n\t ee.andq q6, q6, " KIN " \n\t" \
    "ee.andq q7, q7, q0 \n\t ee.orq q0, q6, q7 \n\t"
static void epilogue3(const plane_t* a, const plane_t* b, const plane_t* x, plane_t* out) {
    __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t"
                         E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1")
                         E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1")
                         "ee.xorq q2, q1, q0 \n\t ee.vst.128.ip q2, %[o], 0 \n\t"
                         : [a] "+r"(a), [b] "+r"(b), [x] "+r"(x), [o] "+r"(out) : : "memory");
}

static void run_vertical(void) {
    if (sd == 2) {                              /* the 4 phases of every channel plane */
        const uint32_t om = OW == 32 ? ~0u : (1u << OW) - 1, rm = W == 32 ? ~0u : (1u << W) - 1;
        for (int c = 0; c < cin; c++) {
            for (int r = 0; r < 4; r++) memset(PHW(r, c), 0, (size_t)nwo * 4);
            const uint32_t* src = PW(c);
            uint32_t *e0 = PHW(0, c), *o0 = PHW(1, c), *e1 = PHW(2, c), *o1 = PHW(3, c);
            for (int y = 0; y < H; y++) {
                const int bit = y * W, ob = (y >> 1) * OW;
                const uint32_t u = unshuffle((src[bit >> 5] >> (bit & 31)) & rm);
                uint32_t *ev = (y & 1) ? e1 : e0, *od = (y & 1) ? o1 : o0;
                ev[ob >> 5] |= (u & om) << (ob & 31);
                od[ob >> 5] |= ((u >> 16) & om) << (ob & 31);
            }
        }
    }
    for (int kh = 0; kh < kq; kh++)             /* the shifted plane of every (kernel pixel, channel) */
        for (int kw = 0; kw < kq; kw++) {
            const int q = kh * kq + kw;
            int sy, sx, ph = 0;
            if (sd == 1) { sy = kh - pad; sx = kw - pad; }
            else { sy = (kh - pad) >> 1; sx = (kw - pad) >> 1; ph = ((kh - pad) & 1) * 2 + ((kw - pad) & 1); }
            const int off = sy * OW + sx;
            const uint32_t m = colmask(sx);
            for (int c = 0; c < cin; c++)
                shift_plane(sd == 1 ? PW(c) : PHW(ph, c), off, m, (uint32_t*)(XQ + q * cin + c));
        }
    if (VSTAGE < 1) return;
    for (int t = 0; t < TL; t++) {              /* pcx of every window, then the planes of 2048 - pcx */
        zero3(); hs_contig(XQ + (size_t)t * K8, K8 / 8, BA); count_finish(K8 / 8, PXB);
        for (int w = 0; w < 4; w++) {
            uint32_t c = ~0u;
            for (int i = 0; i < EP; i++) {
                const uint32_t a = ~PXB[i].w[w], s = a ^ c;
                c &= a;
                AP[t * EP + i].w[w] = i == EP - 1 ? ~s : s;
            }
        }
    }
    if (VSTAGE < 2) return;
    for (int o = 0; o < cout; o++) {
        const uint16_t* lst = LSTu + LOFF[o];
        uint32_t start = LSTART[o];
        int n = LN[o];
        if (LIST_RT == 2) { n = make_list_fast(LC->weights + (size_t)o * LC->row_words, LRB, &start); lst = LRB; }
        else if (LIST_RT) { n = make_list(LC->weights + (size_t)o * LC->row_words, LRB, &start); lst = LRB; }
        const int n8 = (n + 7) & ~7;
        if (THR_RT) {                           /* the threshold planes of every tile type of output o */
            const uint16_t* dk = DKu + (size_t)o * (NK + 1);
            const uint32_t bb = dk[0];
            for (int ty = 0; ty < NTT; ty++) {
                plane_t* P = BT + ty * EP;
                for (int i = 0; i < EP; i++) {
                    const uint32_t v = 0u - ((bb >> i) & 1);
                    P[i].w[0] = v; P[i].w[1] = v; P[i].w[2] = v; P[i].w[3] = v;
                }
                for (int j = 0; j < TKN[ty]; j++) {
                    const int k = TK[ty * MAX_KINDS + j];
                    const plane_t* m = &KM[REP[ty] * NK + k];
                    for (uint32_t d = dk[1 + k]; d; d &= d - 1) {
                        plane_t* q = &P[__builtin_ctz(d)];
                        q->w[0] ^= m->w[0]; q->w[1] ^= m->w[1]; q->w[2] ^= m->w[2]; q->w[3] ^= m->w[3];
                    }
                }
            }
        }
        for (int t = 0; t < TL; t++) {
            zero3();
            if (n8) hs_gather((const uint8_t*)(XQ + (size_t)t * K8) + start, lst, n8 / 8, BA);
            count_finish(n >= 8 ? n8 / 8 : 0, CB + 1);
            epilogue3(AP + t * EP, THR_RT ? BT + TT[t] * EP : BGu + (size_t)BIDX[o * TL + t] * EP, CB, OUTP + o * TL + t);
        }
    }
}

/* the list of output o from its weight row (1 bit per weight): start offset + deltas (bytes) of the planes of the
   weights that are 1, padded to a multiple of 8 with the zero plane K; returns how many are 1 */
static int make_list(const uint32_t* row, uint16_t* d, uint32_t* start) {
    int n = 0; uint32_t prev = 0;
    *start = 0;
    const int words = (K + 31) / 32;
    for (int j = 0; j < words; j++) {
        uint32_t v = row[j];
        if (j == words - 1 && (K & 31)) v &= (1u << (K & 31)) - 1;
        while (v) {
            const uint32_t off = (uint32_t)(j * 32 + __builtin_ctz(v)) * 16;
            v &= v - 1;
            if (n == 0) *start = off; else d[n - 1] = (uint16_t)(off - prev);
            prev = off; n++;
        }
    }
    const int n8 = (n + 7) & ~7;
    if (n) { d[n - 1] = (uint16_t)(n8 > n ? (uint32_t)K * 16 - prev : 0); for (int i = n; i < n8; i++) d[i] = 0; }
    return n;
}

/* the byte table of make_list_fast: per byte value the byte deltas between its consecutive set bits (16 bytes per
   plane), its first and last set bit times 16 and its count of ones; built once at start-up (5 KB) */
typedef struct { uint16_t d[7]; uint16_t f; uint16_t l; uint16_t n; } ListByte;
static ListByte LT[256];
static void make_list_table(void) {
    for (int b = 0; b < 256; b++) {
        int c = 0, prev = 0;
        memset(&LT[b], 0, sizeof LT[b]);
        for (int i = 0; i < 8; i++)
            if ((b >> i) & 1) {
                if (c == 0) LT[b].f = (uint16_t)(i * 16); else LT[b].d[c - 1] = (uint16_t)((i - prev) * 16);
                prev = i; c++;
            }
        LT[b].n = (uint16_t)c; LT[b].l = (uint16_t)(prev * 16);
    }
}
/* the same list as make_list, one weight byte per step and no branch: q is the slot of the delta into the next set
   bit (d[-1] first, which receives the start offset); the in-byte deltas are copied from the table (7 always, the
   buffer has room); a zero byte writes slots that are overwritten later and keeps prev. K is a multiple of 8. */
static int make_list_fast(const uint32_t* row, uint16_t* d, uint32_t* start) {
    const uint8_t* rb = (const uint8_t*)row;
    const int bytes = K / 8;
    uint16_t* q = d - 1;
    uint32_t prev = 0;
    for (int j = 0; j < bytes; j++) {
        const ListByte* e = &LT[rb[j]];
        const uint32_t base = (uint32_t)j * 128, c = e->n;
        q[0] = (uint16_t)(base + e->f - prev);
        q[1] = e->d[0]; q[2] = e->d[1]; q[3] = e->d[2]; q[4] = e->d[3]; q[5] = e->d[4]; q[6] = e->d[5]; q[7] = e->d[6];
        q += c;
        const uint32_t last = base + e->l;
        prev = c ? last : prev;
    }
    const int n = (int)(q - (d - 1));
    *start = n ? d[-1] : 0;
    const int n8 = (n + 7) & ~7;
    if (n) { d[n - 1] = (uint16_t)(n8 > n ? (uint32_t)K * 16 - prev : 0); for (int i = n; i < n8; i++) d[i] = 0; }
    return n;
}

/* what the exporter would precompute from the weights: lists and threshold planes */
static int prep_layer(const struct SymbexBoolConv* L) {
    LC = L; cin = L->in_ch; cout = L->out_ch; kq = L->kernel; pad = kq / 2; sd = L->stride;
    H = L->in_h; W = L->in_w; OH = L->out_h; OW = L->out_w; nout = OH * OW; nwo = nout / 32; nwi = H * W / 32;
    K = kq * kq * cin; K8 = (K + 1 + 7) / 8 * 8; TL = (nout + 127) / 128;
    if (32 % OW || 32 % W || nout % 32 || (sd != 1 && sd != 2) || (sd == 2 && (H != 2 * OH || W != 2 * OW || pad != 1)) ||
        (sd == 1 && (H != OH || W != OW)) || K * 16 > 65535) {
        printf("ERROR: layer shape not supported by this bench\n"); return 0;
    }
    CPI = A((size_t)cin * (nwi + 4) * 4);
    PH = sd == 2 ? A((size_t)4 * cin * (nwo + 4) * 4) : NULL;
    XQ = A((size_t)TL * K8 * 16); OUTP = A((size_t)cout * TL * 16); AP = A((size_t)TL * EP * 16);
    LSTART = A(cout * 4); LOFF = A(cout * 4); LN = A(cout * 4); LN8 = A(cout * 4);
    size_t tot = 0;
    for (int o = 0; o < cout; o++) {
        const uint32_t* row = L->weights + (size_t)o * L->row_words;
        int n = 0;
        for (int k = 0; k < K; k++) n += (row[k >> 5] >> (k & 31)) & 1;
        LN[o] = n; LN8[o] = (n + 7) & ~7; LOFF[o] = tot; tot += LN8[o];
    }
    LST = A(tot * 2 + 16); LST_BYTES = tot * 2;
    for (int o = 0; o < cout; o++) make_list(L->weights + (size_t)o * L->row_words, LST + LOFF[o], &LSTART[o]);
    LRBA = A((size_t)(K + 18) * 2); LRB = LRBA + 2;
    if (K & 7) { printf("ERROR: K not a multiple of 8 (make_list_fast)\n"); return 0; }
    BG = A((size_t)cout * TL * EP * 16); BIDX = A((size_t)cout * TL * 2); NBG = 0;
    plane_t tmp[EP];
    for (int o = 0; o < cout; o++) {
        for (int t = 0; t < TL; t++) {
            memset(tmp, 0, sizeof tmp);
            for (int j = 0; j < 128; j++) {
                const int p = t * 128 + j;
                int32_t E = 0;
                if (p < nout) {
                    const int oy = p / OW, ox = p % OW;
                    int nv = 0, pw = 0;
                    for (int kh = 0; kh < kq; kh++)
                        for (int kw = 0; kw < kq; kw++) {
                            const int y = oy * sd - pad + kh, x = ox * sd - pad + kw;
                            if (y >= 0 && y < H && x >= 0 && x < W) { nv++; pw += L->weight_counts[o * kq * kq + kh * kq + kw]; }
                        }
                    const int64_t R = (int64_t)L->thresholds[o] + 2 * pw - nv * cin;
                    int64_t e = (R + 1) >> 1;                       /* ceil(R / 2) */
                    if (e < -K) e = -K;
                    if (e > K + 1) e = K + 1;
                    E = (int32_t)e;
                }
                const uint32_t b = (uint32_t)(2048 - E);
                for (int i = 0; i < EP; i++) if ((b >> i) & 1) tmp[i].w[j >> 5] |= 1u << (j & 31);
            }
            int g = 0;
            while (g < NBG && memcmp(&BG[(size_t)g * EP], tmp, sizeof tmp)) g++;
            if (g == NBG) { memcpy(&BG[(size_t)g * EP], tmp, sizeof tmp); NBG++; }
            BIDX[o * TL + t] = (uint16_t)g;
        }
    }
    BG_BYTES = (size_t)NBG * EP * 16;
    /* border kinds and tile types (geometry only), then one value per output and kind */
    uint16_t kv[MAX_KINDS];
    int kc[MAX_KINDS];
    uint8_t* lk = A(TL * 128);
    NK = 0;
    for (int p = 0; p < nout; p++) {
        const int oy = p / OW, ox = p % OW;
        uint16_t vm = 0;
        for (int kh = 0; kh < kq; kh++)
            for (int kw = 0; kw < kq; kw++) {
                const int y = oy * sd - pad + kh, x = ox * sd - pad + kw;
                if (y >= 0 && y < H && x >= 0 && x < W) vm |= 1u << (kh * kq + kw);
            }
        int k = 0;
        while (k < NK && kv[k] != vm) k++;
        if (k == NK) { if (NK == MAX_KINDS) { printf("ERROR: too many border kinds\n"); return 0; } kv[NK] = vm; kc[NK++] = 0; }
        kc[k]++; lk[p] = (uint8_t)k;
    }
    KBASE = 0;
    for (int k = 1; k < NK; k++) if (kc[k] > kc[KBASE]) KBASE = k;
    for (int p = nout; p < TL * 128; p++) lk[p] = (uint8_t)KBASE;
    KM = A((size_t)TL * NK * 16);
    for (int p = 0; p < TL * 128; p++) KM[(p / 128) * NK + lk[p]].w[(p % 128) / 32] |= 1u << (p % 32);
    heap_caps_free(lk);
    TT = A(TL); REP = A(TL); TK = A(TL * MAX_KINDS); TKN = A(TL); NTT = 0;
    for (int t = 0; t < TL; t++) {
        int ty = 0;
        while (ty < NTT && memcmp(&KM[REP[ty] * NK], &KM[t * NK], (size_t)NK * 16)) ty++;
        if (ty == NTT) {
            REP[NTT] = (uint8_t)t; TKN[NTT] = 0;
            for (int k = 0; k < NK; k++) {
                const plane_t* m = &KM[t * NK + k];
                if (k != KBASE && (m->w[0] | m->w[1] | m->w[2] | m->w[3])) TK[NTT * MAX_KINDS + TKN[NTT]++] = (uint8_t)k;
            }
            NTT++;
        }
        TT[t] = (uint8_t)ty;
    }
    BT = A((size_t)NTT * EP * 16);
    DKA = A((size_t)cout * (NK + 1) * 2); DK_BYTES = (size_t)cout * (NK + 1) * 2;
    for (int o = 0; o < cout; o++) {
        uint32_t bk[MAX_KINDS];
        for (int k = 0; k < NK; k++) {
            int nv = 0, pw = 0;
            for (int q = 0; q < kq * kq; q++)
                if ((kv[k] >> q) & 1) { nv++; pw += L->weight_counts[o * kq * kq + q]; }
            const int64_t R = (int64_t)L->thresholds[o] + 2 * pw - nv * cin;
            int64_t e = (R + 1) >> 1;
            if (e < -K) e = -K;
            if (e > K + 1) e = K + 1;
            bk[k] = (uint32_t)(2048 - e);
        }
        DKA[o * (NK + 1)] = (uint16_t)bk[KBASE];
        for (int k = 0; k < NK; k++) DKA[o * (NK + 1) + 1 + k] = (uint16_t)(bk[k] ^ bk[KBASE]);
    }
    LSTu = LST; BGu = BG; DKu = DKA;
    return 1;
}
static void free_layer(void) {
    heap_caps_free(CPI); if (PH) heap_caps_free(PH);
    heap_caps_free(XQ); heap_caps_free(OUTP); heap_caps_free(AP);
    heap_caps_free(LSTART); heap_caps_free(LOFF); heap_caps_free(LN); heap_caps_free(LN8);
    heap_caps_free(LST); heap_caps_free(LRBA); heap_caps_free(BG); heap_caps_free(BIDX);
    heap_caps_free(KM); heap_caps_free(BT); heap_caps_free(TT); heap_caps_free(REP); heap_caps_free(TK); heap_caps_free(TKN);
    heap_caps_free(DKA);
}

/* ================= the library side: inputs and reference ================= */
static const struct SymbexBoolConv* LAYERS[9];
static uint8_t IMG[32 * 32 * 3];
static uint32_t rs;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static uint32_t map_words(int h, int w, int ch) { return (uint32_t)h * w * ((ch + 31) / 32); }
/* a random image through the library's network up to layer `layer`; its input map copied to dst */
static void capture(uint32_t seed, int layer, uint32_t* dst) {
    rs = seed * 2654435761u + 12345u;
    for (int i = 0; i < 32 * 32 * 3; i++) IMG[i] = (uint8_t)(rnd() >> 24);
    const struct SymbexBoolNet* net = &symbex_net;
    uint32_t* const* m = net->maps;
    symbex_thermometer(IMG, net->in_h, net->in_w, net->levels, m[0]);
    if (layer == 0) { memcpy(dst, m[0], map_words(32, 32, 24) * 4); return; }
    symbex_bool_conv(&net->stem, m[0], m[1], net->window, net->scratch);
    int x = 1, y = 0, p = 2, s = 3, idx = 1;
    for (uint16_t i = 0; i < net->n_blocks; i++) {
        const struct SymbexBoolBlock* b = &net->blocks[i];
        const struct SymbexBoolConv* c = &b->conv;
        if (layer == idx) { memcpy(dst, m[x], map_words(c->in_h, c->in_w, c->in_ch) * 4); return; }
        idx++;
        symbex_bool_conv(c, m[x], m[y], net->window, net->scratch);
        const uint32_t n = map_words(c->out_h, c->out_w, c->out_ch);
        if (b->shortcut) {
            const uint32_t* src = m[x];
            if (b->shortcut_pool) { symbex_bool_or_pool(m[x], m[p], c->in_ch, c->in_h, c->in_w); src = m[p]; }
            const struct SymbexBoolConv* sc = b->shortcut;
            if (layer == idx) { memcpy(dst, src, map_words(sc->in_h, sc->in_w, sc->in_ch) * 4); return; }
            idx++;
            symbex_bool_conv(sc, src, m[s], net->window, net->scratch);
            symbex_bool_or(m[y], m[s], n);
        } else {
            symbex_bool_or(m[y], m[x], n);
        }
        const int t = x; x = y; y = t;
    }
}
/* HWC map -> channel planes (CPI); the test harness, not timed */
static void to_planes(const uint32_t* in) {
    const int pin = (cin + 31) / 32;
    for (int c = 0; c < cin; c++) memset(PW(c), 0, (size_t)nwi * 4);
    for (int p = 0; p < H * W; p++)
        for (int c = 0; c < cin; c++)
            if ((in[p * pin + c / 32] >> (c % 32)) & 1) PW(c)[p >> 5] |= 1u << (p & 31);
}
static int same_as(const uint32_t* ref) {      /* every output bit of the vertical layer equals the library's */
    const int pout = (cout + 31) / 32;
    for (int o = 0; o < cout; o++)
        for (int p = 0; p < nout; p++) {
            const int r = (ref[p * pout + o / 32] >> (o % 32)) & 1;
            const int v = (OUTP[o * TL + p / 128].w[(p % 128) / 32] >> (p % 32)) & 1;
            if (r != v) return 0;
        }
    return 1;
}
static uint32_t cyc(void) { return (uint32_t)esp_cpu_get_cycle_count(); }
/* cold cache: 128 KB of other flash read before a run, so the layer's flash data (weights, lists, thresholds) is
   not cached -- as in a real inference, where every layer reads its data once */
static const volatile uint32_t* EVICT;
static esp_partition_mmap_handle_t EVH;
static int COLD = 0;
static volatile uint32_t sink;
static void evict(void) { uint32_t a = 0; for (int i = 0; i < 128 * 1024 / 4; i += 8) a += EVICT[i]; sink = a; }
static uint32_t best_lib(const uint32_t* in, uint32_t* out) {
    uint32_t best = ~0u;
    for (int r = 0; r < 5; r++) {
        if (COLD) evict();
        const uint32_t t0 = cyc();
        symbex_bool_conv(LC, in, out, symbex_net.window, symbex_net.scratch);
        const uint32_t t = cyc() - t0;
        if (t < best) best = t;
    }
    return best;
}
/* the list builders alone: every output's list of the layer, best of 5 (warm) */
static uint32_t best_lists(int fast) {
    uint32_t best = ~0u, st;
    for (int r = 0; r < 5; r++) {
        const uint32_t t0 = cyc();
        for (int o = 0; o < cout; o++)
            (fast ? make_list_fast : make_list)(LC->weights + (size_t)o * LC->row_words, LRB, &st);
        const uint32_t t = cyc() - t0;
        if (t < best) best = t;
    }
    return best;
}
static uint32_t best_vert(void) {
    uint32_t best = ~0u;
    for (int r = 0; r < 5; r++) {
        if (COLD) evict();
        const uint32_t t0 = cyc();
        run_vertical();
        const uint32_t t = cyc() - t0;
        if (t < best) best = t;
    }
    return best;
}

static const esp_partition_t* PART;
static esp_partition_mmap_handle_t FLH;
static const uint8_t* to_flash(const void* a, size_t na, const void* b, size_t nb, const void* c, size_t nc) {
    const size_t ob = (na + 15) / 16 * 16, oc = ob + (nb + 15) / 16 * 16, total = oc + nc;
    if (total > PART->size) { printf("ERROR: %u bytes do not fit the bench partition\n", (unsigned)total); return NULL; }
    const void* map;
    if (esp_partition_erase_range(PART, 0, (total + 4095) / 4096 * 4096) != ESP_OK ||
        esp_partition_write(PART, 0, a, na) != ESP_OK || (nb && esp_partition_write(PART, ob, b, nb) != ESP_OK) ||
        (nc && esp_partition_write(PART, oc, c, nc) != ESP_OK) ||
        esp_partition_mmap(PART, 0, (total + 65535) / 65536 * 65536, ESP_PARTITION_MMAP_DATA, &map, &FLH) != ESP_OK) {
        printf("ERROR: writing or mapping the bench partition failed\n"); return NULL;
    }
    return (const uint8_t*)map;
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(500));
    BA = A(160 * 16); BB = A(160 * 16); CB = A((NB + 1) * 16); PXB = A(NB * 16);
    make_list_table();
    PART = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "bench");
    if (!PART) { printf("ERROR: no bench partition (partitions.csv)\n"); return; }
    {
        const void* ev;
        if (esp_partition_mmap(PART, 0x40000, 0x20000, ESP_PARTITION_MMAP_DATA, &ev, &EVH) != ESP_OK) {
            printf("ERROR: mapping the eviction area failed\n"); return;
        }
        EVICT = (const volatile uint32_t*)ev;
    }
    const struct SymbexBoolNet* net = &symbex_net;
    LAYERS[0] = &net->stem;
    for (int i = 0, j = 1; i < net->n_blocks; i++) {
        LAYERS[j++] = &net->blocks[i].conv;
        if (net->blocks[i].shortcut) LAYERS[j++] = net->blocks[i].shortcut;
    }
    printf("\nS3 VERTICAL LAYERS ready\n"); fflush(stdout);
    uint32_t* inp = A(map_words(32, 32, 32) * 4);
    uint32_t* ref = A(map_words(32, 32, 32) * 4);
    for (int li = 0; li < 9; li++) {
        if (!prep_layer(LAYERS[li])) return;
        int exact = 0;
        uint32_t lib = 0, vs[3] = {0, 0, 0}, vfl = 0;
        for (int im = 0; im < N_IMAGES; im++) {
            capture((uint32_t)im + 1, li, inp);
            symbex_bool_conv(LC, inp, ref, net->window, net->scratch);
            to_planes(inp);
            memset(OUTP, 0, (size_t)cout * TL * 16);
            run_vertical();
            exact += same_as(ref);
            if (im == 0) {
                lib = best_lib(inp, ref);
                for (VSTAGE = 0; VSTAGE < 3; VSTAGE++) vs[VSTAGE] = best_vert();
                VSTAGE = 2;
            }
        }
        const uint8_t* fl = to_flash(LST, LST_BYTES, BG, BG_BYTES, DKA, DK_BYTES);
        if (!fl) return;
        LSTu = (const uint16_t*)fl; BGu = (const plane_t*)(fl + (LST_BYTES + 15) / 16 * 16);
        DKu = (const uint16_t*)((const uint8_t*)BGu + (BG_BYTES + 15) / 16 * 16);
        capture(1, li, inp);
        symbex_bool_conv(LC, inp, ref, net->window, net->scratch);
        to_planes(inp);
        memset(OUTP, 0, (size_t)cout * TL * 16);
        run_vertical();
        const int exact_fl = same_as(ref);
        vfl = best_vert();
        COLD = 1;
        const uint32_t vcold = best_vert(), libcold = best_lib(inp, ref);
        COLD = 0;
        LIST_RT = 1;
        memset(OUTP, 0, (size_t)cout * TL * 16);
        run_vertical();
        const int exact_rt = same_as(ref);
        const uint32_t vrt = best_vert();
        LIST_RT = 0;
        THR_RT = 1;                             /* thresholds from one value per border kind (lists stored) */
        memset(OUTP, 0, (size_t)cout * TL * 16);
        run_vertical();
        const int exact_thr = same_as(ref);
        const uint32_t vthr = best_vert();
        COLD = 1;
        const uint32_t vthr_cold = best_vert();
        COLD = 0;
        LIST_RT = 2; THR_RT = 0;                /* table-built lists, stored threshold planes */
        memset(OUTP, 0, (size_t)cout * TL * 16);
        run_vertical();
        const int exact_fast = same_as(ref);
        const uint32_t vfast = best_vert();
        THR_RT = 1;                             /* only the weights and one value per border kind stored */
        memset(OUTP, 0, (size_t)cout * TL * 16);
        run_vertical();
        const int exact_both = same_as(ref);
        const uint32_t vboth = best_vert();
        COLD = 1;
        const uint32_t vboth_cold = best_vert();
        COLD = 0;
        LIST_RT = 0; THR_RT = 0;
        esp_partition_munmap(FLH);
        const size_t wbytes = (size_t)cout * LC->row_words * 4 + cout * 4 + (size_t)cout * kq * kq * 2;
        printf("L:%d|%u|%u|%u|%u|%u|%d|%d|%d|%u|%u|%u|%u|%u|%d|%u|%u|%u|%u|%d|%u|%u|%d|%u|%u|%d|%u|%u\n", li, (unsigned)lib, (unsigned)vs[0], (unsigned)vs[1],
               (unsigned)vs[2], (unsigned)vfl, exact, N_IMAGES, exact_fl, (unsigned)LST_BYTES, (unsigned)BG_BYTES,
               (unsigned)wbytes, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)vrt, exact_rt, (unsigned)libcold, (unsigned)vcold,
               (unsigned)vthr, (unsigned)vthr_cold, exact_thr, (unsigned)(DK_BYTES + (size_t)TL * NK * 16),
               (unsigned)vfast, exact_fast, (unsigned)vboth, (unsigned)vboth_cold, exact_both,
               (unsigned)best_lists(0), (unsigned)best_lists(1));
        fflush(stdout);
        free_layer();
    }
    printf("DONE\n");
    fflush(stdout);
    while (1) vTaskDelay(portMAX_DELAY);
}
"""


def write_project(port, run_dir, images, dcache64=False):
    src = os.path.join(PROJECT, "src")
    os.makedirs(src, exist_ok=True)
    w = lambda name, text: open(os.path.join(PROJECT, name), "w").write(text)
    w("platformio.ini", f"[env:s3]\nplatform = espressif32\nboard = esp32-s3-devkitc-1\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\nboard_build.partitions = partitions.csv\n")
    w("partitions.csv", "# Name, Type, SubType, Offset, Size\nnvs, data, nvs, 0x9000, 0x6000\n"
                        "phy_init, data, phy, 0xf000, 0x1000\nfactory, app, factory, 0x10000, 0x100000\n"
                        "bench, data, 0x40, 0x110000, 0x80000\n")
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n"
      + ("CONFIG_ESP32S3_DATA_CACHE_64KB=y\nCONFIG_ESP32S3_DATA_CACHE_LINE_64B=y\n" if dcache64 else ""))
    for f in os.listdir(PROJECT):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(PROJECT, f))
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(s3_vertical_layers)\n")
    inc = os.path.join(src, "include")
    if os.path.isdir(inc):
        subprocess.run(["rm", "-rf", inc], check=True)
    subprocess.run(["cp", "-r", os.path.join(LIB, "include"), inc], check=True)
    sources = lib_sources(LIB)
    for name, path in sources.items():
        subprocess.run(["cp", path, os.path.join(src, name)], check=True)
    subprocess.run(["cp", os.path.join(run_dir, "model_symbex.h"), os.path.join(src, "model_symbex.h")], check=True)
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in sources)
      + ' INCLUDE_DIRS "." "include" REQUIRES esp_partition)\n')
    w("src/main.c", f"#define N_IMAGES {images}\n" + FIRMWARE_HEAD + primitives() + FIRMWARE_BODY)


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
    keys = ["layer", "lib", "planes", "pcx", "vert", "vert_flash", "exact", "images", "exact_flash", "list_bytes",
            "thr_bytes", "weight_bytes", "free_heap", "vert_rt", "exact_rt", "lib_cold", "vert_cold", "vert_thr", "vert_thr_cold", "exact_thr", "thr_rt_bytes", "vert_fast", "exact_fast",
            "vert_both", "vert_both_cold", "exact_both", "lists_bit", "lists_byte"]
    while True:
        line = s.readline().decode(errors="replace").strip()
        if "ERROR" in line:
            sys.exit(f"[X] {line}")
        if "S3 VERTICAL LAYERS ready" in line:
            ready = True
        elif ready and line.startswith("L:"):
            r = dict(zip(keys, map(int, line[2:].split("|"))))
            r["name"] = NAMES[r["layer"]]
            rows.append(r)
            print(f"      {r['name']:<26}lib {r['lib']:>9}   vertical {r['vert']:>8} cycles   "
                  f"exact {r['exact']}/{r['images']}")
        elif ready and line == "DONE":
            return rows
        if not ready and time.time() - t0 > 10:
            print("  >>> no answer to the software reset: press RST once")
            t0 = time.time() + 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--run", default=os.path.join("runs", "w1_s42"), help="run folder with model_symbex.h")
    ap.add_argument("--images", type=int, default=8, help="random images checked bit by bit per layer")
    ap.add_argument("--dcache64", action="store_true", help="64 KB data cache (flash reads; ESP-DL's setting)")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    run_dir = args.run if os.path.isabs(args.run) else os.path.join(HERE, args.run)
    write_project(args.port, run_dir, args.images, args.dcache64)
    if not args.no_upload:
        print("[*] building and flashing the ESP32-S3 ...")
        if subprocess.run(["pio", "run", "-d", PROJECT, "-t", "upload"]).returncode != 0:
            sys.exit(f"[X] build or upload failed (pio run -d {PROJECT} -t upload)")
    rows = read_results(args.port)
    ms = lambda c: c / 240e3
    print("\n    RESNET-8 CONV LAYERS ON THE ESP32-S3, measured  (ms, best of 5; library weights in flash)\n")
    print(f"    {'layer':<26}{'library':>9}{'vertical':>10}{'(flash)':>9}{'speedup':>9}{'planes':>8}{'pcx':>6}"
          f"{'rt lists':>9}{'lists KB':>10}{'thr KB':>8}{'weights KB':>12}   exact")
    tot = dict(lib=0, vert=0, vert_flash=0, vert_rt=0, lib_cold=0, vert_cold=0, vert_thr=0, vert_thr_cold=0, thr_rt_bytes=0, vert_fast=0, vert_both=0, vert_both_cold=0, lists_bit=0, lists_byte=0, list_bytes=0, thr_bytes=0, weight_bytes=0)
    for r in rows:
        for k in tot:
            tot[k] += r[k]
        ok = "yes" if r["exact"] == r["images"] and r["exact_flash"] and r["exact_rt"] and r["exact_thr"] and r["exact_fast"] and r["exact_both"] else f"NO [X] {r['exact']}/{r['images']}"
        print(f"    {r['name']:<26}{ms(r['lib']):>9.2f}{ms(r['vert']):>10.2f}{ms(r['vert_flash']):>9.2f}"
              f"{r['lib'] / r['vert']:>8.2f}x{100 * r['planes'] / r['vert']:>7.0f}%"
              f"{100 * (r['pcx'] - r['planes']) / r['vert']:>5.0f}%{ms(r['vert_rt']):>9.2f}{r['list_bytes'] / 1024:>10.1f}"
              f"{r['thr_bytes'] / 1024:>8.1f}{r['weight_bytes'] / 1024:>12.1f}   {ok}")
    print(f"    {'all convs':<26}{ms(tot['lib']):>9.2f}{ms(tot['vert']):>10.2f}{ms(tot['vert_flash']):>9.2f}"
          f"{tot['lib'] / tot['vert']:>8.2f}x{'':>13}{ms(tot['vert_rt']):>9.2f}{tot['list_bytes'] / 1024:>10.1f}{tot['thr_bytes'] / 1024:>8.1f}"
          f"{tot['weight_bytes'] / 1024:>12.1f}")
    print("\n    COLD CACHE (flash data not cached before the run: one layer of a real inference)\n")
    print(f"    {'layer':<26}{'library':>9}{'vertical':>10}{'speedup':>9}   |  thresholds per border kind: "
          f"{'warm':>6}{'cold':>7}{'speedup':>9}{'thr KB':>8}")
    for r in rows + [dict(name="all convs", **tot)]:
        print(f"    {r['name']:<26}{ms(r['lib_cold']):>9.2f}{ms(r['vert_cold']):>10.2f}{r['lib_cold'] / r['vert_cold']:>8.2f}x"
              f"   |  {'':>28}{ms(r['vert_thr']):>6.2f}{ms(r['vert_thr_cold']):>7.2f}"
              f"{r['lib_cold'] / r['vert_thr_cold']:>8.2f}x{r['thr_rt_bytes'] / 1024:>8.1f}")
    print("\n    ONLY THE WEIGHTS STORED (lists from a byte table + thresholds per border kind, built on the chip)\n")
    print(f"    {'layer':<26}{'library':>9}{'fast lists':>11}{'warm':>7}{'cold':>7}{'speedup cold':>14}{'stored KB':>11}{'build bit':>11}{'build byte':>12}")
    for r in rows + [dict(name="all convs", **tot)]:
        print(f"    {r['name']:<26}{ms(r['lib_cold']):>9.2f}{ms(r['vert_fast']):>11.2f}{ms(r['vert_both']):>7.2f}"
              f"{ms(r['vert_both_cold']):>7.2f}{r['lib_cold'] / r['vert_both_cold']:>13.2f}x"
              f"{(r['weight_bytes'] + r['thr_rt_bytes']) / 1024:>11.1f}{ms(r['lists_bit']):>11.2f}{ms(r['lists_byte']):>12.2f}")
    print("\n    fast lists: table-built lists with the stored threshold planes (warm). library column: cold."
          "\n    build bit / byte: the list builders alone (all outputs of the layer, warm), bit by bit / byte table.")
    print("\n    rt lists: the lists built at run time from the 1-bit weights (no list bytes), threshold planes in"
          " flash.\n    planes / pcx: share of the vertical time spent building the shifted planes / counting the windows."
          "\n    Not in the vertical time: HWC <-> planes at the ends, OR pool and OR on planes, the classifier.\n")
    out = os.path.join(HERE, "results", "symbex", "s3_vertical_layers" + ("_dcache64" if args.dcache64 else "") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(board="esp32s3", mhz=240, data_cache_kb=64 if args.dcache64 else 32, run=os.path.relpath(run_dir, HERE), images=args.images,
                       inputs="random RGB images through the library's network", rows=rows), f, indent=1)
    print(f"    saved {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
