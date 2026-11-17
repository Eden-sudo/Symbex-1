/*
 * vertical1d.h -- EXPERIMENT (benchmarks/experiments/vertical1d): lib_symbex's vertical engine (docs/vertical.md) for
 * the binary 1D convolutions of symbex_bool1d.h. The same layer struct, the same time-major streams in and out, the
 * same output bits as symbex_bool1d_conv; it lives here, not in lib_symbex, until it is proven.
 *
 * The standard 1D layer counts one window at a time against every weight row (on the S3: 128-bit XOR + a bit count,
 * then the sum across the register for every output channel and step). Here the loops are turned around:
 *     planes     the input stream turned into channel planes (bit t of plane c = channel c at step t), then shifted by
 *                every kernel tap: plane k * in_ch + c of a tile holds channel c at steps t0 + k .. t0 + k + 127, the
 *                order of the weight row bits. 128 conv steps ("lanes") per 128-bit plane, no sum across a register
 *     pcx        the ones of every window, counted vertically (Harley-Seal carry-save adders, 128 lanes at a time)
 *     count      per output channel, the planes of its weights that are 1 summed the same way (a list of plane offsets
 *                built from the 1-bit row, one byte of weights per step)
 *     threshold  raw = n - 2 (pcx + pw - 2 count) >= thr  <=>  2 count - pcx >= E = ceil((thr + 2 pw - n) / 2); a
 *                bit-sliced add of 2 count + (2048 - pcx) + (2048 - E) gives the 128 output bits in its bit 12
 *     pool       OR of `pool` consecutive lanes (a power of two up to 32), packed, then the planes turned back into the
 *                time-major stream (transpose32 when the channels are a multiple of 32)
 * The counting code (CSA, hs_gather, hs_contig, count_finish, epilogue3, list_table, make_list, transpose32) is a copy
 * of lib_symbex/src/layers/conv/symbex_bool_vertical.c, where it is static.
 *
 * Shapes: stride 1, n = kernel * in_ch <= 1016, any channel count, pool 1 or a power of two up to 32 (other pools: the
 * output bits are set one by one). work: V1D_WORDS(...) words, 16-byte aligned, zeroed before the first call (static
 * storage), one block per instance (it also holds the list table, built on the first call). The layers of one network
 * may share a block of the largest V1D_WORDS: the list table sits first, the same place for all.
 * On the ESP32-S3 the counts run on PIE (128-bit logic); elsewhere the same steps in portable C.
 */
#ifndef V1D_VERTICAL1D_H
#define V1D_VERTICAL1D_H

#if defined(__has_include)                       /* CONFIG_IDF_TARGET_ESP32S3 (ESP-IDF) lives in sdkconfig.h */
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif
#include <stdint.h>
#include <string.h>
#include "layers/conv/symbex_bool1d.h"
#include "hal/symbex_hal.h"                      /* SYMBEX_FAST_CODE */

#if defined(CONFIG_IDF_TARGET_ESP32S3) && !defined(V1D_NO_PIE)
#define V1D_PIE 1
#endif

/* the work block of one layer, in 32-bit words (every part a multiple of 4 words: 16-byte aligned) */
#define V1D_TILES(steps) (((steps) + 127) / 128)
#define V1D_KR(n) (((n) + 7) / 8 * 8)                                       /* n rounded up to 8: the zero plane */
#define V1D_K8(n) (V1D_KR(n) + 8)                                           /* planes per tile */
#define V1D_PW(in_len, kernel, steps) (((((in_len) + V1D_TILES(steps) * 128 + (kernel) + 31) >> 5) + 2 + 3) & ~3)
#define V1D_WORDS(in_ch, out_ch, kernel, pool, in_len, out_len)                                                  \
    (1280 + 4                                                              /* the list table, its flag: first, \
                                                                              shared by the layers of one block */ \
     + (in_ch) * V1D_PW(in_len, kernel, (out_len) * (pool))                /* channel planes */                  \
     + V1D_K8((kernel) * (in_ch)) * 4                                      /* shifted planes of one tile */      \
     + 2 * (V1D_K8((kernel) * (in_ch)) / 8 + 8) * 4                        /* eights and their next level */     \
     + (16 + 16 + 12 + 12 + 1) * 4                                         /* count, pcx, 2048-pcx, 2048-E, out */ \
     + (out_ch) * V1D_TILES((out_len) * (pool)) * 4                        /* pooled output planes */            \
     + ((V1D_K8((kernel) * (in_ch)) + 16) / 2 + 3) / 4 * 4)                /* the plane list */
#define V1D_MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct { uint32_t w[4]; } __attribute__((aligned(16))) v1d_plane;
#define V1D_EP 12                   /* planes of the threshold operands (every value < 4096) */

struct V1DCount {
    v1d_plane *ba, *bb;
#if !defined(V1D_PIE)
    v1d_plane q[3];
#endif
};

/* ================= vertical counting (copied from symbex_bool_vertical.c) ================= */
#if defined(V1D_PIE)
#define V1D_CSA(A_, B_, C_, T_) "ee.xorq " T_ ", " A_ ", " B_ " \n\t ee.andq " B_ ", " A_ ", " B_ " \n\t ee.xorq " A_ ", " T_ ", " C_ " \n\t" \
                                "ee.andq " T_ ", " T_ ", " C_ " \n\t ee.orq " B_ ", " B_ ", " T_ " \n\t"
#define V1D_HS8(L0, L1, L2, L3, L4, L5, L6, L7) L0 L1 V1D_CSA("q0","q3","q4","q5") L2 L3 V1D_CSA("q0","q4","q5","q6") V1D_CSA("q1","q3","q4","q5") \
    L4 L5 V1D_CSA("q0","q4","q5","q6") L6 L7 V1D_CSA("q0","q5","q6","q7") V1D_CSA("q1","q4","q5","q6") V1D_CSA("q2","q3","q4","q5") \
    "ee.vst.128.ip q3, %[o], 16 \n\t"
#define V1D_CL(Q) "ee.vld.128.ip " Q ", %[p], 16 \n\t"

SYMBEX_FAST_CODE static void v1d_hs_gather(struct V1DCount* S, const uint8_t* p, const uint16_t* l, int groups, v1d_plane* o) {
    (void)S;
    uint32_t d0, d1;
    for (int g = 0; g < groups; g++)
        __asm__ __volatile__(
            "l16ui %[d0], %[l], 0 \n\t l16ui %[d1], %[l], 2 \n\t"
            "ee.vld.128.xp q3, %[p], %[d0] \n\t ee.vld.128.xp q4, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 4 \n\t l16ui %[d1], %[l], 6 \n\t"
            V1D_CSA("q0","q3","q4","q5")
            "ee.vld.128.xp q4, %[p], %[d0] \n\t ee.vld.128.xp q5, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 8 \n\t l16ui %[d1], %[l], 10 \n\t"
            V1D_CSA("q0","q4","q5","q6") V1D_CSA("q1","q3","q4","q5")
            "ee.vld.128.xp q4, %[p], %[d0] \n\t ee.vld.128.xp q5, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 12 \n\t l16ui %[d1], %[l], 14 \n\t"
            V1D_CSA("q0","q4","q5","q6")
            "ee.vld.128.xp q5, %[p], %[d0] \n\t ee.vld.128.xp q6, %[p], %[d1] \n\t"
            "addi %[l], %[l], 16 \n\t"
            V1D_CSA("q0","q5","q6","q7") V1D_CSA("q1","q4","q5","q6") V1D_CSA("q2","q3","q4","q5")
            "ee.vst.128.ip q3, %[o], 16 \n\t"
            : [p] "+r"(p), [l] "+r"(l), [o] "+r"(o), [d0] "=&r"(d0), [d1] "=&r"(d1) : : "memory");
}
SYMBEX_FAST_CODE static void v1d_hs_contig(struct V1DCount* S, const v1d_plane* p, int groups, v1d_plane* o) {
    (void)S;
    for (int g = 0; g < groups; g++)
        __asm__ __volatile__(V1D_HS8(V1D_CL("q3"), V1D_CL("q4"), V1D_CL("q4"), V1D_CL("q5"), V1D_CL("q4"), V1D_CL("q5"),
                                     V1D_CL("q5"), V1D_CL("q6"))
                             : [p] "+r"(p), [o] "+r"(o) : : "memory");
}
static inline void v1d_zero_planes(v1d_plane* p, int count) {
    if (count <= 0) return;
    __asm__ __volatile__("ee.zero.q q3");
    while (count--) __asm__ __volatile__("ee.vst.128.ip q3, %0, 16" : "+r"(p) : : "memory");
}
static inline void v1d_zero3(struct V1DCount* S) { (void)S; __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t ee.zero.q q2"); }
static inline void v1d_store3(struct V1DCount* S, v1d_plane* d) {
    (void)S;
    __asm__ __volatile__("ee.vst.128.ip q0, %0, 16 \n\t ee.vst.128.ip q1, %0, 16 \n\t ee.vst.128.ip q2, %0, 16" : "+r"(d) : : "memory");
}
#define V1D_E3(KIN, KOUT) "ee.vld.128.ip q3, %[a], 16 \n\t ee.vld.128.ip q4, %[b], 16 \n\t ee.vld.128.ip q5, %[x], 16 \n\t" \
    "ee.xorq q6, q3, q4 \n\t ee.andq " KOUT ", q3, q4 \n\t ee.andq q3, q6, q5 \n\t ee.xorq q6, q6, q5 \n\t" \
    "ee.orq " KOUT ", " KOUT ", q3 \n\t ee.xorq q7, q6, " KIN " \n\t ee.andq q6, q6, " KIN " \n\t" \
    "ee.andq q7, q7, q0 \n\t ee.orq q0, q6, q7 \n\t"
SYMBEX_FAST_CODE static void v1d_epilogue3(const v1d_plane* a, const v1d_plane* b, const v1d_plane* x, v1d_plane* out) {
    __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t"
                         V1D_E3("q1","q2") V1D_E3("q2","q1") V1D_E3("q1","q2") V1D_E3("q2","q1") V1D_E3("q1","q2") V1D_E3("q2","q1")
                         V1D_E3("q1","q2") V1D_E3("q2","q1") V1D_E3("q1","q2") V1D_E3("q2","q1") V1D_E3("q1","q2") V1D_E3("q2","q1")
                         "ee.xorq q2, q1, q0 \n\t ee.vst.128.ip q2, %[o], 0 \n\t"
                         : [a] "+r"(a), [b] "+r"(b), [x] "+r"(x), [o] "+r"(out) : : "memory");
}
#else
static inline void v1d_csa(v1d_plane* a, v1d_plane* b, const v1d_plane* c) {
    for (int w = 0; w < 4; w++) {
        const uint32_t t = a->w[w] ^ b->w[w];
        b->w[w] = (a->w[w] & b->w[w]) | (t & c->w[w]);
        a->w[w] = t ^ c->w[w];
    }
}
SYMBEX_FAST_CODE static void v1d_hs8(v1d_plane* Q, const v1d_plane* x0, const v1d_plane* x1, const v1d_plane* x2,
                                     const v1d_plane* x3, const v1d_plane* x4, const v1d_plane* x5, const v1d_plane* x6,
                                     const v1d_plane* x7, v1d_plane* o) {
    v1d_plane a = *x0, b = *x1, c, d, e, f;
    v1d_csa(&Q[0], &a, &b);
    c = *x2; d = *x3;
    v1d_csa(&Q[0], &c, &d);
    v1d_csa(&Q[1], &a, &c);
    c = *x4; d = *x5;
    v1d_csa(&Q[0], &c, &d);
    e = *x6; f = *x7;
    v1d_csa(&Q[0], &e, &f);
    v1d_csa(&Q[1], &c, &e);
    v1d_csa(&Q[2], &a, &c);
    *o = a;
}
SYMBEX_FAST_CODE static void v1d_hs_gather(struct V1DCount* S, const uint8_t* p, const uint16_t* l, int groups, v1d_plane* o) {
    for (int g = 0; g < groups; g++, l += 8, o++) {
        const v1d_plane* x[8];
        for (int i = 0; i < 8; i++) { x[i] = (const v1d_plane*)p; p += l[i]; }
        v1d_hs8(S->q, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], o);
    }
}
SYMBEX_FAST_CODE static void v1d_hs_contig(struct V1DCount* S, const v1d_plane* p, int groups, v1d_plane* o) {
    for (int g = 0; g < groups; g++, p += 8, o++) v1d_hs8(S->q, p, p + 1, p + 2, p + 3, p + 4, p + 5, p + 6, p + 7, o);
}
static inline void v1d_zero_planes(v1d_plane* p, int count) { if (count > 0) memset(p, 0, (size_t)count * sizeof *p); }
static inline void v1d_zero3(struct V1DCount* S) { memset(S->q, 0, sizeof S->q); }
static inline void v1d_store3(struct V1DCount* S, v1d_plane* d) { d[0] = S->q[0]; d[1] = S->q[1]; d[2] = S->q[2]; }
SYMBEX_FAST_CODE static void v1d_epilogue3(const v1d_plane* a, const v1d_plane* b, const v1d_plane* x, v1d_plane* out) {
    for (int w = 0; w < 4; w++) {
        uint32_t chain = 0, kin = 0;
        for (int i = 0; i < V1D_EP; i++) {
            const uint32_t A = a[i].w[w], B = b[i].w[w], X = x[i].w[w], ab = A ^ B, s = ab ^ X;
            const uint32_t kout = (A & B) | (ab & X);
            chain = (s & kin) | ((s ^ kin) & chain);
            kin = kout;
        }
        out->w[w] = kin ^ chain;
    }
}
#endif

SYMBEX_FAST_CODE static void v1d_count_finish(struct V1DCount* S, int n, v1d_plane* bits, int need) {
    v1d_store3(S, bits);
    int lvl = 1;
    v1d_plane *src = S->ba, *dst = S->bb;
    while (n > 0) {
        const int g = (n + 7) >> 3;
        v1d_zero_planes(src + n, g * 8 - n);
        v1d_zero3(S); v1d_hs_contig(S, src, g, dst); v1d_store3(S, bits + 3 * lvl); lvl++;
        if (n < 8) break;
        v1d_plane* t = src; src = dst; dst = t; n = g;
    }
    v1d_zero_planes(bits + 3 * lvl, need - 3 * lvl);
}

typedef struct { uint16_t d[7]; uint16_t first, last, n; } V1DListByte;
static void v1d_list_table(V1DListByte* lt) {
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
static int v1d_make_list(const uint32_t* row, int K, const V1DListByte* lt, uint16_t* d, uint32_t* start) {
    const uint8_t* rb = (const uint8_t*)row;
    const int bytes = K / 8;
    uint16_t* q = d - 1;
    uint32_t prev = 0;
    for (int j = 0; j < bytes; j++) {
        const V1DListByte* e = &lt[rb[j]];
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
/* a[r] bit c -> a[c] bit r */
SYMBEX_FAST_CODE static void v1d_transpose32(uint32_t a[32]) {
    uint32_t m = 0x0000FFFFu;
    for (int j = 16; j; j >>= 1, m ^= m << j)
        for (int k = 0; k < 32; k = (k + j + 1) & ~j) {
            const uint32_t t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
        }
}

/* ================= the layer ================= */
/* the stream bits [bit, bit + n) (n <= 32), least significant first; reads one word past them at most */
static inline uint32_t v1d_bits(const uint32_t* s, uint32_t bit, int n) {
    const uint32_t* p = s + (bit >> 5);
    const uint32_t sh = bit & 31;
    const uint32_t w = sh ? (p[0] >> sh) | (p[1] << (32 - sh)) : p[0];
    return n == 32 ? w : w & ((1u << n) - 1u);
}
/* bits 0, 4, 8, ... 28 of x -> bits 0..7 */
static inline uint32_t v1d_pack4(uint32_t x) {
    x &= 0x11111111u;
    x = (x | (x >> 3)) & 0x03030303u;
    x = (x | (x >> 6)) & 0x000F000Fu;
    return (x | (x >> 12)) & 0xFFu;
}

SYMBEX_FAST_CODE static void v1d_conv(const struct SymbexBool1D* L, const uint32_t* in, uint32_t* out, uint32_t* work) {
    const int cin = L->in_ch, cout = L->out_ch, kq = L->kernel, pool = L->pool, n = kq * cin, K8 = V1D_K8(n);
    const int in_len = L->in_len, out_len = L->out_len, steps = out_len * pool, tiles = V1D_TILES(steps);
    const int PW = V1D_PW(in_len, kq, steps);
    const int pow2 = pool <= 32 && (pool & (pool - 1)) == 0;
    /* carve the work block (the order of V1D_WORDS) */
    uint32_t* LTW = work;                                /* the list table first: the same place for every layer */
    V1DListByte* lt = (V1DListByte*)LTW;
    if (LTW[1280] != 0x56314431u) { v1d_list_table(lt); LTW[1280] = 0x56314431u; }
    uint32_t* CP = work + 1284;
    v1d_plane* X = (v1d_plane*)(CP + (size_t)cin * PW);
    v1d_plane* BA = X + K8;
    v1d_plane* BB = BA + K8 / 8 + 8;
    v1d_plane* CB = BB + K8 / 8 + 8;                     /* CB[0] = 0 (zeroed below, never written): CB = 2 count */
    v1d_plane* PXB = CB + 16;
    v1d_plane* AP = PXB + 16;                            /* 2048 - pcx */
    v1d_plane* BT = AP + V1D_EP;                         /* 2048 - E */
    v1d_plane* PO = BT + V1D_EP;                         /* one output plane */
    uint32_t* OUTP = (uint32_t*)(PO + 1);                /* pooled output planes: cout x tiles x 4 words */
    uint16_t* LIST = (uint16_t*)(OUTP + (size_t)cout * tiles * 4);
    struct V1DCount S = {.ba = BA, .bb = BB};

    /* 1. channel planes: bit t of plane c = stream bit t * cin + c (bits past in_len stay 0) */
    memset(CP, 0, (size_t)cin * PW * 4);
    if ((cin & 31) == 0) {
        const int sw = cin / 32;
        uint32_t a[32];
        for (int b = 0; b * 32 < in_len; b++)
            for (int cw = 0; cw < sw; cw++) {
                for (int i = 0; i < 32; i++) a[i] = b * 32 + i < in_len ? in[(size_t)(b * 32 + i) * sw + cw] : 0;
                v1d_transpose32(a);
                for (int c = 0; c < 32; c++) CP[(size_t)(cw * 32 + c) * PW + b] = a[c];
            }
    } else {
        for (int t = 0; t < in_len; t++)
            for (int c0 = 0; c0 < cin; c0 += 32) {
                const int m = cin - c0 < 32 ? cin - c0 : 32;
                for (uint32_t v = v1d_bits(in, (uint32_t)t * cin + c0, m); v; v &= v - 1) {
                    const int c = c0 + __builtin_ctz(v);
                    CP[(size_t)c * PW + (t >> 5)] |= 1u << (t & 31);
                }
            }
    }
    if (!pow2) memset(out, 0, (((size_t)out_len * cout + 31) / 32 + 1) * 4);
    v1d_zero_planes(X + n, K8 - n);                      /* the planes past n and the zero plane of the lists */
    v1d_zero_planes(CB, 1);                              /* other layers may share the block */
    const V1DListByte* ltc = lt;
    for (int t = 0; t < tiles; t++) {
        const int t0 = t * 128;
        /* 2. the shifted planes of this tile: plane k * cin + c = channel c at steps t0 + k .. t0 + k + 127 */
        for (int k = 0; k < kq; k++) {
            const int off = t0 + k, wi = off >> 5, sh = off & 31;
            for (int c = 0; c < cin; c++) {
                const uint32_t* p = CP + (size_t)c * PW + wi;
                uint32_t* d = X[k * cin + c].w;
                if (sh == 0) { d[0] = p[0]; d[1] = p[1]; d[2] = p[2]; d[3] = p[3]; }
                else {
                    const int r = 32 - sh;
                    d[0] = (p[0] >> sh) | (p[1] << r); d[1] = (p[1] >> sh) | (p[2] << r);
                    d[2] = (p[2] >> sh) | (p[3] << r); d[3] = (p[3] >> sh) | (p[4] << r);
                }
            }
        }
        /* 3. pcx, kept as 2048 - pcx = ~pcx + 1 with bit 11 flipped */
        v1d_zero3(&S); v1d_hs_contig(&S, X, K8 / 8, BA); v1d_count_finish(&S, K8 / 8, PXB, V1D_EP);
        for (int w = 0; w < 4; w++) {
            uint32_t c = ~0u;
            for (int i = 0; i < V1D_EP; i++) {
                const uint32_t a = ~PXB[i].w[w], s = a ^ c;
                c &= a;
                AP[i].w[w] = i == V1D_EP - 1 ? ~s : s;
            }
        }
        /* 4. every output channel: its list, count, threshold, pool */
        for (int o = 0; o < cout; o++) {
            uint32_t start;
            const int cnt = v1d_make_list(L->weights + (size_t)o * L->row_words, V1D_KR(n), ltc, LIST + 1, &start);
            const int n8 = (cnt + 7) & ~7;
            int64_t thr = L->thresholds[o];
            if (thr < -n) thr = -n;
            if (thr > n + 1) thr = n + 1;
            int64_t e = (thr + 2 * cnt - n + 1) >> 1;
            if (e < -n) e = -n;
            if (e > n + 1) e = n + 1;
            const uint32_t B = (uint32_t)(2048 - e);
            for (int i = 0; i < V1D_EP; i++) {
                const uint32_t m = 0u - ((B >> i) & 1u);
                BT[i].w[0] = m; BT[i].w[1] = m; BT[i].w[2] = m; BT[i].w[3] = m;
            }
            v1d_zero3(&S);
            if (n8) v1d_hs_gather(&S, (const uint8_t*)X + start, LIST + 1, n8 / 8, BA);
            v1d_count_finish(&S, cnt >= 8 ? n8 / 8 : 0, CB + 1, V1D_EP - 1);
            v1d_epilogue3(AP, BT, CB, PO);
            if (pow2) {
                /* OR of `pool` lanes at the lanes that are multiples of pool, packed: 128 / pool bits per tile */
                uint32_t* op = OUTP + ((size_t)o * tiles + t) * 4;
                if (pool == 1) { op[0] = PO->w[0]; op[1] = PO->w[1]; op[2] = PO->w[2]; op[3] = PO->w[3]; }
                else if (pool == 4) {
                    uint32_t y[4];
                    for (int w = 0; w < 4; w++) { uint32_t v = PO->w[w]; v |= v >> 1; v |= v >> 2; y[w] = v1d_pack4(v); }
                    op[0] = y[0] | (y[1] << 8) | (y[2] << 16) | (y[3] << 24);
                } else {
                    const int per = 32 / pool;                 /* pooled bits per word: 128 / pool <= 64 per tile */
                    op[0] = 0; op[1] = 0;
                    for (int w = 0; w < 4; w++) {
                        uint32_t v = PO->w[w];
                        for (int s = 1; s < pool; s <<= 1) v |= v >> s;
                        for (int i = 0; i < per; i++)
                            if ((v >> (i * pool)) & 1u) { const int b = w * per + i; op[b >> 5] |= 1u << (b & 31); }
                    }
                }
            } else {
                for (int l = 0; l < 128 && t0 + l < steps; l++)
                    if ((PO->w[l >> 5] >> (l & 31)) & 1u) {
                        const uint32_t i = (uint32_t)((t0 + l) / pool) * cout + o;
                        out[i >> 5] |= 1u << (i & 31);
                    }
            }
        }
    }
    if (!pow2) return;
    /* 5. the pooled planes back into the time-major stream: bit q * cout + o, q < out_len */
    const int per_tile = 128 / pool;                                       /* pooled bits per tile */
    const size_t out_words = ((size_t)out_len * cout + 31) / 32 + 1;
#define V1D_OUTBIT(o, q) ((OUTP[((size_t)(o) * tiles + (q) / per_tile) * 4 + ((q) % per_tile) / 32] >> ((q) % per_tile % 32)) & 1u)
    if ((cout & 31) == 0 && per_tile % 32 == 0) {
        const int sw = cout / 32;
        uint32_t a[32];
        for (int b = 0; b * 32 < out_len; b++) {
            const int q0 = b * 32, tile = q0 / per_tile, wq = (q0 % per_tile) / 32;
            for (int cw = 0; cw < sw; cw++) {
                for (int c = 0; c < 32; c++) a[c] = OUTP[((size_t)(cw * 32 + c) * tiles + tile) * 4 + wq];
                v1d_transpose32(a);
                for (int i = 0; i < 32 && q0 + i < out_len; i++) out[(size_t)(q0 + i) * sw + cw] = a[i];
            }
        }
        out[out_words - 1] = 0;
    } else {
        memset(out, 0, out_words * 4);
        for (int q = 0; q < out_len; q++)
            for (int o = 0; o < cout; o++)
                if (V1D_OUTBIT(o, q)) { const uint32_t i = (uint32_t)q * cout + o; out[i >> 5] |= 1u << (i & 31); }
    }
#undef V1D_OUTBIT
}

#endif /* V1D_VERTICAL1D_H */
