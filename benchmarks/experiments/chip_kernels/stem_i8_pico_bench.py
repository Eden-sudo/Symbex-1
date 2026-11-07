"""
stem_i8_pico_bench.py -- where does the int8 stem's time go on the Pico 2, and does a ring-of-rows kernel beat it?

On the Pico 2 the int8 stem is most of ResNet-8's time (M33 14.8 ms, Hazard3 18.6 ms of ~70 ms). lib_symbex's
symbex_bool_stem_i8 builds a 27-byte window per output pixel, then one dot product per output channel (SMLAD on the
M33, plain C on Hazard3). This bench times, on the chip:
    lib       symbex_bool_stem_i8 itself (the library's code, copied from Symbex1_PIO/lib/lib_symbex)
    window    its window building alone (the same loop, no dot products)
    dots      its dot products + threshold + packing alone (a fixed window, every output pixel)
    ring      a candidate kernel (in this file): the image kept as a ring of 3 padded rows (pixel ^ 0x80 = pixel - 128,
              a zero pixel left and right), so no window is built; every channel reads its 3 x 9 bytes straight from
              the rows. M33: the rows read as unaligned words (3 per kernel row, the 3 extra bytes meet zero weights),
              split once per pixel by SXTB16 into even / odd halves; the weights split once per call; 2 channels per
              pass, SMLAD. Hazard3: the 27 inputs widened once per pixel, 2 channels per pass, plain C.
    rows      the second candidate, the fastest (in lib_symbex since Nov 6 as stem3_rows: `lib` now runs it too):
              the same ring, then strips of 16 output pixels; per channel and kernel row the 9 weights stay in
              registers while the strip streams past. M33: the strip's rows split once into SXTB16 halves (5 words
              per pixel), then 5 SMLAD per pixel. Hazard3: 2 channels per pass, each byte loaded once for both.
Every kernel is checked against a plain C reference on the chip, otherwise its time is not trusted.

    python stem_i8_pico_bench.py --board pico2_arm --port /dev/ttyACM1      # Cortex-M33
    python stem_i8_pico_bench.py --board pico2_riscv --port /dev/ttyACM1    # Hazard3
    python stem_i8_pico_bench.py --board pico2_arm --port /dev/ttyACM1 --code ram   # the loops in RAM

Shapes (as stem_i8_pie_bench.py): ResNet-8's stem (32 x 32, 16 channels, stride 1), 40 channels with stride 2 and
a 48 x 48 image with 24 channels. Weights in flash (const), the image in RAM. --code ram: the timed loops in RAM
(-DSYMBEX_CODE_IN_RAM). Times: best of 7, micros(). Results: results/stem_i8_pico_bench_<board>_<code>.json.
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

# (name, in_hw, out_ch, kernel, stride): the first is cnn_cifar_resnet8's stem
SHAPES = [("resnet8", 32, 16, 3, 1), ("s2_c40", 32, 40, 3, 2), ("w48_c24", 48, 24, 3, 1)]
KINDS = ("lib", "window", "dots", "ring", "rows")

FIRMWARE = r"""
#include <Arduino.h>
#include <string.h>
#include <stdint.h>
#if defined(__ARM_FEATURE_SIMD32)
#include <arm_acle.h>
#endif
#include "hal/symbex_hal.h"
#include "layers/conv/symbex_bool.h"
#include "bench_data.h"
#define NOW() ((uint32_t)micros())
#define REPS 7

/* ---- the candidate kernel: a ring of 3 padded rows ----------------------------------------------------------- */

#define RING_MAX_W 128
#define RING_BYTES ((((RING_MAX_W + 2) * 3 + 3) + 3) & ~3)   /* padded row + 3 bytes read past it (zero weights) */
#define GROUP 8                                            /* channels per pass over the image (stack: weights) */

static inline uint32_t ld32(const int8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* the layer of symbex_bool_stem_i8 (k = 3, padding 1, stride 1 or 2), out HWC bits. 0 = done, -1 = not this kernel's */
static int SYMBEX_FAST_CODE __attribute__((noinline)) stem3_ring(const uint8_t* img, int H, int W, int s, int out_h, int out_w,
                                                const int8_t* w, int row_bytes, const int32_t* thr, int n_out,
                                                uint32_t* out) {
    if (W > RING_MAX_W || (s != 1 && s != 2)) return -1;
    int8_t ring[3][RING_BYTES] __attribute__((aligned(4)));
    const int rb = (W + 2) * 3, pout = (n_out + 31) / 32;
    memset(out, 0, (size_t)out_h * out_w * pout * 4);
    for (int g = 0; g < n_out; g += GROUP) {
        const int ng = n_out - g < GROUP ? n_out - g : GROUP;
#if defined(__ARM_FEATURE_SIMD32)
        uint32_t we[GROUP][9], wo[GROUP][9];               /* [kernel row * 3 + word]: even / odd bytes, 16 bits */
        for (int c = 0; c < GROUP; c++)
            for (int kh = 0; kh < 3; kh++)
                for (int j = 0; j < 3; j++) {
                    uint32_t v = 0;
                    for (int b = 0; b < 4; b++) {
                        const int i = 4 * j + b;
                        const int8_t q = (c < ng && i < 9) ? w[(g + c) * row_bytes + kh * 9 + i] : 0;
                        v |= (uint32_t)(uint8_t)q << (8 * b);
                    }
                    we[c][kh * 3 + j] = __sxtb16(v);
                    wo[c][kh * 3 + j] = __sxtb16(__ror(v, 8));
                }
#else
        int32_t wk[GROUP][27];
        for (int c = 0; c < GROUP; c++)
            for (int i = 0; i < 27; i++) wk[c][i] = c < ng ? w[(g + c) * row_bytes + i] : 0;
#endif
        int last = -2;                                     /* the last input row in the ring (-1, H: zero rows) */
        for (int oy = 0; oy < out_h; oy++) {
            const int iy0 = oy * s - 1;
            for (int r = last + 1 > iy0 ? last + 1 : iy0; r <= iy0 + 2; r++) {
                int8_t* p = ring[(r + 3) % 3];
                memset(p, 0, RING_BYTES);
                if (r >= 0 && r < H) {
                    const uint8_t* src = img + (size_t)r * W * 3;
                    for (int i = 0; i < W * 3; i++) p[3 + i] = (int8_t)(src[i] ^ 0x80u);
                }
            }
            last = iy0 + 2;
            const int8_t* rows[3] = {ring[(iy0 + 3) % 3], ring[(iy0 + 4) % 3], ring[(iy0 + 5) % 3]};
            uint32_t* o_row = out + (size_t)oy * out_w * pout + g / 32;
            for (int ox = 0; ox < out_w; ox++) {
                const int base = ox * s * 3;
                uint32_t bits = 0;
#if defined(__ARM_FEATURE_SIMD32)
                uint32_t xe[9], xo[9];
                for (int kh = 0; kh < 3; kh++)
                    for (int j = 0; j < 3; j++) {
                        const uint32_t v = ld32(rows[kh] + base + 4 * j);
                        xe[kh * 3 + j] = __sxtb16(v);
                        xo[kh * 3 + j] = __sxtb16(__ror(v, 8));
                    }
                for (int c = 0; c < ng; c += 2) {
                    int32_t a = 0, b = 0;
                    for (int j = 0; j < 9; j++) {
                        a = __smlad(xe[j], we[c][j], a);
                        a = __smlad(xo[j], wo[c][j], a);
                        b = __smlad(xe[j], we[c + 1][j], b);
                        b = __smlad(xo[j], wo[c + 1][j], b);
                    }
                    bits |= (uint32_t)(a >= thr[g + c]) << c;
                    if (c + 1 < ng) bits |= (uint32_t)(b >= thr[g + c + 1]) << (c + 1);
                }
#else
                int32_t x[27];
                for (int kh = 0; kh < 3; kh++)
                    for (int i = 0; i < 9; i++) x[kh * 9 + i] = rows[kh][base + i];
                for (int c = 0; c < ng; c += 2) {
                    int32_t a = 0, b = 0;
                    const int32_t *w0 = wk[c], *w1 = wk[c + 1];
                    for (int i = 0; i < 27; i++) { a += x[i] * w0[i]; b += x[i] * w1[i]; }
                    bits |= (uint32_t)(a >= thr[g + c]) << c;
                    if (c + 1 < ng) bits |= (uint32_t)(b >= thr[g + c + 1]) << (c + 1);
                }
#endif
                o_row[ox * pout] |= bits << (g % 32);
            }
        }
    }
    return 0;
}

/* ---- the second candidate: weights in registers, pixels streamed --------------------------------------------- */

/* the same ring of rows; then per strip of up to STRIP output pixels of an output row, per channel, per kernel row:
   that row's weights stay in registers (M33: 6 words, even / odd halves; Hazard3: 9 ints) while the strip's pixels
   stream past, sums in acc[STRIP]. M33: the strip's 3 rows are split once into even / odd halves (5 words per pixel
   and kernel row, in the order the loop reads them; the 6th, bytes 9-11, would meet zero weights), so the inner loop
   is loads + SMLAD only, the 5 loads first. Hazard3: 2 channels per pass (each input byte loaded once for both). */
#define STRIP 16

static int SYMBEX_FAST_CODE __attribute__((noinline)) stem3_rows(const uint8_t* img, int H, int W, int s, int out_h,
                                                                 int out_w, const int8_t* w, int row_bytes,
                                                                 const int32_t* thr, int n_out, uint32_t* out) {
    if (W > RING_MAX_W || (s != 1 && s != 2)) return -1;
    int8_t ring[3][RING_BYTES] __attribute__((aligned(4)));
    int32_t acc[STRIP], acc2[STRIP];
#if defined(__ARM_FEATURE_SIMD32)
    uint32_t xs[3][STRIP * 5];
#endif
    const int pout = (n_out + 31) / 32;
    memset(out, 0, (size_t)out_h * out_w * pout * 4);
    int last = -2;
    for (int oy = 0; oy < out_h; oy++) {
        const int iy0 = oy * s - 1;
        for (int r = last + 1 > iy0 ? last + 1 : iy0; r <= iy0 + 2; r++) {
            int8_t* p = ring[(r + 3) % 3];
            memset(p, 0, RING_BYTES);
            if (r >= 0 && r < H) {
                const uint8_t* src = img + (size_t)r * W * 3;
                for (int i = 0; i < W * 3; i++) p[3 + i] = (int8_t)(src[i] ^ 0x80u);
            }
        }
        last = iy0 + 2;
        const int8_t* rows[3] = {ring[(iy0 + 3) % 3], ring[(iy0 + 4) % 3], ring[(iy0 + 5) % 3]};
        uint32_t* o_row = out + (size_t)oy * out_w * pout;
        for (int x0 = 0; x0 < out_w; x0 += STRIP) {
            const int np = out_w - x0 < STRIP ? out_w - x0 : STRIP;
#if defined(__ARM_FEATURE_SIMD32)
            for (int kh = 0; kh < 3; kh++)
                for (int i = 0; i < np; i++) {
                    const int8_t* q = rows[kh] + (x0 + i) * s * 3;
                    uint32_t* d = &xs[kh][i * 5];
                    const uint32_t v0 = ld32(q), v1 = ld32(q + 4), v2 = (uint8_t)q[8];
                    d[0] = __sxtb16(v0);
                    d[1] = __sxtb16(__ror(v0, 8));
                    d[2] = __sxtb16(v1);
                    d[3] = __sxtb16(__ror(v1, 8));
                    d[4] = __sxtb16(v2);
                }
#endif
#if defined(__ARM_FEATURE_SIMD32)
            for (int c = 0; c < n_out; c++) {
                for (int i = 0; i < np; i++) acc[i] = 0;
                const int8_t* wc = w + c * row_bytes;
                for (int kh = 0; kh < 3; kh++) {
                    const uint32_t v0 = ld32(wc + kh * 9), v1 = ld32(wc + kh * 9 + 4), v2 = (uint8_t)wc[kh * 9 + 8];
                    const uint32_t e0 = __sxtb16(v0), o0 = __sxtb16(__ror(v0, 8)), e1 = __sxtb16(v1),
                                   o1 = __sxtb16(__ror(v1, 8)), e2 = __sxtb16(v2);
                    const uint32_t* xp = xs[kh];
                    for (int i = 0; i < np; i++, xp += 5) {
                        const uint32_t a0 = xp[0], a1 = xp[1], a2 = xp[2], a3 = xp[3], a4 = xp[4];
                        int32_t a = acc[i];
                        a = __smlad(a0, e0, a);
                        a = __smlad(a1, o0, a);
                        a = __smlad(a2, e1, a);
                        a = __smlad(a3, o1, a);
                        a = __smlad(a4, e2, a);
                        acc[i] = a;
                    }
                }
                const int32_t t_c = thr[c];
                const uint32_t m = 1u << (c % 32);
                uint32_t* o = o_row + x0 * pout + c / 32;
                for (int i = 0; i < np; i++, o += pout)
                    if (acc[i] >= t_c) *o |= m;
            }
#else
            for (int c = 0; c < n_out; c += 2) {
                const int two = c + 1 < n_out;
                for (int i = 0; i < np; i++) acc[i] = acc2[i] = 0;
                for (int kh = 0; kh < 3; kh++) {
                    const int8_t* wk = w + c * row_bytes + kh * 9;
                    const int8_t* wl = two ? wk + row_bytes : wk;  /* one channel left: computed twice, kept once */
                    const int32_t w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6],
                                  w7 = wk[7], w8 = wk[8];
                    const int32_t u0 = wl[0], u1 = wl[1], u2 = wl[2], u3 = wl[3], u4 = wl[4], u5 = wl[5], u6 = wl[6],
                                  u7 = wl[7], u8 = wl[8];
                    const int8_t* xp = rows[kh] + x0 * s * 3;
                    const int step = s * 3;
                    for (int i = 0; i < np; i++, xp += step) {
                        const int32_t x0_ = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3], x4 = xp[4], x5 = xp[5],
                                      x6 = xp[6], x7 = xp[7], x8 = xp[8];
                        acc[i] += x0_ * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + x5 * w5 + x6 * w6 + x7 * w7
                                  + x8 * w8;
                        acc2[i] += x0_ * u0 + x1 * u1 + x2 * u2 + x3 * u3 + x4 * u4 + x5 * u5 + x6 * u6 + x7 * u7
                                   + x8 * u8;
                    }
                }
                for (int q = 0; q < 1 + two; q++) {
                    const int cc = c + q;
                    const int32_t t_c = thr[cc];
                    const int32_t* aq = q ? acc2 : acc;
                    const uint32_t m = 1u << (cc % 32);
                    uint32_t* o = o_row + x0 * pout + cc / 32;
                    for (int i = 0; i < np; i++, o += pout)
                        if (aq[i] >= t_c) *o |= m;
                }
            }
#endif
        }
    }
    return 0;
}

/* ---- the library's parts, apart (copies of symbex_bool_stem_i8's loops) --------------------------------------- */

static volatile int32_t sink;

static void SYMBEX_FAST_CODE __attribute__((noinline)) window_only(const struct SymbexBoolStemI8* L, const uint8_t* img, int8_t* window) {
    const int k = L->kernel, pad = k / 2, s = L->stride, H = L->in_h, W = L->in_w;
    int32_t acc = 0;
    memset(window, 0, SYMBEX_BOOL_STEM_I8_WINDOW);
    for (int oy = 0; oy < L->out_h; oy++) {
        const int iy0 = oy * s - pad;
        for (int ox = 0; ox < L->out_w; ox++) {
            const int ix0 = ox * s - pad;
            if (iy0 >= 0 && ix0 >= 0 && iy0 + k <= H && ix0 + k <= W) {
                for (int kh = 0; kh < k; kh++) {
                    const uint8_t* src = img + ((iy0 + kh) * W + ix0) * 3;
                    int8_t* dst = window + kh * k * 3;
                    for (int i = 0; i < k * 3; i++) dst[i] = (int8_t)(src[i] ^ 0x80u);
                }
            } else {
                for (int kh = 0; kh < k; kh++) {
                    const int iy = iy0 + kh;
                    for (int kw = 0; kw < k; kw++) {
                        const int ix = ix0 + kw;
                        int8_t* dst = window + (kh * k + kw) * 3;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                            const uint8_t* src = img + (iy * W + ix) * 3;
                            dst[0] = (int8_t)(src[0] ^ 0x80u);
                            dst[1] = (int8_t)(src[1] ^ 0x80u);
                            dst[2] = (int8_t)(src[2] ^ 0x80u);
                        } else {
                            dst[0] = dst[1] = dst[2] = 0;
                        }
                    }
                }
            }
            acc += window[(ox + oy) % 27];                 /* the window is used: not optimized away */
        }
    }
    sink = acc;
}

static void SYMBEX_FAST_CODE dot_rows(const int8_t* x, int n, const int8_t* w, int stride, int n_rows, int32_t* out) {
#if defined(__ARM_FEATURE_SIMD32)
    (void)n;
    uint32_t xe[SYMBEX_BOOL_STEM_I8_WINDOW / 4], xo[SYMBEX_BOOL_STEM_I8_WINDOW / 4];
    const int nw = stride >> 2;
    const uint32_t* xw = (const uint32_t*)(const void*)x;
    for (int i = 0; i < nw; i++) { xe[i] = __sxtb16(xw[i]); xo[i] = __sxtb16(__ror(xw[i], 8)); }
    for (int r = 0; r < n_rows; r++) {
        const uint32_t* ww = (const uint32_t*)(const void*)(w + r * stride);
        int32_t a = 0, b = 0;
        for (int i = 0; i < nw; i++) {
            const uint32_t v = ww[i];
            a = __smlad(xe[i], __sxtb16(v), a);
            b = __smlad(xo[i], __sxtb16(__ror(v, 8)), b);
        }
        out[r] = a + b;
    }
#else
    int r = 0;
    for (; r + 4 <= n_rows; r += 4) {
        const int8_t *r0 = w + r * stride, *r1 = r0 + stride, *r2 = r1 + stride, *r3 = r2 + stride;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int i = 0; i < n; i++) {
            const int32_t u = x[i];
            a0 += u * r0[i]; a1 += u * r1[i]; a2 += u * r2[i]; a3 += u * r3[i];
        }
        out[r] = a0; out[r + 1] = a1; out[r + 2] = a2; out[r + 3] = a3;
    }
    for (; r < n_rows; r++) {
        const int8_t* r0 = w + r * stride;
        int32_t a0 = 0;
        for (int i = 0; i < n; i++) a0 += (int32_t)x[i] * r0[i];
        out[r] = a0;
    }
#endif
}

static void SYMBEX_FAST_CODE __attribute__((noinline)) dots_only(const struct SymbexBoolStemI8* L, const int8_t* window, int32_t* dot,
                                                uint32_t* out) {
    const int n = L->kernel * L->kernel * 3, n_out = L->out_ch, pout = (n_out + 31) / 32;
    for (int oy = 0; oy < L->out_h; oy++)
        for (int ox = 0; ox < L->out_w; ox++) {
            dot_rows(window, n, L->weights, L->row_bytes, n_out, dot);
            uint32_t* o_px = &out[(oy * L->out_w + ox) * pout];
            for (int ow = 0, o = 0; ow < pout; ow++) {
                uint32_t bits = 0;
                const int end = (ow + 1) * 32 < n_out ? (ow + 1) * 32 : n_out;
                for (int b = 0; o < end; o++, b++) bits |= (uint32_t)(dot[o] >= L->thresholds[o]) << b;
                o_px[ow] = bits;
            }
        }
}

/* ---- plain C reference ---------------------------------------------------------------------------------------- */

static void stem_ref(const struct SymbexBoolStemI8* L, const uint8_t* img, uint32_t* out) {
    const int k = L->kernel, pad = k / 2, pout = (L->out_ch + 31) / 32;
    memset(out, 0, (size_t)L->out_h * L->out_w * pout * 4);
    for (int oy = 0; oy < L->out_h; oy++)
        for (int ox = 0; ox < L->out_w; ox++)
            for (int o = 0; o < L->out_ch; o++) {
                int32_t sum = 0;
                for (int kh = 0; kh < k; kh++)
                    for (int kw = 0; kw < k; kw++) {
                        const int iy = oy * L->stride - pad + kh, ix = ox * L->stride - pad + kw;
                        if (iy < 0 || iy >= L->in_h || ix < 0 || ix >= L->in_w) continue;
                        for (int c = 0; c < 3; c++)
                            sum += ((int32_t)img[(iy * L->in_w + ix) * 3 + c] - 128)
                                   * L->weights[o * L->row_bytes + (kh * k + kw) * 3 + c];
                    }
                if (sum >= L->thresholds[o]) out[(oy * L->out_w + ox) * pout + o / 32] |= 1u << (o % 32);
            }
}

static uint32_t o_lib[MAX_OUT_WORDS], o_ring[MAX_OUT_WORDS], o_rows[MAX_OUT_WORDS], o_ref[MAX_OUT_WORDS], o_tmp[MAX_OUT_WORDS];
static int8_t win[SYMBEX_BOOL_STEM_I8_WINDOW] __attribute__((aligned(16)));
static int32_t scratch[64];

static void bench_once(void) {
    Serial.printf("STEM I8 PICO BENCH ready (%s, %lu Hz)\n", WEIGHTS_WHERE, (unsigned long)rp2040.f_cpu());
    for (int c = 0; c < N_CASES; c++) {
        const struct SymbexBoolStemI8* L = &CASES[c];
        const size_t nw = (size_t)L->out_h * L->out_w * ((L->out_ch + 31) / 32);
        uint32_t best[5] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        int rc = 0, rc2 = 0;
        for (int r = 0; r < REPS; r++) {
            memset(o_ring, 0xA5, sizeof o_ring);
            memset(o_rows, 0xA5, sizeof o_rows);
            uint32_t t0 = NOW();
            symbex_bool_stem_i8(L, CASE_IMG[c], o_lib, win, scratch);
            uint32_t t1 = NOW();
            window_only(L, CASE_IMG[c], win);
            uint32_t t2 = NOW();
            dots_only(L, win, scratch, o_tmp);
            uint32_t t3 = NOW();
            rc |= stem3_ring(CASE_IMG[c], L->in_h, L->in_w, L->stride, L->out_h, L->out_w, L->weights, L->row_bytes,
                             L->thresholds, L->out_ch, o_ring);
            uint32_t t4 = NOW();
            rc2 |= stem3_rows(CASE_IMG[c], L->in_h, L->in_w, L->stride, L->out_h, L->out_w, L->weights, L->row_bytes,
                              L->thresholds, L->out_ch, o_rows);
            uint32_t t5 = NOW();
            const uint32_t t[5] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4};
            for (int i = 0; i < 5; i++) if (t[i] < best[i]) best[i] = t[i];
        }
        stem_ref(L, CASE_IMG[c], o_ref);
        const int ok_lib = memcmp(o_lib, o_ref, nw * 4) == 0, ok_ring = rc == 0 && memcmp(o_ring, o_ref, nw * 4) == 0,
                  ok_rows = rc2 == 0 && memcmp(o_rows, o_ref, nw * 4) == 0;
        int ones = 0;
        for (size_t i = 0; i < nw; i++) ones += __builtin_popcount(o_ref[i]);
        Serial.printf("R:%s|%lu|%lu|%lu|%lu|%lu|%d|%d|%d|%d\n", CASE_NAMES[c], (unsigned long)best[0],
                      (unsigned long)best[1], (unsigned long)best[2], (unsigned long)best[3], (unsigned long)best[4],
                      ok_lib, ok_ring, ok_rows, ones);
    }
    Serial.printf("DONE\n");
}

void setup() { Serial.begin(115200); }
void loop() {
    bench_once();
    delay(2000);
}
"""


def c_array(ctype, name, values, rom, per_line=24):
    vals = [str(int(v)) for v in np.asarray(values).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return (f"static {'ROM ' if rom else ''}{ctype} {name}[{len(vals)}] __attribute__((aligned(16))) = {{\n  "
            + ",\n  ".join(lines) + "\n};\n")


def write_data(path):
    """random images and int8 weights (-127..127, as the exporter's); thresholds near the middle of each channel's
    dot products (both bit values)"""
    rng = np.random.default_rng(2026)
    parts, cases, max_words = [], [], 0
    for c, (name, hw, ch, k, s) in enumerate(SHAPES):
        n, rb = k * k * 3, (k * k * 3 + 3) & ~3
        img = rng.integers(0, 256, (hw, hw, 3))
        w = np.zeros((ch, rb), np.int64)
        w[:, :n] = rng.integers(-127, 128, (ch, n))
        x = img.astype(np.int64) - 128
        pad = k // 2
        xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
        out_hw = hw // s
        win = np.stack([xp[ky:ky + out_hw * s:s, kx:kx + out_hw * s:s] for ky in range(k) for kx in range(k)], 2)
        dots = win.reshape(out_hw * out_hw, n) @ w[:, :n].T
        thr = np.median(dots, 0).astype(np.int64) + rng.integers(-50, 51, ch)
        parts += [c_array("uint8_t", f"IMG{c}", img, False), c_array("int8_t", f"W{c}", w, True),
                  c_array("int32_t", f"T{c}", thr, True)]
        cases.append(f"{{{ch}, {k}, {s}, {hw}, {hw}, {out_hw}, {out_hw}, {rb}, W{c}, T{c}}}")
        max_words = max(max_words, out_hw * out_hw * ((ch + 31) // 32))
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define ROM const\n#define WEIGHTS_WHERE \"weights in flash\"\n"
                f"#define N_CASES {len(SHAPES)}\n#define MAX_OUT_WORDS {max_words}\n")
        f.write("".join(parts))
        f.write("static const struct SymbexBoolStemI8 CASES[N_CASES] = {\n  " + ",\n  ".join(cases) + "\n};\n")
        f.write("static const uint8_t* const CASE_IMG[N_CASES] = {"
                + ", ".join(f"IMG{c}" for c in range(len(SHAPES))) + "};\n")
        f.write("static const char* const CASE_NAMES[N_CASES] = {"
                + ", ".join(f'"{s[0]}"' for s in SHAPES) + "};\n")


LIB_FILES = {"symbex_and.c": ("ops", "symbex_and.c"), "symbex_bool.c": ("layers", "conv", "symbex_bool.c"),
             "hal_generic.c": ("hal", "generic", "symbex_hal_impl.c"),
             "hal_rp2350.c": ("hal", "rp2350", "symbex_hal_impl.c")}


def write_project(project, port, board, code="flash"):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    for name, rel in LIB_FILES.items():
        shutil.copy(os.path.join(LIB, "src", *rel), os.path.join(src, name))
    write_data(os.path.join(src, "bench_data.h"))
    w("platformio.ini", f"[env:{board}]\nplatform = raspberrypi\nboard = {BOARDS[board]}\nframework = arduino\n"
                        f"board_build.core = earlephilhower\nupload_protocol = picotool\nupload_port = {port}\n"
                        + ("board_build.mcu = rp2350-riscv\n" if board == "pico2_riscv" else "")
                        + "build_flags = -Isrc/include -O2" + (" -DSYMBEX_CODE_IN_RAM" if code == "ram" else "")
                        + "\nbuild_unflags = -Os\n")
    w("src/main.cpp", FIRMWARE)


def read_results(port):
    s = open_port(port, 115200, reset="pico")
    rows, ready, hz, t0 = {}, False, 0, time.time()
    while time.time() - t0 < 60:
        line = s.readline().decode(errors="replace").strip()
        if "STEM I8 PICO BENCH ready" in line:
            ready, rows, hz = True, {}, int(line.rsplit(", ", 1)[1].split()[0])
        elif ready and line.startswith("R:"):
            name, *us, ok_lib, ok_ring, ok_rows, ones = line[2:].split("|")
            rows[name] = dict(zip([k + "_us" for k in KINDS], map(int, us)), lib_equal=ok_lib == "1",
                              ring_equal=ok_ring == "1", rows_equal=ok_rows == "1", ones=int(ones))
        elif ready and line == "DONE":
            s.close()
            return rows, hz
    s.close()
    sys.exit("[X] no results from the board in 60 s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=["pico2_arm", "pico2_riscv"], required=True)
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--code", choices=["flash", "ram"], default="flash",
                    help="ram: the timed loops in RAM (-DSYMBEX_CODE_IN_RAM, SYMBEX_FAST_CODE), not through the XIP cache")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", f"stem_i8_pico_bench_{a.board}_{a.code}")
    write_project(project, a.port, a.board, a.code)
    flash(project, verbose=False, what=a.board)
    rows, hz = read_results(a.port)
    print(f"\n{a.board}, weights in flash, code in {a.code} (best of 7, ms; clock {hz / 1e6:g} MHz)")
    print(f"  {'shape':<10}{'lib':>8}{'window':>8}{'dots':>8}{'ring':>8}{'rows':>8}{'rows x':>8}  equal (lib/ring/rows)")
    for name, r in rows.items():
        print(f"  {name:<10}" + "".join(f"{r[k + '_us'] / 1e3:>8.2f}" for k in KINDS)
              + f"{r['lib_us'] / r['rows_us']:>7.2f}x  {r['lib_equal']} / {r['ring_equal']} / {r['rows_equal']}")
    if not all(r["lib_equal"] and r["ring_equal"] and r["rows_equal"] for r in rows.values()):
        print("[X] a kernel differs from the plain C reference: the times are not trusted")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"stem_i8_pico_bench_{a.board}_{a.code}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, mhz=hz / 1e6, reps="best of 7", weights="flash", code=a.code, shapes=SHAPES,
                       cases=rows), f,
                  indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
