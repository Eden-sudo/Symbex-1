/*
 * planes1d.h -- har_ultracompact benchmark (README section 10): the 1D layers of the two-bit and ternary networks of
 * README section 9 (1-bit weights, activations in 4 or 3 levels), on lib_symbex's HAL row functions. An experiment:
 * it lives here, not in lib_symbex, until it is proven (acts1d.py embeds this file in the exported model_symbex.h).
 *
 * Maps: two bit planes, each one stream as in symbex_bool1d.h (bit of channel c at step t = stream bit t * ch + c),
 * plane 1 `words` words after plane 0:
 *     2 bits    plane 0 = high bit, plane 1 = low bit of the level L (0..3); the value is 3v = 2L - 3 (-3, -1, 1, 3)
 *     ternary   plane 0 = (L == 2), plane 1 = (L == 0); the value is v = L - 1 (-1, 0, 1)
 * A binary layer on such a map counts each plane with the HAL, the same functions as the 1-bit layers:
 *     2 bits    raw = sum of w * 3v = 2 (n - 2 d_high) + (n - 2 d_low),  d = count(window XOR row)
 *     ternary   raw = sum over +1 of w - sum over -1 of w = 2 a_pos - count(pos) - 2 a_neg + count(neg),
 *               a = count(window AND row)
 * (w = +1 where the row bit is 1, -1 where it is 0). The level of an output is the number of its ascending thresholds
 * raw reaches (3 for 2 bits, 2 for ternary). Pooling takes the largest raw of `pool` steps before the thresholds: the
 * levels are monotonic in raw, so it is the max of the levels (as the network trains).
 * The first layer: uint8 input (int8 + 128), int8 weights (the networks section 9 chose on validation), the same
 * arithmetic as symbex_bool1d's int8 layer (plain C two steps x four rows; ESP32-S3: PIE dot products).
 */
#ifndef HAR_PLANES1D_H
#define HAR_PLANES1D_H

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif
#include <stdint.h>
#include "hal/symbex_hal.h"

struct Planes1D {
    uint16_t in_ch, out_ch, kernel, stride, pool, in_len, out_len;
    uint16_t row_words;                     /* words per weight row (1-bit), rounded up to 4; 0 for the uint8 layer */
    uint16_t in_words, out_words;           /* words per plane of the input / output map */
    uint8_t levels_in;                      /* 4 (2 bits) or 3 (ternary); 0: uint8 input */
    uint8_t n_thr;                          /* thresholds per output channel: 3 (2 bits) or 2 (ternary) */
    const uint32_t* weights;                /* out_ch rows, time-major bits (as symbex_bool1d) */
    const int8_t* weights_i8;               /* uint8 layer: out_ch rows of kernel * in_ch int8 weights */
    const int32_t* thresholds;              /* out_ch x n_thr, ascending; uint8 layer: on the raw of the uint8 input */
    const int32_t* weight_sums;             /* uint8 layer: the sum of each row (ESP32-S3: input taken as x - 128) */
};

static void pl_clear(uint32_t* p, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) p[i] = 0;
}

/* the n_bits stream bits from `bit` -> w, the bits past n_bits cleared (symbex_bool1d.c's window_bits) */
static void pl_window(const uint32_t* stream, uint32_t bit, int n_bits, uint32_t* w) {
    const uint32_t* p = stream + (bit >> 5);
    const int sh = (int)(bit & 31);
    const int nw = (n_bits + 31) >> 5;
    if (sh == 0) {
        for (int j = 0; j < nw; j++) w[j] = p[j];
    } else {
        for (int j = 0; j < nw; j++) w[j] = (p[j] >> sh) | (p[j + 1] << (32 - sh));
    }
    if (n_bits & 31) w[nw - 1] &= (1UL << (n_bits & 31)) - 1UL;
}

/* the levels of pooled step q (acc: its largest raw per channel) into the two output planes; no branches, the
   thresholds unrolled (3 for 2 bits, 2 for ternary) */
static inline void pl_emit(const struct Planes1D* L, uint32_t q, const int32_t* acc, uint32_t* out) {
    uint32_t* out1 = out + L->out_words;
    const int oc = L->out_ch;
    uint32_t i = q * (uint32_t)oc;
    const int32_t* t = L->thresholds;
    if (L->n_thr == 3) {
        for (int o = 0; o < oc; o++, i++, t += 3) {
            const int32_t a = acc[o];
            const uint32_t lv = (uint32_t)(a >= t[0]) + (uint32_t)(a >= t[1]) + (uint32_t)(a >= t[2]);
            const uint32_t w = i >> 5, sh = i & 31;
            out[w] |= (lv >> 1) << sh;
            out1[w] |= (lv & 1u) << sh;
        }
    } else {
        for (int o = 0; o < oc; o++, i++, t += 2) {
            const int32_t a = acc[o];
            const uint32_t w = i >> 5, sh = i & 31;
            out[w] |= (uint32_t)(a >= t[1]) << sh;
            out1[w] |= (uint32_t)(a < t[0]) << sh;
        }
    }
}

/* step t's raw into the pool's running max; at the pool's last step, its levels out (no pool: straight out) */
static inline void pl_step(const struct Planes1D* L, uint32_t t, const int32_t* raw, int32_t* acc, uint32_t* out) {
    if (L->pool == 1) {
        pl_emit(L, t, raw, out);
        return;
    }
    const uint32_t k = t % L->pool;
    if (k == 0) {
        for (int o = 0; o < L->out_ch; o++) acc[o] = raw[o];
    } else {
        for (int o = 0; o < L->out_ch; o++) acc[o] = raw[o] > acc[o] ? raw[o] : acc[o];
    }
    if (k == (uint32_t)L->pool - 1) pl_emit(L, t / L->pool, acc, out);
}

/* a binary layer on a two-plane map. window: 2 * row_words words, 16-byte aligned; raw, tmp, acc: out_ch int32 each */
static void pl_conv(const struct Planes1D* L, const uint32_t* in, uint32_t* out, uint32_t* window, int32_t* raw,
                    int32_t* tmp, int32_t* acc) {
    const int n = L->kernel * L->in_ch, nw = (n + 31) >> 5, rw = L->row_words, oc = L->out_ch;
    uint32_t *w0 = window, *w1 = window + rw;
    const uint32_t* in1 = in + L->in_words;
    for (int j = nw; j < rw; j++) w0[j] = w1[j] = 0;              /* the PIE count reads whole blocks of 4 words */
    pl_clear(out, 2u * L->out_words);
    const uint32_t steps = (uint32_t)L->out_len * L->pool, hop = (uint32_t)L->stride * L->in_ch;
    for (uint32_t t = 0; t < steps; t++) {
        pl_window(in, t * hop, n, w0);
        pl_window(in1, t * hop, n, w1);
        if (L->levels_in == 4) {
            symbex_xor_popcount_rows_n(w0, L->weights, nw, rw, oc, raw);
            symbex_xor_popcount_rows_n(w1, L->weights, nw, rw, oc, tmp);
            for (int o = 0; o < oc; o++) raw[o] = 3 * n - 4 * raw[o] - 2 * tmp[o];
        } else {
            int32_t cp = 0, cn = 0;
            for (int j = 0; j < nw; j++) {
                cp += symbex_hal_popcount32(w0[j]);
                cn += symbex_hal_popcount32(w1[j]);
            }
            symbex_and_popcount_rows_n(w0, L->weights, nw, rw, oc, raw);
            symbex_and_popcount_rows_n(w1, L->weights, nw, rw, oc, tmp);
            for (int o = 0; o < oc; o++) raw[o] = 2 * raw[o] - cp - 2 * tmp[o] + cn;
        }
        pl_step(L, t, raw, acc, out);
    }
}

/* the first layer: uint8 input, int8 weights. work (ESP32-S3 only): 16 * ceil(n / 16) bytes, 16-byte aligned */
static void pl_first(const struct Planes1D* L, const uint8_t* in, uint32_t* out, int8_t* work, int32_t* raw,
                     int32_t* tmp, int32_t* acc) {
    const int n = L->kernel * L->in_ch, oc = L->out_ch;
    const uint32_t steps = (uint32_t)L->out_len * L->pool, hop = (uint32_t)L->stride * L->in_ch;
    pl_clear(out, 2u * L->out_words);
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    const int padded = (n + 15) & ~15;
    for (int i = n; i < padded; i++) work[i] = 0;
    (void)tmp;
    for (uint32_t t = 0; t < steps; t++) {
        const uint8_t* p = in + t * hop;
        for (int i = 0; i < n; i++) work[i] = (int8_t)(p[i] ^ 0x80u);   /* x - 128 */
        for (int o0 = 0; o0 < oc; o0 += 32) {
            const int rows = oc - o0 < 32 ? oc - o0 : 32;
            symbex_s8_dot_rows(work, n, L->weights_i8 + o0 * n, n, rows, raw + o0);
        }
        for (int o = 0; o < oc; o++) raw[o] += 128 * L->weight_sums[o];   /* back to the raw of the uint8 input */
        pl_step(L, t, raw, acc, out);
    }
#else
    (void)work;
    const int groups = (oc & 3) == 0 ? oc >> 2 : 0;
    for (uint32_t t = 0; t < steps; t += 2) {
        const uint32_t t1 = (t + 1 < steps) ? t + 1 : t;
        const uint8_t *x0 = in + t * hop, *x1 = in + t1 * hop;
        const int8_t* r0 = L->weights_i8;
        int o = 0;
        for (int g = 0; g < groups; g++, o += 4, r0 += 4 * n) {
            const int8_t *r1 = r0 + n, *r2 = r1 + n, *r3 = r2 + n;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int i = 0; i < n; i++) {
                const int32_t u0 = x0[i], u1 = x1[i];
                const int32_t v0 = r0[i], v1 = r1[i], v2 = r2[i], v3 = r3[i];
                a0 += u0 * v0; a1 += u0 * v1; a2 += u0 * v2; a3 += u0 * v3;
                c0 += u1 * v0; c1 += u1 * v1; c2 += u1 * v2; c3 += u1 * v3;
            }
            raw[o] = a0; raw[o + 1] = a1; raw[o + 2] = a2; raw[o + 3] = a3;
            tmp[o] = c0; tmp[o + 1] = c1; tmp[o + 2] = c2; tmp[o + 3] = c3;
        }
        for (; o < oc; o++, r0 += n) {
            int32_t a0 = 0, c0 = 0;
            for (int i = 0; i < n; i++) {
                a0 += x0[i] * r0[i];
                c0 += x1[i] * r0[i];
            }
            raw[o] = a0;
            tmp[o] = c0;
        }
        pl_step(L, t, raw, acc, out);
        if (t1 != t) pl_step(L, t1, tmp, acc, out);
    }
#endif
}

/* the classifier on the final two-plane map (each plane n_bits bits, zero padded to row_words words): the class with
   the largest scale * raw + bias (int64, first largest), raw as in pl_conv. a, b: n_classes int32 */
static int pl_classify(const uint32_t* map, uint16_t plane_words, int levels, uint16_t n_words, int32_t n_bits,
                       const uint32_t* weights, uint16_t row_words, uint16_t n_classes, const int32_t* scale,
                       const int32_t* bias, int32_t* a, int32_t* b) {
    const uint32_t* map1 = map + plane_words;
    int32_t cp = 0, cn = 0;
    if (levels == 4) {
        symbex_xor_popcount_rows_n(map, weights, n_words, row_words, n_classes, a);
        symbex_xor_popcount_rows_n(map1, weights, n_words, row_words, n_classes, b);
    } else {
        for (int j = 0; j < n_words; j++) {
            cp += symbex_hal_popcount32(map[j]);
            cn += symbex_hal_popcount32(map1[j]);
        }
        symbex_and_popcount_rows_n(map, weights, n_words, row_words, n_classes, a);
        symbex_and_popcount_rows_n(map1, weights, n_words, row_words, n_classes, b);
    }
    int best = 0;
    int64_t best_logit = 0;
    for (int o = 0; o < n_classes; o++) {
        const int32_t raw = levels == 4 ? 3 * n_bits - 4 * a[o] - 2 * b[o] : 2 * a[o] - cp - 2 * b[o] + cn;
        const int64_t logit = (int64_t)scale[o] * raw + bias[o];
        if (o == 0 || logit > best_logit) { best_logit = logit; best = o; }
    }
    return best;
}

#endif
