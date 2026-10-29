/*
 * symbex_bool1d.c -- binary 1D convolutions (see symbex_bool1d.h). Integers only, no multiplication: the windows are
 * cut from the bit stream with shifts, the bits are counted by the HAL row functions.
 */
#include "layers/conv/symbex_bool1d.h"
#include "layers/conv/symbex_bool.h"
#include "hal/symbex_hal.h"

/* Where the compiler counts bits with one instruction (x86 POPCNT, AArch64, RISC-V Zbb, or forced with
   -DSYMBEX_POPCOUNT_FORCE_NATIVE), the XOR form runs inline, two time steps per pass over the weights: each weight word
   is read once for two windows, the two sums stay in registers and meet their thresholds at once. Elsewhere (the
   ESP32 table, the ESP32-S3 PIE) the HAL row functions are faster, and they are used. */
#if defined(SYMBEX_POPCOUNT_FORCE_NATIVE) || defined(__POPCNT__) || defined(__aarch64__) || defined(__riscv_zbb)
#define BOOL1D_INLINE 1
#endif

/* the n_bits stream bits starting at `bit` -> w (least significant first), the bits past n_bits cleared */
static void window_bits(const uint32_t* stream, uint32_t bit, int n_bits, uint32_t* w) {
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

static void clear(uint32_t* out, const struct SymbexBool1D* L) {
    const uint32_t n = ((uint32_t)L->out_len * L->out_ch + 31) / 32 + 1;
    for (uint32_t i = 0; i < n; i++) out[i] = 0;
}

/* the output bits of one conv step, ORed into its pooled step */
static inline void put_bits(const struct SymbexBool1D* L, uint32_t step, const int32_t* raw, uint32_t* out) {
    const uint32_t base = (step / L->pool) * L->out_ch;
    for (uint16_t o = 0; o < L->out_ch; o++) {
        if (raw[o] >= L->thresholds[o]) out[(base + o) >> 5] |= 1UL << ((base + o) & 31);
    }
}

#if defined(BOOL1D_INLINE)
static inline void set_bit(uint32_t* out, uint32_t i) { out[i >> 5] |= 1UL << (i & 31); }

/* n - 2d >= thr  <=>  d <= (n - thr) / 2. raw = n - 2d is always in [-n, n], so the threshold is clamped to [-n, n + 1]
   first (INT_MIN = always on, INT_MAX = always off, as the exporter writes them): n - thr cannot overflow. */
static inline int32_t limit_of(int n_bits, int32_t thr) {
    const int32_t t = thr < -n_bits ? -n_bits : (thr > n_bits + 1 ? n_bits + 1 : thr);
    return (n_bits - t) >> 1;                                     /* floor: arithmetic shift */
}

/* Two time steps x four output channels per pass: every window word is read once for four rows, every weight word once
   for two windows (6 loads for 8 counts). The threshold becomes a limit on the disagreements, one compare per output:
   n - 2 d >= threshold  <=>  d <= floor((n - threshold) / 2). The four bits of a group go out in one OR (out_ch a
   multiple of 4: a group never crosses a word). Windows that start on a word with whole words (stride * in_ch and
   kernel * in_ch multiples of 32) are read in place, not copied. */
static void conv_pairs(const struct SymbexBool1D* L, const uint32_t* in, uint32_t* out, uint32_t* window,
                       int32_t* lim) {
    const int n_bits = L->kernel * L->in_ch;
    const int nw = (n_bits + 31) >> 5;
    const uint32_t steps = (uint32_t)L->out_len * L->pool;
    const uint32_t hop = (uint32_t)L->stride * L->in_ch;
    const int in_place = (hop & 31) == 0 && (n_bits & 31) == 0;
    const int rw = L->row_words;
    const int groups = (L->out_ch & 3) == 0 ? L->out_ch >> 2 : 0;
    for (uint16_t o = 0; o < L->out_ch; o++) lim[o] = limit_of(n_bits, L->thresholds[o]);
    for (uint32_t t = 0; t < steps; t += 2) {
        const uint32_t t1 = (t + 1 < steps) ? t + 1 : t;          /* an odd number of steps: the last one twice */
        const uint32_t *w0, *w1;
        if (in_place) {
            w0 = in + ((t * hop) >> 5);
            w1 = in + ((t1 * hop) >> 5);
        } else {
            window_bits(in, t * hop, n_bits, window);
            window_bits(in, t1 * hop, n_bits, window + rw);
            w0 = window;
            w1 = window + rw;
        }
        const uint32_t b0 = (t / L->pool) * L->out_ch, b1 = (t1 / L->pool) * L->out_ch;
        const uint32_t* row = L->weights;
        int o = 0;
        for (int g = 0; g < groups; g++, o += 4, row += 4 * rw) {
            const uint32_t *r1 = row + rw, *r2 = r1 + rw, *r3 = r2 + rw;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int j = 0; j < nw; j++) {
                const uint32_t x0 = w0[j], x1 = w1[j];
                const uint32_t v0 = row[j], v1 = r1[j], v2 = r2[j], v3 = r3[j];
                a0 += __builtin_popcount(x0 ^ v0);
                a1 += __builtin_popcount(x0 ^ v1);
                a2 += __builtin_popcount(x0 ^ v2);
                a3 += __builtin_popcount(x0 ^ v3);
                c0 += __builtin_popcount(x1 ^ v0);
                c1 += __builtin_popcount(x1 ^ v1);
                c2 += __builtin_popcount(x1 ^ v2);
                c3 += __builtin_popcount(x1 ^ v3);
            }
            const int32_t l0 = lim[o], l1 = lim[o + 1], l2 = lim[o + 2], l3 = lim[o + 3];
            const uint32_t m0 = (uint32_t)(a0 <= l0) | (uint32_t)(a1 <= l1) << 1 | (uint32_t)(a2 <= l2) << 2 |
                                (uint32_t)(a3 <= l3) << 3;
            const uint32_t m1 = (uint32_t)(c0 <= l0) | (uint32_t)(c1 <= l1) << 1 | (uint32_t)(c2 <= l2) << 2 |
                                (uint32_t)(c3 <= l3) << 3;
            out[(b0 + o) >> 5] |= m0 << ((b0 + o) & 31);
            out[(b1 + o) >> 5] |= m1 << ((b1 + o) & 31);
        }
        for (; o < L->out_ch; o++, row += rw) {                  /* out_ch not a multiple of 4: one row at a time */
            int32_t d0 = 0, d1 = 0;
            for (int j = 0; j < nw; j++) {
                const uint32_t wv = row[j];
                d0 += __builtin_popcount(w0[j] ^ wv);
                d1 += __builtin_popcount(w1[j] ^ wv);
            }
            if (d0 <= lim[o]) set_bit(out, b0 + o);
            if (d1 <= lim[o]) set_bit(out, b1 + o);
        }
    }
}

/* Windows of one word (kernel * in_ch <= 32): the cost per step is all fixed cost, so it is cut to the bone. Groups of
   four output channels run over the whole layer: their four weight words and four limits stay in registers, a step is
   the window (two loads and a shift), four counts and four compares; the pool ORs the bits in a register and each
   pooled step's four bits go out once (they may cross into the next word). A last group of 1-3 channels gets limits
   of -1 (never set) for the missing ones. */
static void conv_word(const struct SymbexBool1D* L, const uint32_t* in, uint32_t* out, int32_t* lim) {
    const int n_bits = L->kernel * L->in_ch;
    const uint32_t mask = n_bits == 32 ? 0xFFFFFFFFUL : (1UL << n_bits) - 1UL;
    const uint32_t hop = (uint32_t)L->stride * L->in_ch;
    const int rw = L->row_words, oc = L->out_ch;
    for (int o = 0; o < oc; o++) lim[o] = limit_of(n_bits, L->thresholds[o]);
    for (int g = 0; g < oc; g += 4) {
        const int last = oc - 1;
        const uint32_t v0 = L->weights[g * rw], v1 = L->weights[(g + 1 <= last ? g + 1 : g) * rw];
        const uint32_t v2 = L->weights[(g + 2 <= last ? g + 2 : g) * rw], v3 = L->weights[(g + 3 <= last ? g + 3 : g) * rw];
        const int32_t l0 = lim[g], l1 = g + 1 <= last ? lim[g + 1] : -1;
        const int32_t l2 = g + 2 <= last ? lim[g + 2] : -1, l3 = g + 3 <= last ? lim[g + 3] : -1;
        uint32_t bit = 0, base = (uint32_t)g;
        for (uint16_t q = 0; q < L->out_len; q++, base += oc) {
            uint32_t acc = 0;
            for (uint16_t k = 0; k < L->pool; k++, bit += hop) {
                const uint32_t* p = in + (bit >> 5);
                const uint32_t sh = bit & 31;
                const uint32_t w = (sh ? (p[0] >> sh) | (p[1] << (32 - sh)) : p[0]) & mask;
                acc |= (uint32_t)(__builtin_popcount(w ^ v0) <= l0) | (uint32_t)(__builtin_popcount(w ^ v1) <= l1) << 1 |
                       (uint32_t)(__builtin_popcount(w ^ v2) <= l2) << 2 | (uint32_t)(__builtin_popcount(w ^ v3) <= l3) << 3;
            }
            const uint32_t sh = base & 31;
            out[base >> 5] |= acc << sh;
            if (sh > 28) out[(base >> 5) + 1] |= acc >> (32 - sh);
        }
    }
}
#endif

void symbex_bool1d_conv(const struct SymbexBool1D* L, const uint32_t* in, uint32_t* out, uint32_t* window,
                        int32_t* scratch) {
    const int n_bits = L->kernel * L->in_ch;
    const int nw = (n_bits + 31) >> 5;
#if defined(BOOL1D_INLINE)
    if (symbex_bool_force_mode != 1) {                           /* the AND form can still be forced (tests) */
        clear(out, L);
        if (nw == 1)
            conv_word(L, in, out, scratch);
        else
            conv_pairs(L, in, out, window, scratch);
        return;
    }
#endif
    const int xor_form = symbex_bool_force_mode == 2 || (symbex_bool_force_mode == 0 && symbex_hal_prefers_xor());
    for (int j = nw; j < L->row_words; j++) window[j] = 0;     /* the PIE count reads whole blocks of 4 words */
    clear(out, L);
    const uint32_t steps = (uint32_t)L->out_len * L->pool;      /* the conv outputs the pooled map uses */
    for (uint32_t t = 0; t < steps; t++) {
        window_bits(in, t * L->stride * L->in_ch, n_bits, window);
        if (xor_form) {
            /* bits past n_bits are 0 in the window and in the rows: the count is the disagreements directly */
            symbex_xor_popcount_rows_n(window, L->weights, nw, L->row_words, L->out_ch, scratch);
            for (uint16_t o = 0; o < L->out_ch; o++) scratch[o] = n_bits - 2 * scratch[o];
        } else {
            int32_t ones = 0;
            for (int j = 0; j < nw; j++) ones += symbex_hal_popcount32(window[j]);
            symbex_and_popcount_rows_n(window, L->weights, nw, L->row_words, L->out_ch, scratch);
            /* disagreements = ones + weight ones - 2 * (both ones); raw = n - 2 * disagreements */
            for (uint16_t o = 0; o < L->out_ch; o++)
                scratch[o] = n_bits - 2 * (ones + L->weight_counts[o] - 2 * scratch[o]);
        }
        put_bits(L, t, scratch, out);
    }
}

/* -DSYMBEX_BOOL1D_U8_EXTERNAL: the application gives its own uint8 first layer (same contract), e.g. a chip's byte
   dot product: benchmarks/conv1d/har_ultracompact/pulp/symbex_u8_pulp.c */
#if !defined(SYMBEX_BOOL1D_U8_EXTERNAL)
/* int8 weights (weights_i8): a plain integer convolution, two time steps x four output channels per pass (each input
   byte serves 4 rows, each weight byte 2 steps). The compiler maps the multiply-accumulate to what the chip has. */
static void conv_u8_i8(const struct SymbexBool1D* L, const uint8_t* in, uint32_t* out) {
    const int n = L->kernel * L->in_ch;
    const uint32_t steps = (uint32_t)L->out_len * L->pool;
    const uint32_t hop = (uint32_t)L->stride * L->in_ch;
    const int groups = (L->out_ch & 3) == 0 ? L->out_ch >> 2 : 0;
    for (uint32_t t = 0; t < steps; t += 2) {
        const uint32_t t1 = (t + 1 < steps) ? t + 1 : t;          /* an odd number of steps: the last one twice */
        const uint8_t *x0 = in + t * hop, *x1 = in + t1 * hop;
        const uint32_t b0 = (t / L->pool) * L->out_ch, b1 = (t1 / L->pool) * L->out_ch;
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
            const int32_t* th = L->thresholds + o;
            const uint32_t m0 = (uint32_t)(a0 >= th[0]) | (uint32_t)(a1 >= th[1]) << 1 | (uint32_t)(a2 >= th[2]) << 2 |
                                (uint32_t)(a3 >= th[3]) << 3;
            const uint32_t m1 = (uint32_t)(c0 >= th[0]) | (uint32_t)(c1 >= th[1]) << 1 | (uint32_t)(c2 >= th[2]) << 2 |
                                (uint32_t)(c3 >= th[3]) << 3;
            out[(b0 + o) >> 5] |= m0 << ((b0 + o) & 31);
            out[(b1 + o) >> 5] |= m1 << ((b1 + o) & 31);
        }
        for (; o < L->out_ch; o++, r0 += n) {                    /* out_ch not a multiple of 4: one row at a time */
            int32_t a0 = 0, c0 = 0;
            for (int i = 0; i < n; i++) {
                a0 += x0[i] * r0[i];
                c0 += x1[i] * r0[i];
            }
            if (a0 >= L->thresholds[o]) out[(b0 + o) >> 5] |= 1UL << ((b0 + o) & 31);
            if (c0 >= L->thresholds[o]) out[(b1 + o) >> 5] |= 1UL << ((b1 + o) & 31);
        }
    }
}

#if defined(BOOL1D_INLINE)
/* 1-bit weights where the bits are counted with one instruction: the uint8 input is 8 binary streams, its bit planes
   (plane q = bit q of every input, time-major like every map), built ONCE for the whole input. Each step only cuts
   its window out of the 8 planes, as the binary layers do; then, for every output,
       raw = sum of x * (+1 / -1) = 2 * sum_q 2^q count(P_q AND w) - sum_q 2^q count(P_q)
   8 counts per window word, no copy and no sum of the bytes per step. workspace: SYMBEX_BOOL1D_U8_WORK_BYTES. */
static void planes_of(const uint8_t* in, int n_in, uint32_t* planes, int pw) {
    for (int j = 0; j < pw; j++) {
        uint32_t pl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int g = 0; g < 8; g++) {                            /* 4 inputs at a time, one per byte of v */
            const int i = j * 32 + g * 4;
            uint32_t v = 0;
            for (int k = 0; k < 4; k++) v |= (uint32_t)(i + k < n_in ? in[i + k] : 0) << (8 * k);
            for (int q = 0; q < 8; q++) {
                /* bit q of the 4 bytes (bits 0, 8, 16, 24) -> 4 adjacent bits: the products land on distinct bits
                   (no carries), the 4 wanted ones on bits 28-31 */
                const uint32_t b = (v >> q) & 0x01010101u;
                pl[q] |= ((uint32_t)(b * 0x10204080u) >> 28) << (4 * g);   /* 32-bit product (UL is 64 on a PC) */
            }
        }
        for (int q = 0; q < 8; q++) planes[q * pw + j] = pl[q];
    }
}

static void conv_u8_planes(const struct SymbexBool1D* L, const uint8_t* in, uint32_t* out, uint32_t* work) {
    const int n = L->kernel * L->in_ch;
    const int nw = (n + 31) >> 5;
    const int pw = (L->in_len * L->in_ch + 31) / 32 + 1;          /* words per plane (+1: a window may read past) */
    uint32_t* planes = work;
    uint32_t* win = work + 8 * pw;                                  /* the step's 8 windows, nw words each */
    planes_of(in, L->in_len * L->in_ch, planes, pw);
    const uint32_t steps = (uint32_t)L->out_len * L->pool;
    const uint32_t hop = (uint32_t)L->stride * L->in_ch;
    for (uint32_t t = 0; t < steps; t++) {
        int32_t sum = 0;
        for (int q = 0; q < 8; q++) {
            window_bits(planes + q * pw, t * hop, n, win + q * nw);
            int32_t c = 0;
            for (int j = 0; j < nw; j++) c += __builtin_popcount(win[q * nw + j]);
            sum += c << q;
        }
        const uint32_t base = (t / L->pool) * L->out_ch;
        const uint32_t* row = L->weights;
        for (uint16_t o = 0; o < L->out_ch; o++, row += L->row_words) {
            int32_t a = 0;
            for (int q = 0; q < 8; q++) {
                const uint32_t* wq = win + q * nw;
                int32_t c = 0;
                for (int j = 0; j < nw; j++) c += __builtin_popcount(wq[j] & row[j]);
                a += c << q;
            }
            if (2 * a - sum >= L->thresholds[o]) out[(base + o) >> 5] |= 1UL << ((base + o) & 31);
        }
    }
}
#endif

void symbex_bool1d_conv_u8(const struct SymbexBool1D* L, const uint8_t* in, uint32_t* out, uint8_t* workspace,
                           int32_t* scratch) {
    clear(out, L);
    if (L->weights_i8) {
        conv_u8_i8(L, in, out);
        return;
    }
#if defined(BOOL1D_INLINE)
    conv_u8_planes(L, in, out, (uint32_t*)workspace);
    (void)scratch;
#else
    const int n = L->kernel * L->in_ch;
    /* the HAL row functions read only the real inputs, except the ESP32-S3's PIE, which reads whole padded rows */
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    const int words = L->row_words;
#else
    const int words = (n + 31) >> 5;
#endif
    const uint32_t steps = (uint32_t)L->out_len * L->pool;
    const uint32_t hop = (uint32_t)L->stride * L->in_ch;
    int32_t sum = 0;
    for (uint32_t t = 0; t < steps; t++) {
        const uint8_t* x = in + t * hop;
        if (t == 0 || hop > (uint32_t)n) {
            sum = 0;
            for (int i = 0; i < n; i++) sum += x[i];
        } else {                                                     /* running sum: hop bytes leave, hop enter */
            for (uint32_t i = 0; i < hop; i++) sum += x[n - hop + i] - x[(int)i - (int)hop];
        }
        symbex_u8_prepare(x, n, workspace, words);
        symbex_u8_sum_bits_rows(workspace, n, L->weights, L->row_words, L->out_ch, scratch);
        /* raw = sum of x * (+1 where the weight bit is 1, -1 where it is 0) = 2 * (sum where 1) - sum */
        for (uint16_t o = 0; o < L->out_ch; o++) scratch[o] = 2 * scratch[o] - sum;
        put_bits(L, t, scratch, out);
    }
#endif
}
#endif
