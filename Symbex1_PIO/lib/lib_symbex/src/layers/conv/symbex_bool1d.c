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

static void conv_pairs(const struct SymbexBool1D* L, const uint32_t* in, uint32_t* out, uint32_t* window) {
    const int n_bits = L->kernel * L->in_ch;
    const int nw = (n_bits + 31) >> 5;
    const uint32_t steps = (uint32_t)L->out_len * L->pool;
    uint32_t* w0 = window;
    uint32_t* w1 = window + L->row_words;
    uint32_t t = 0;
    for (; t + 1 < steps; t += 2) {
        window_bits(in, t * L->stride * L->in_ch, n_bits, w0);
        window_bits(in, (t + 1) * L->stride * L->in_ch, n_bits, w1);
        const uint32_t b0 = (t / L->pool) * L->out_ch, b1 = ((t + 1) / L->pool) * L->out_ch;
        const uint32_t* row = L->weights;
        for (uint16_t o = 0; o < L->out_ch; o++, row += L->row_words) {
            int32_t d0 = 0, d1 = 0;
            for (int j = 0; j < nw; j++) {
                const uint32_t wv = row[j];
                d0 += __builtin_popcount(w0[j] ^ wv);
                d1 += __builtin_popcount(w1[j] ^ wv);
            }
            if (n_bits - 2 * d0 >= L->thresholds[o]) set_bit(out, b0 + o);
            if (n_bits - 2 * d1 >= L->thresholds[o]) set_bit(out, b1 + o);
        }
    }
    if (t < steps) {                                             /* an odd number of steps: the last one alone */
        window_bits(in, t * L->stride * L->in_ch, n_bits, w0);
        const uint32_t b0 = (t / L->pool) * L->out_ch;
        const uint32_t* row = L->weights;
        for (uint16_t o = 0; o < L->out_ch; o++, row += L->row_words) {
            int32_t d0 = 0;
            for (int j = 0; j < nw; j++) d0 += __builtin_popcount(w0[j] ^ row[j]);
            if (n_bits - 2 * d0 >= L->thresholds[o]) set_bit(out, b0 + o);
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
        conv_pairs(L, in, out, window);
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

void symbex_bool1d_conv_u8(const struct SymbexBool1D* L, const uint8_t* in, uint32_t* out, uint8_t* workspace,
                           int32_t* scratch) {
    const int n = L->kernel * L->in_ch;
    clear(out, L);
    const uint32_t steps = (uint32_t)L->out_len * L->pool;
    for (uint32_t t = 0; t < steps; t++) {
        const uint8_t* x = in + t * L->stride * L->in_ch;
        int32_t sum = 0;
        for (int i = 0; i < n; i++) sum += x[i];
        symbex_u8_prepare(x, n, workspace, L->row_words);
        symbex_u8_sum_bits_rows(workspace, n, L->weights, L->row_words, L->out_ch, scratch);
        /* raw = sum of x * (+1 where the weight bit is 1, -1 where it is 0) = 2 * (sum where 1) - sum */
        for (uint16_t o = 0; o < L->out_ch; o++) scratch[o] = 2 * scratch[o] - sum;
        put_bits(L, t, scratch, out);
    }
}
