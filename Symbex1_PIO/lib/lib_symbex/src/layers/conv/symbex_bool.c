/*
 * symbex_bool.c -- fully binary convolutional layers (see include/layers/conv/symbex_bool.h).
 * The counts come from the HAL's whole-layer primitives, in the form it measured as faster (symbex_hal_prefers_xor):
 * symbex_xor_popcount_rows (pie_swar on the ESP32-S3; popcount instruction or SWAR elsewhere) or
 * symbex_and_popcount_rows (the 64 KB table, ESP32) -- the same tested kernels every network uses.
 */
#include <string.h>
#include "layers/conv/symbex_bool.h"
#include "hal/symbex_hal.h"

#ifdef SYMBEX_FINE_PROFILE
#if defined(ESP_PLATFORM)
#include "esp_cpu.h"
static inline uint32_t pf_now(void) { return (uint32_t)esp_cpu_get_cycle_count(); }
#else
static inline uint32_t pf_now(void) { return 0; }       /* no cycle counter here: the profile is for the boards */
#endif
const char* const symbex_pf_names[SYMBEX_PF_N] = {
    "thermometer (input encoding)", "conv: window copy (interior)", "conv: window copy (borders)",
    "conv: popcount of the window (AND form)", "conv: count, whole layer (HAL)", "conv: thresholds + output bits",
    "OR pool 2x2", "OR shortcut", "classifier", "int8 stem (whole)"};
uint64_t symbex_pf_cycles[SYMBEX_PF_N];
uint32_t symbex_pf_calls[SYMBEX_PF_N];
uint32_t symbex_pf_counter_cost;
#define PF_T(v) const uint32_t v = pf_now()
#define PF_ADD(k, t0) do { symbex_pf_cycles[k] += (uint32_t)(pf_now() - (t0)); symbex_pf_calls[k]++; } while (0)
void symbex_pf_reset(void) {
    for (int k = 0; k < SYMBEX_PF_N; k++) { symbex_pf_cycles[k] = 0; symbex_pf_calls[k] = 0; }
    for (int i = 0; i < 1000; i++) { PF_T(t0); PF_ADD(0, t0); }       /* what an empty measurement carries */
    symbex_pf_counter_cost = (uint32_t)(symbex_pf_cycles[0] / 1000);
    symbex_pf_cycles[0] = 0; symbex_pf_calls[0] = 0;
}
#else
#define PF_T(v) do { } while (0)
#define PF_ADD(k, t0) do { } while (0)
#endif

int symbex_bool_force_mode = 0;

/* appends the `nbits` bits of a map pixel (src, bits above nbits are 0) to the window at bit `off` (window zeroed);
   win_words = words of the window: nothing is written past it */
static inline void put_bits(uint32_t* win, const uint32_t* src, int nbits, int off, int win_words) {
    for (int j = 0; nbits > 0; j++, nbits -= 32, off += 32) {
        const uint32_t v = src[j];
        const int w = off >> 5, sh = off & 31;
        win[w] |= v << sh;
        if (sh && w + 1 < win_words) win[w + 1] |= v >> (32 - sh);
    }
}

void symbex_bool_conv(const struct SymbexBoolConv* L, const uint32_t* in, uint32_t* out,
                      uint32_t* window, int32_t* scratch) {
    const int xor_form = symbex_bool_force_mode ? symbex_bool_force_mode == 2 : symbex_hal_prefers_xor();
    const int pin = (L->in_ch + 31) / 32, pout = (L->out_ch + 31) / 32, n_out = L->out_ch;
    const int k = L->kernel, pad = k / 2, kk = k * k, row_run = k * pin;
    const int cin = L->in_ch, dense = (cin & 31) != 0;   /* channels not a multiple of 32: the window is packed */
    const int win_words = (kk * cin + 31) / 32;               /* words of the window (= kk * pin when not packed) */
    const int H = L->in_h, W = L->in_w, s = L->stride, st = L->row_words;
    if (kk > SYMBEX_BOOL_MAX_KK) return;        /* kernels up to 3 x 3 (invalid[] below); the exporters never exceed it */
    const int32_t* thr = L->thresholds;
    const int16_t* wc = L->weight_counts;
    int32_t* adj = scratch;                      /* AND form: thresholds + 2 * popcount of all the weights of output o */
    int32_t* cnt = scratch + n_out;              /* the counts of the current position, one per output */
    if (!xor_form) {
        for (int o = 0; o < n_out; o++) {
            int32_t tot = 0;
            for (int q = 0; q < kk; q++) tot += wc[o * kk + q];
            adj[o] = thr[o] + 2 * tot;
        }
    }
    for (int i = win_words; i < st; i++) window[i] = 0;   /* row padding, once: the copies below never write there */
    int invalid[SYMBEX_BOOL_MAX_KK];
    for (int oy = 0; oy < L->out_h; oy++) {
        for (int ox = 0; ox < L->out_w; ox++) {
            const int y0 = oy * s - pad, x0 = ox * s - pad;
            int nv = 0, ni = 0;
            PF_T(t_win);
            const int interior = y0 >= 0 && x0 >= 0 && y0 + k <= H && x0 + k <= W;
            if (dense && interior && pin == 1) {
                /* packed, interior, every pixel in one word (16 or 24 channels...): the bits pile up in a 64-bit
                   accumulator and every window word is written once -- no border checks, no read-modify-write */
                uint64_t acc = 0;
                int nacc = 0, o = 0;
                for (int kh = 0; kh < k; kh++) {
                    const uint32_t* src = &in[(y0 + kh) * W + x0];
                    for (int kw = 0; kw < k; kw++) {
                        acc |= (uint64_t)src[kw] << nacc;
                        nacc += cin;
                        if (nacc >= 32) {
                            window[o++] = (uint32_t)acc;
                            acc >>= 32;
                            nacc -= 32;
                        }
                    }
                }
                if (nacc > 0) window[o] = (uint32_t)acc;
                nv = kk;
            } else if (dense) {
                /* packed: the window is the bits of its k*k pixels one after the other (pixel q at bit q * cin) */
                for (int i = 0; i < win_words; i++) window[i] = 0;
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        const int y = y0 + kh, x = x0 + kw, q = kh * k + kw;
                        if (y >= 0 && y < H && x >= 0 && x < W) {
                            put_bits(window, &in[(y * W + x) * pin], cin, q * cin, win_words);
                            nv++;
                        } else {
                            invalid[ni++] = q;
                        }
                    }
                }
            } else if (interior) {
                /* the k rows of the window are k runs of `row_run` words of the map: a word loop (memcpy's fixed cost
                   was most of the time for copies this small) */
                for (int kh = 0; kh < k; kh++) {
                    const uint32_t* src = &in[((y0 + kh) * W + x0) * pin];
                    uint32_t* dst = &window[kh * row_run];
                    for (int i = 0; i < row_run; i++) dst[i] = src[i];
                }
                nv = kk;
            } else {
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        const int y = y0 + kh, x = x0 + kw;
                        uint32_t* dst = &window[(kh * k + kw) * pin];
                        if (y >= 0 && y < H && x >= 0 && x < W) {
                            const uint32_t* src = &in[(y * W + x) * pin];
                            for (int j = 0; j < pin; j++) dst[j] = src[j];
                            nv++;
                        } else {
                            for (int j = 0; j < pin; j++) dst[j] = 0;
                            invalid[ni++] = kh * k + kw;
                        }
                    }
                }
            }
            PF_ADD(interior ? SYMBEX_PF_WINDOW : SYMBEX_PF_WINDOW_BORDER, t_win);
            (void)interior;
            const int32_t nbits = nv * L->in_ch;
            uint32_t* o_px = &out[(oy * L->out_w + ox) * pout];
            if (xor_form) {
                /* differences = count(window XOR weights); outside the image the window is 0, so there it counted the
                   weight bits themselves: taken out with the per-pixel weight counts. bit = nbits - 2 d >= thr */
                PF_T(t_cnt);
                symbex_xor_popcount_rows_n(window, L->weights, win_words, st, n_out, cnt);
                PF_ADD(SYMBEX_PF_COUNT, t_cnt);
                PF_T(t_thr);
                for (int ow = 0, o = 0; ow < pout; ow++) {
                    uint32_t bits = 0;                  /* the output word, built in a register, written once */
                    const int end = (ow + 1) * 32 < n_out ? (ow + 1) * 32 : n_out;
                    if (ni == 0) {
                        for (int b = 0; o < end; o++, b++) bits |= (uint32_t)(nbits - 2 * cnt[o] >= thr[o]) << b;
                    } else {
                        for (int b = 0; o < end; o++, b++) {
                            int32_t d = cnt[o];
                            for (int q = 0; q < ni; q++) d -= wc[o * kk + invalid[q]];
                            bits |= (uint32_t)(nbits - 2 * d >= thr[o]) << b;
                        }
                    }
                    o_px[ow] = bits;
                }
                PF_ADD(SYMBEX_PF_THRESHOLD, t_thr);
            } else {
                /* differences = popcount(window) + popcount(weights in the image) - 2 * count(window AND weights);
                   bit = nbits - 2 d >= thr  <=>  4 * count >= thr + 2 * popcount(all weights) + 2 * popcount(window)
                   - nbits + 2 * (weight bits outside the image) */
                PF_T(t_pcx);
                int32_t pcx;
                symbex_and_popcount_rows_n(window, window, win_words, st, 1, &pcx);    /* popcount(window): one call */
                PF_ADD(SYMBEX_PF_WINDOW_POPCOUNT, t_pcx);
                PF_T(t_cnt);
                symbex_and_popcount_rows_n(window, L->weights, win_words, st, n_out, cnt);
                PF_ADD(SYMBEX_PF_COUNT, t_cnt);
                PF_T(t_thr);
                const int32_t c0 = 2 * pcx - nbits;
                for (int ow = 0, o = 0; ow < pout; ow++) {
                    uint32_t bits = 0;
                    const int end = (ow + 1) * 32 < n_out ? (ow + 1) * 32 : n_out;
                    if (ni == 0) {
                        for (int b = 0; o < end; o++, b++) bits |= (uint32_t)(4 * cnt[o] >= adj[o] + c0) << b;
                    } else {
                        for (int b = 0; o < end; o++, b++) {
                            int32_t outside = 0;
                            for (int q = 0; q < ni; q++) outside += wc[o * kk + invalid[q]];
                            bits |= (uint32_t)(4 * cnt[o] >= adj[o] + c0 - 2 * outside) << b;
                        }
                    }
                    o_px[ow] = bits;
                }
                PF_ADD(SYMBEX_PF_THRESHOLD, t_thr);
            }
        }
    }
}

void symbex_thermometer(const uint8_t* img, uint16_t h, uint16_t w, uint8_t levels, uint32_t* out) {
    PF_T(t_all);
    const int ppx = (3 * levels + 31) / 32;
    /* a 256-entry table: pixel value -> its `levels` thermometer bits, built once per call (256 x levels steps); then
       every pixel is 3 lookups and a few ORs instead of 3 x levels comparisons (9x fewer cycles at 96 x 96, measured
       by the fine profile) */
    uint32_t lut[256];
    {
        int k = 0;
        uint32_t bits = 0;
        for (int v = 0; v < 256; v++) {
            while (k < levels && v >= (k + 1) * 256 / (levels + 1)) bits |= 1UL << k++;
            lut[v] = bits;
        }
    }
    const uint32_t n = (uint32_t)h * w;
    if (3 * levels <= 32) {                      /* the three colors fit in one word: one write per pixel */
        const int l2 = 2 * levels;
        for (uint32_t p = 0; p < n; p++) {
            const uint8_t* px = &img[p * 3];
            out[p] = lut[px[0]] | (lut[px[1]] << levels) | (l2 < 32 ? lut[px[2]] << l2 : 0);
        }
    } else {                                     /* more levels: the bits of each color at bit c * levels */
        for (uint32_t p = 0; p < n; p++) {
            uint32_t* o = &out[p * ppx];
            for (int j = 0; j < ppx; j++) o[j] = 0;
            for (int c = 0; c < 3; c++) {
                const uint64_t v = (uint64_t)lut[img[p * 3 + c]] << ((c * levels) & 31);
                const int wd = (c * levels) >> 5;
                o[wd] |= (uint32_t)v;
                if (wd + 1 < ppx) o[wd + 1] |= (uint32_t)(v >> 32);
            }
        }
    }
    PF_ADD(SYMBEX_PF_THERMOMETER, t_all);
}

/* the int8 stem's dot products: out[r] = sum_{i < n} x[i] * w[r * stride + i]; x and the rows zero padded to
   stride bytes (a multiple of 4), x 16-byte aligned and SYMBEX_BOOL_STEM_I8_WINDOW bytes */
#if defined(__ARM_FEATURE_SIMD32) || defined(SYMBEX_HAL_TEST_SMLAD)
#if defined(__ARM_FEATURE_SIMD32)
#include <arm_acle.h>
#define STEM_SXTB16(v) __sxtb16(v)
#define STEM_ROR8(v) __ror((v), 8)
#define STEM_SMLAD(a, b, acc) __smlad((a), (b), (acc))
#else               /* the tests: the same steps in plain C (SXTB16, ROR, SMLAD as the Arm manual defines them) */
static inline uint32_t STEM_SXTB16(uint32_t v) {
    return (uint32_t)(uint16_t)(int16_t)(int8_t)(v & 0xFFu) | ((uint32_t)(uint16_t)(int16_t)(int8_t)(v >> 16) << 16);
}
static inline uint32_t STEM_ROR8(uint32_t v) { return (v >> 8) | (v << 24); }
static inline int32_t STEM_SMLAD(uint32_t a, uint32_t b, int32_t acc) {
    return acc + (int32_t)(int16_t)(a & 0xFFFFu) * (int16_t)(b & 0xFFFFu)
               + (int32_t)(int16_t)(a >> 16) * (int16_t)(b >> 16);
}
#endif
#endif

static void stem_i8_dot_rows(const int8_t* x, int n, const int8_t* w, int stride, int n_rows, int32_t* out) {
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    (void)stride;
    symbex_s8_dot_rows(x, n, w, stride, n_rows, out);        /* PIE: 16 products per instruction */
#elif defined(__ARM_FEATURE_SIMD32) || defined(SYMBEX_HAL_TEST_SMLAD)
    /* SMLAD: two 16 x 16 products per instruction; SXTB16 splits 4 bytes into the even and the odd pair. The window's
       pairs are split once, then every row costs 2 SXTB16 + 2 SMLAD per 4 bytes. */
    (void)n;
    uint32_t xe[SYMBEX_BOOL_STEM_I8_WINDOW / 4], xo[SYMBEX_BOOL_STEM_I8_WINDOW / 4];
    const int nw = stride >> 2;
    const uint32_t* xw = (const uint32_t*)(const void*)x;
    for (int i = 0; i < nw; i++) { xe[i] = STEM_SXTB16(xw[i]); xo[i] = STEM_SXTB16(STEM_ROR8(xw[i])); }
    for (int r = 0; r < n_rows; r++) {
        const uint32_t* ww = (const uint32_t*)(const void*)(w + r * stride);
        int32_t a = 0, b = 0;
        for (int i = 0; i < nw; i++) {
            const uint32_t v = ww[i];
            a = STEM_SMLAD(xe[i], STEM_SXTB16(v), a);
            b = STEM_SMLAD(xo[i], STEM_SXTB16(STEM_ROR8(v)), b);
        }
        out[r] = a + b;
    }
#else
    /* plain C, 4 rows per pass (each window byte loaded once for 4 products) */
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

#if !defined(CONFIG_IDF_TARGET_ESP32S3)
/* a whole 3 x 3 int8 stem (padding 1, stride 1 or 2) without windows, on every board but the S3 (PIE has its own):
   the image kept as a ring of 3 padded rows (pixel ^ 0x80 = pixel - 128, a zero pixel left and right, zero rows above
   and below); then per strip of up to STEM3_STRIP output pixels of an output row, per channel, per kernel row, that
   row's 9 weights stay in registers while the strip's pixels stream past, sums in acc[]. SMLAD (Cortex-M33 and other
   DSP cores): the strip's 3 rows are split once into even / odd 16-bit halves (5 words per pixel and kernel row:
   bytes 0-3, 4-7 and 8), so the inner loop is 5 loads + 5 SMLAD. Plain C (Hazard3, ESP32): 2 channels per pass, each
   input byte loaded once for both. Measured on the Pico 2, ResNet-8's stem: M33 14.6 -> 9.2 ms, Hazard3 19.1 -> 12.3
   ms (benchmarks/experiments/chip_kernels/stem_i8_pico_bench.py). 0 = done; -1 = not this kernel's (in_w past
   STEM3_MAX_W), nothing written. Stack: ~0.6 KB of rows + 128 bytes (+ 0.9 KB with SMLAD). */
#define STEM3_MAX_W 64
#define STEM3_RING_BYTES ((((STEM3_MAX_W + 2) * 3 + 3) + 3) & ~3)  /* padded row + 3 bytes read past it (zero weights) */
#define STEM3_STRIP 16

static inline uint32_t stem3_ld32(const int8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }

static int stem3_rows(const uint8_t* img, int H, int W, int s, int out_h, int out_w, const int8_t* w, int row_bytes,
                      const int32_t* thr, int n_out, uint32_t* out) {
    if (W > STEM3_MAX_W || (s != 1 && s != 2)) return -1;
    int8_t ring[3][STEM3_RING_BYTES] __attribute__((aligned(4)));
    int32_t acc[STEM3_STRIP], acc2[STEM3_STRIP];
#if defined(__ARM_FEATURE_SIMD32) || defined(SYMBEX_HAL_TEST_SMLAD)
    uint32_t xs[3][STEM3_STRIP * 5];
    (void)acc2;
#endif
    const int pout = (n_out + 31) / 32;
    memset(out, 0, (size_t)out_h * out_w * pout * 4);
    int last = -2;                                           /* the last input row in the ring (-1, H: zero rows) */
    for (int oy = 0; oy < out_h; oy++) {
        const int iy0 = oy * s - 1;
        for (int r = last + 1 > iy0 ? last + 1 : iy0; r <= iy0 + 2; r++) {
            int8_t* p = ring[(r + 3) % 3];
            memset(p, 0, STEM3_RING_BYTES);
            if (r >= 0 && r < H) {
                const uint8_t* src = img + (size_t)r * W * 3;
                for (int i = 0; i < W * 3; i++) p[3 + i] = (int8_t)(src[i] ^ 0x80u);
            }
        }
        last = iy0 + 2;
        const int8_t* rows[3] = {ring[(iy0 + 3) % 3], ring[(iy0 + 4) % 3], ring[(iy0 + 5) % 3]};
        uint32_t* o_row = out + (size_t)oy * out_w * pout;
        for (int x0 = 0; x0 < out_w; x0 += STEM3_STRIP) {
            const int np = out_w - x0 < STEM3_STRIP ? out_w - x0 : STEM3_STRIP;
#if defined(__ARM_FEATURE_SIMD32) || defined(SYMBEX_HAL_TEST_SMLAD)
            for (int kh = 0; kh < 3; kh++)
                for (int i = 0; i < np; i++) {
                    const int8_t* q = rows[kh] + (x0 + i) * s * 3;
                    uint32_t* d = &xs[kh][i * 5];
                    const uint32_t v0 = stem3_ld32(q), v1 = stem3_ld32(q + 4), v2 = (uint8_t)q[8];
                    d[0] = STEM_SXTB16(v0);
                    d[1] = STEM_SXTB16(STEM_ROR8(v0));
                    d[2] = STEM_SXTB16(v1);
                    d[3] = STEM_SXTB16(STEM_ROR8(v1));
                    d[4] = STEM_SXTB16(v2);
                }
            for (int c = 0; c < n_out; c++) {
                for (int i = 0; i < np; i++) acc[i] = 0;
                const int8_t* wc = w + c * row_bytes;
                for (int kh = 0; kh < 3; kh++) {
                    const uint32_t v0 = stem3_ld32(wc + kh * 9), v1 = stem3_ld32(wc + kh * 9 + 4),
                                   v2 = (uint8_t)wc[kh * 9 + 8];
                    const uint32_t e0 = STEM_SXTB16(v0), o0 = STEM_SXTB16(STEM_ROR8(v0)), e1 = STEM_SXTB16(v1),
                                   o1 = STEM_SXTB16(STEM_ROR8(v1)), e2 = STEM_SXTB16(v2);
                    const uint32_t* xp = xs[kh];
                    for (int i = 0; i < np; i++, xp += 5) {
                        int32_t a = acc[i];
                        a = STEM_SMLAD(xp[0], e0, a);
                        a = STEM_SMLAD(xp[1], o0, a);
                        a = STEM_SMLAD(xp[2], e1, a);
                        a = STEM_SMLAD(xp[3], o1, a);
                        a = STEM_SMLAD(xp[4], e2, a);
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
#endif

SYMBEX_FAST_CODE void symbex_bool_stem_i8(const struct SymbexBoolStemI8* L, const uint8_t* img, uint32_t* out,
                                          int8_t* window, int32_t* scratch) {
    PF_T(t_all);
    const int k = L->kernel, pad = k / 2, s = L->stride, H = L->in_h, W = L->in_w;
    const int n = k * k * 3, n_out = L->out_ch, pout = (n_out + 31) / 32;
    const int32_t* thr = L->thresholds;
    int32_t* dot = scratch;
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    /* PIE: the whole 3 x 3 layer, 16 channels per instruction (refuses what it cannot run exactly: then the path below) */
    if (k == 3 && symbex_s8_stem3_bits(img, H, W, s, L->out_h, L->out_w, L->weights, L->row_bytes, thr, n_out,
                                       out) == 0) {
        PF_ADD(SYMBEX_PF_STEM_I8, t_all);
        return;
    }
#else
    /* every other board: the 3 x 3 stem without windows (stem3_rows; refuses images wider than STEM3_MAX_W) */
    if (k == 3 && stem3_rows(img, H, W, s, L->out_h, L->out_w, L->weights, L->row_bytes, thr, n_out, out) == 0) {
        PF_ADD(SYMBEX_PF_STEM_I8, t_all);
        return;
    }
#endif
    memset(window, 0, SYMBEX_BOOL_STEM_I8_WINDOW);           /* past n: zero, for the padded rows and PIE's blocks */
    for (int oy = 0; oy < L->out_h; oy++) {
        const int iy0 = oy * s - pad;
        for (int ox = 0; ox < L->out_w; ox++) {
            const int ix0 = ox * s - pad;
            /* the window [kh][kw][color] as int8: pixel - 128 is pixel XOR 0x80; outside the image 0 (= pixel 128,
               the zero padding of the centered input the network was trained on) */
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
            stem_i8_dot_rows(window, n, L->weights, L->row_bytes, n_out, dot);
            uint32_t* o_px = &out[(oy * L->out_w + ox) * pout];
            for (int ow = 0, o = 0; ow < pout; ow++) {
                uint32_t bits = 0;
                const int end = (ow + 1) * 32 < n_out ? (ow + 1) * 32 : n_out;
                for (int b = 0; o < end; o++, b++) bits |= (uint32_t)(dot[o] >= thr[o]) << b;
                o_px[ow] = bits;
            }
        }
    }
    PF_ADD(SYMBEX_PF_STEM_I8, t_all);
}

void symbex_bool_or_pool(const uint32_t* in, uint32_t* out, uint16_t ch, uint16_t h, uint16_t w) {
    PF_T(t_all);
    const int ppx = (ch + 31) / 32, h2 = h / 2, w2 = w / 2;
    for (int y = 0; y < h2; y++) {
        for (int x = 0; x < w2; x++) {
            const uint32_t* a = &in[((2 * y) * w + 2 * x) * ppx];
            const uint32_t* b = a + ppx;
            const uint32_t* c = a + w * ppx;
            const uint32_t* d = c + ppx;
            uint32_t* o = &out[(y * w2 + x) * ppx];
            for (int j = 0; j < ppx; j++) o[j] = a[j] | b[j] | c[j] | d[j];
        }
    }
    PF_ADD(SYMBEX_PF_OR_POOL, t_all);
}

void symbex_bool_or(uint32_t* dst, const uint32_t* src, uint32_t n_words) {
    PF_T(t_all);
    for (uint32_t i = 0; i < n_words; i++) dst[i] |= src[i];
    PF_ADD(SYMBEX_PF_OR, t_all);
}

void symbex_bool_cell_head(const uint32_t* map, uint16_t h, uint16_t w, uint16_t ch, const uint32_t* weights,
                           uint16_t row_words, uint16_t n_classes, const int32_t* scale, const int32_t* bias,
                           int32_t* scratch, uint8_t* out) {
    PF_T(t_all);
    const int ppx = (ch + 31) / 32;
    /* every cell is copied to an aligned buffer padded with zeros up to 4 words: with PIE (ESP32-S3) the count runs
       over whole 128-bit blocks, and reading the map in place would count the next cell's words too */
    uint32_t cell[SYMBEX_BOOL_MAX_CELL_WORDS + 4] __attribute__((aligned(16)));
    const int n4 = (ppx + 3) & ~3;
    for (int i = ppx; i < n4; i++) cell[i] = 0;
    for (uint32_t p = 0; p < (uint32_t)h * w; p++) {
        for (int i = 0; i < ppx; i++) cell[i] = map[p * ppx + i];
        /* the bits above ch are 0 in the cell and in the rows: count(cell XOR row) is the differences directly */
        symbex_xor_popcount_rows_n(cell, weights, ppx, row_words, n_classes, scratch);
        int best = 0;
        int64_t best_logit = 0;
        for (int o = 0; o < n_classes; o++) {
            const int64_t logit = (int64_t)scale[o] * ((int32_t)ch - 2 * scratch[o]) + bias[o];
            if (o == 0 || logit > best_logit) { best_logit = logit; best = o; }
        }
        out[p] = (uint8_t)best;
    }
    PF_ADD(SYMBEX_PF_CLASSIFY, t_all);
}

int symbex_bool_classify(const uint32_t* bits, uint16_t n_words, uint32_t n_bits, const uint32_t* weights,
                         uint16_t row_words, uint16_t n_classes, const int32_t* weight_counts,
                         const int32_t* scale, const int32_t* bias, int32_t* scratch) {
    PF_T(t_all);
    int32_t pcx = 0;
    for (uint16_t i = 0; i < n_words; i++) pcx += symbex_hal_popcount32(bits[i]);
    symbex_and_popcount_rows(bits, weights, row_words, n_classes, scratch);
    int best = 0;
    int64_t best_logit = 0;
    for (int o = 0; o < n_classes; o++) {
        const int32_t d = pcx + weight_counts[o] - 2 * scratch[o];
        const int64_t logit = (int64_t)scale[o] * ((int32_t)n_bits - 2 * d) + bias[o];
        if (o == 0 || logit > best_logit) { best_logit = logit; best = o; }
    }
    PF_ADD(SYMBEX_PF_CLASSIFY, t_all);
    return best;
}

SYMBEX_FAST_CODE int symbex_bool_pool_head(const struct SymbexBoolPoolHead* head, const uint32_t* map,
                                           uint32_t* counts, int32_t* scratch) {
    PF_T(t_all);
    const int k = head->window, ppx = (head->channels + 31) / 32, rw = head->row_words, np = head->n_planes;
    const int nwy = head->in_h / k, nwx = head->in_w / k, nc = head->n_classes;
    memset(counts, 0, (size_t)np * rw * 4);
    for (int wy = 0; wy < nwy; wy++)                 /* bit-sliced counters: add each pixel's word to its window's */
        for (int wx = 0; wx < nwx; wx++)
            for (int dy = 0; dy < k; dy++)
                for (int dx = 0; dx < k; dx++) {
                    const uint32_t* px = map + ((size_t)(wy * k + dy) * head->in_w + wx * k + dx) * ppx;
                    uint32_t* at = counts + (wy * nwx + wx) * ppx;
                    for (int j = 0; j < ppx; j++) {
                        uint32_t carry = px[j];
                        for (int p = 0; p < np && carry; p++) {
                            const uint32_t c = at[p * rw + j];
                            at[p * rw + j] = c ^ carry;
                            carry &= c;
                        }
                    }
                }
    int32_t* cnt = scratch;
    int32_t* dot = scratch + nc;
    for (int o = 0; o < nc; o++) dot[o] = 0;
    for (int p = 0; p < np; p++)
        for (int q = 0; q < 8; q++) {
            symbex_and_popcount_rows(counts + p * rw, head->weight_planes + (size_t)q * nc * rw, rw, nc, cnt);
            for (int o = 0; o < nc; o++)
                dot[o] += (q == 7 ? -128 * cnt[o] : cnt[o] * (1 << q)) * (1 << p);
        }
    int best = 0;
    int64_t best_logit = 0;
    for (int o = 0; o < nc; o++) {
        const int64_t logit = (int64_t)head->scale[o] * dot[o] + head->bias[o];
        if (o == 0 || logit > best_logit) { best_logit = logit; best = o; }
    }
    PF_ADD(SYMBEX_PF_CLASSIFY, t_all);
    return best;
}
