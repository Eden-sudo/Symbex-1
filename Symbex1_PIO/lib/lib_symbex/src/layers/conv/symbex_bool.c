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
    "OR pool 2x2", "OR shortcut", "classifier"};
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
   npal = words of the window: nothing is written past it */
static inline void put_bits(uint32_t* win, const uint32_t* src, int nbits, int off, int npal) {
    for (int j = 0; nbits > 0; j++, nbits -= 32, off += 32) {
        const uint32_t v = src[j];
        const int w = off >> 5, sh = off & 31;
        win[w] |= v << sh;
        if (sh && w + 1 < npal) win[w + 1] |= v >> (32 - sh);
    }
}

void symbex_bool_conv(const struct SymbexBoolConv* L, const uint32_t* in, uint32_t* out,
                      uint32_t* window, int32_t* scratch) {
    const int xor_form = symbex_bool_force_mode ? symbex_bool_force_mode == 2 : symbex_hal_prefers_xor();
    const int pin = (L->in_ch + 31) / 32, pout = (L->out_ch + 31) / 32, n_out = L->out_ch;
    const int k = L->kernel, pad = k / 2, kk = k * k, fila = k * pin;
    const int cin = L->in_ch, dense = (cin & 31) != 0;   /* channels not a multiple of 32: the window is packed */
    const int npal = (kk * cin + 31) / 32;               /* words of the window (= kk * pin when not packed) */
    const int H = L->in_h, W = L->in_w, s = L->stride, st = L->row_words;
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
    for (int i = npal; i < st; i++) window[i] = 0;   /* row padding, once: the copies below never write there */
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
                for (int i = 0; i < npal; i++) window[i] = 0;
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        const int y = y0 + kh, x = x0 + kw, q = kh * k + kw;
                        if (y >= 0 && y < H && x >= 0 && x < W) {
                            put_bits(window, &in[(y * W + x) * pin], cin, q * cin, npal);
                            nv++;
                        } else {
                            invalid[ni++] = q;
                        }
                    }
                }
            } else if (interior) {
                /* the k rows of the window are k runs of `fila` words of the map: a word loop (memcpy's fixed cost
                   was most of the time for copies this small) */
                for (int kh = 0; kh < k; kh++) {
                    const uint32_t* src = &in[((y0 + kh) * W + x0) * pin];
                    uint32_t* dst = &window[kh * fila];
                    for (int i = 0; i < fila; i++) dst[i] = src[i];
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
                symbex_xor_popcount_rows_n(window, L->weights, npal, st, n_out, cnt);
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
                symbex_and_popcount_rows_n(window, window, npal, st, 1, &pcx);    /* popcount(window): one call */
                PF_ADD(SYMBEX_PF_WINDOW_POPCOUNT, t_pcx);
                PF_T(t_cnt);
                symbex_and_popcount_rows_n(window, L->weights, npal, st, n_out, cnt);
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
