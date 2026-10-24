/*
 * symbex_dense.c -- multi-bit dense (vector) layer.
 *
 * MIGRATION CHANGE: each bit plane is processed with ONE call to symbex_xnor_popcount
 * (hal/, PIE-accelerated on the ESP32-S3) instead of a word-by-word loop with the old
 * SYMBEX_BIT_MAC. Mathematically identical: summing first and shifting afterwards gives the
 * same result as shifting each word and summing (the shift distributes over the sum).
 *
 * symbex_dense_score_neuron is shared by forward and argmax, so the k_bits switch is not duplicated.
 *
 * SYMBEX v3 fix: the HAL receives the REAL number of inputs (in_features), not in_words*32.
 * Before, every padding bit of the last word added +1 to the score of every neuron (and in the
 * classifier that error was multiplied by a different scale per class). With in_features a
 * multiple of 32 the result is identical to before.
 *
 * Weights and thresholds are read with SYMBEX_READ_W32 (core/symbex_core.h): a normal read on every chip,
 * pgm_read_dword on AVR, where they live in flash (PROGMEM).
 *
 * AVR fix: int is 16 bits there, so __builtin_ctz(uint32_t) only saw the low 16 bits; CTZ32 uses
 * __builtin_ctzl on AVR.
 *
 * PENDING (not changed here): with k_bits > 1 each weight plane is compared with a DIFFERENT
 * input plane (input_planes[k * in_words]). That is only correct if the input carries k planes;
 * between layers, symbex_bitpack_from_scores writes only ONE.
 *
 * Whole layers: with k_bits = 1 the binary layers are ONE call to the HAL (symbex_xnor_popcount_rows), in chunks of
 * ROWS_CHUNK neurons where only bits are kept (forward_bits, argmax): the per-neuron call and its checks were
 * ~65-90 cycles per neuron on the ESP32 / S3 (measured), more than half the time on rows of 1-4 words.
 * First layer, uint8 x 1 bit, on 32-bit chips: SWAR, 4 inputs per 32-bit operation (ops/symbex_swar.h);
 * -DSYMBEX_U8_SCALAR keeps the previous one-input-at-a-time mask (to compare). AVR keeps its 8-bit path.
 *
 * DEBUG: SYMBEX_DEBUG_CLASSIFIER prints raw/logit_q per class in
 * symbex_binary_classifier_argmax, with printf, not Serial (this is a plain .c file;
 * Serial is an Arduino C++ object and is not available here).
 */

#include <stdio.h>
#include "layers/dense/symbex_dense.h"
#include "hal/symbex_hal.h"
#include "core/symbex_core.h"
#include "ops/symbex_swar.h"

/* neurons per HAL call when only their bits (or the best one) are kept: a small int32 array on the stack */
#define ROWS_CHUNK 32

//#define SYMBEX_DEBUG_CLASSIFIER

// Evaluation order: 1 = plane 0 is the most significant bit (MSB)
#define SYMBEX_PLANE_0_IS_MSB 1

/* count of trailing zeros of a 32-bit word (int is 16 bits on AVR) */
#if defined(__AVR__)
  #define CTZ32(x) __builtin_ctzl(x)
#else
  #define CTZ32(x) __builtin_ctz(x)
#endif

/* a threshold (int32) that may live in flash */
#define THRESHOLD(t, j) ((int32_t)SYMBEX_READ_W32((const uint32_t*)&(t)[j]))

static int32_t symbex_dense_score_neuron(const uint32_t* input_planes, const uint32_t** w_ptr_io,
                                        uint16_t in_words, uint16_t n_bits, uint16_t k_bits) {
    const uint32_t* w_ptr = *w_ptr_io;
    int32_t score = 0;

    switch (k_bits) {
        case 1:
            score += symbex_xnor_popcount(input_planes, w_ptr, in_words, n_bits);
            w_ptr += in_words;
            break;

#if SYMBEX_PLANE_0_IS_MSB
        case 8:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, n_bits) << 7; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, n_bits) << 6; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, n_bits) << 5; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, n_bits) << 4; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[4 * in_words], w_ptr, in_words, n_bits) << 3; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[5 * in_words], w_ptr, in_words, n_bits) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[6 * in_words], w_ptr, in_words, n_bits) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[7 * in_words], w_ptr, in_words, n_bits) << 0; w_ptr += in_words;
            break;
        case 4:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, n_bits) << 3; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, n_bits) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, n_bits) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, n_bits) << 0; w_ptr += in_words;
            break;
        default:
            for (uint16_t k = 0; k < k_bits; k++) {
                score += symbex_xnor_popcount(&input_planes[k * in_words], w_ptr, in_words, n_bits) << (k_bits - 1 - k);
                w_ptr += in_words;
            }
            break;
#else
        case 8:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, n_bits) << 0; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, n_bits) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, n_bits) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, n_bits) << 3; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[4 * in_words], w_ptr, in_words, n_bits) << 4; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[5 * in_words], w_ptr, in_words, n_bits) << 5; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[6 * in_words], w_ptr, in_words, n_bits) << 6; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[7 * in_words], w_ptr, in_words, n_bits) << 7; w_ptr += in_words;
            break;
        case 4:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, n_bits) << 0; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, n_bits) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, n_bits) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, n_bits) << 3; w_ptr += in_words;
            break;
        default:
            for (uint16_t k = 0; k < k_bits; k++) {
                score += symbex_xnor_popcount(&input_planes[k * in_words], w_ptr, in_words, n_bits) << k;
                w_ptr += in_words;
            }
            break;
#endif
    }

    *w_ptr_io = w_ptr;
    return score;
}

void symbex_dense_forward(const uint32_t* input_planes, int32_t* output_buffer, const struct SymbexLayerData* layer) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->weights;

    if (layer->k_bits == 1) {                             /* the whole layer in one call */
        symbex_xnor_popcount_rows(input_planes, w_ptr, in_words, layer->in_features, layer->out_features, output_buffer);
        return;
    }
    for (uint16_t n = 0; n < layer->out_features; n++) {
        output_buffer[n] = symbex_dense_score_neuron(input_planes, &w_ptr, in_words, layer->in_features, layer->k_bits);
    }
}

int symbex_dense_argmax(const uint32_t* input_planes, const struct SymbexLayerData* layer) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->weights;

    int best_class = 0;
    int32_t max_score = -2147483647L - 1;

    if (layer->k_bits == 1) {                             /* chunks of ROWS_CHUNK neurons per HAL call */
        int32_t s[ROWS_CHUNK];
        for (uint16_t n0 = 0; n0 < layer->out_features; n0 += ROWS_CHUNK) {
            const uint16_t m = (uint16_t)((layer->out_features - n0) < ROWS_CHUNK ? (layer->out_features - n0) : ROWS_CHUNK);
            symbex_xnor_popcount_rows(input_planes, w_ptr + (uint32_t)n0 * in_words, in_words, layer->in_features, m, s);
            for (uint16_t i = 0; i < m; i++) {
                if (s[i] > max_score) { max_score = s[i]; best_class = n0 + i; }
            }
        }
        return best_class;
    }
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t score = symbex_dense_score_neuron(input_planes, &w_ptr, in_words, layer->in_features, layer->k_bits);
        if (score > max_score) {
            max_score = score;
            best_class = n;
        }
    }

    return best_class;
}

int symbex_binary_classifier_argmax(
    const uint32_t* input_planes,
    const struct SymbexLayerData* layer,
    const int64_t* scale, const int32_t* shift_bn,
    uint8_t q_act, uint8_t q_mult
) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->weights;

    uint8_t rescale_shift = q_mult - q_act;
    int64_t rounding = (int64_t)1 << (rescale_shift - 1);

    int best_class = 0;
    int64_t max_logit_q = INT64_MIN;
    int32_t raws[ROWS_CHUNK];

    for (uint16_t n = 0; n < layer->out_features; n++) {
        // k_bits is always 1 here: the classifier is purely binary, no multi-bit slicing.
        // The raw scores come from the HAL in chunks of ROWS_CHUNK classes (one call per chunk).
        if (n % ROWS_CHUNK == 0) {
            const uint16_t m = (uint16_t)((layer->out_features - n) < ROWS_CHUNK ? (layer->out_features - n) : ROWS_CHUNK);
            symbex_xnor_popcount_rows(input_planes, w_ptr + (uint32_t)n * in_words, in_words, layer->in_features, m, raws);
        }
        int32_t raw = raws[n % ROWS_CHUNK];
        int64_t logit_q = (((int64_t)raw * scale[n]) + rounding) >> rescale_shift;
        logit_q += shift_bn[n];

        #ifdef SYMBEX_DEBUG_CLASSIFIER
        printf("    class %d | raw=%ld | logit_q=%ld\n", n, (long)raw, (long)logit_q);
        #endif

        if (logit_q > max_logit_q) {
            max_logit_q = logit_q;
            best_class = n;
        }
    }

    return best_class;
}

/* ---- SYMBEX v3: exported 1-bit MLPs ---- */

void symbex_dense_input_u8(const uint8_t* x, int32_t* out, const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    int32_t total = 0;                                    /* sum of all inputs, the same for every neuron */
    for (uint16_t i = 0; i < n_in; i++) total += x[i];
    const uint32_t* w = layer->weights;
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t positives = 0;                            /* sum of the inputs whose weight is +1 */
        for (uint16_t p = 0; p < in_words; p++) {
            uint32_t word = SYMBEX_READ_W32(&w[p]);
            const uint16_t base = (uint16_t)(p * 32);
            while (word) {
                const uint16_t i = (uint16_t)(base + CTZ32(word));
                if (i < n_in) positives += x[i];
                word &= word - 1;
            }
        }
        out[n] = 2 * positives - total;                    /* sum(+x) - sum(-x) */
        w += in_words;
    }
}

void symbex_dense_threshold(const int32_t* raw, const int32_t* threshold, uint32_t* bits, uint16_t n) {
    const uint16_t words = (n + 31) / 32;
    for (uint16_t p = 0; p < words; p++) bits[p] = 0;
    for (uint16_t j = 0; j < n; j++) {
        if (raw[j] >= threshold[j]) bits[j / 32] |= (1UL << (j % 32));
    }
}

/*
 * First layer, bit-plane version: the uint8 inputs are split into their 8 bit planes once per image
 * (planes = 8 * ceil(in_features / 32) words, provided by the caller), and each neuron then needs
 * 8 x ceil(in / 32) AND + popcount instead of visiting every input:
 *     sum_i x_i * s_i = 2 * sum_{s_i = +1} x_i - sum_i x_i,    sum_{s_i = +1} x_i = sum_k 2^k popcount(plane_k AND w)
 * Exactly the same result as symbex_dense_input_u8 (integers only).
 */
void symbex_dense_input_u8_planes(const uint8_t* x, uint32_t* planes, int32_t* out,
                                  const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    for (uint16_t p = 0; p < 8 * in_words; p++) planes[p] = 0;
    int32_t total = 0;
    for (uint16_t i = 0; i < n_in; i++) {
        uint32_t v = x[i];
        total += (int32_t)v;
        const uint32_t bit = 1UL << (i % 32);
        uint32_t* column = &planes[i / 32];
        while (v) {                                         /* only the bits that are set */
            const uint32_t k = (uint32_t)CTZ32(v);
            column[k * in_words] |= bit;
            v &= v - 1;
        }
    }
    /* the whole layer in one HAL call: each weight word read once for the 8 planes, the counts inlined */
    symbex_and_planes_rows(planes, 8, 0, layer->weights, in_words, layer->out_features, out);
    for (uint16_t n = 0; n < layer->out_features; n++) out[n] = 2 * out[n] - total;
}

/* sum of the inputs whose weight bit is 1, for one neuron.
 * 32-bit chips: mask method (no branches, each weight word read once).
 * 8-bit chips (AVR, or -DSYMBEX_U8_BYTEWISE to check it on a PC): the weights are read ONE BYTE at a time and
 * tested with an 8-bit mask that moves one bit per input. On AVR a 32-bit shift by a variable amount is a loop
 * of single-bit shifts over 4 bytes, so (wv >> b) cost ~100 cycles per input; this costs a few. The sum of one
 * word (at most 32 x 255) fits in 16 bits. Same result, bit for bit. */
#if defined(__AVR__)
  #define READ_B8(p) pgm_read_byte(p)
#else
  #define READ_B8(p) (*(p))
#endif
static int32_t u8_positives(const uint8_t* x, const uint32_t* w, uint16_t n_in, uint16_t in_words) {
#if !defined(__AVR__) && !defined(SYMBEX_U8_BYTEWISE) && !defined(SYMBEX_U8_SCALAR)
    (void)in_words;
    return symbex_swar_u8_masked_sum(x, w, n_in);            /* 4 inputs per 32-bit operation */
#else
    int32_t positives = 0;
#endif
#if defined(__AVR__) || defined(SYMBEX_U8_BYTEWISE)
    const uint8_t* wb = (const uint8_t*)w;                 /* little-endian: byte q of a word = its bits 8q..8q+7 */
    for (uint16_t p = 0; p < in_words; p++) {
        const uint8_t* xp = &x[p * 32];
        const uint16_t lim = (uint16_t)((n_in - p * 32) < 32 ? (n_in - p * 32) : 32);
        uint16_t sum = 0;
        uint8_t i = 0;
        for (uint8_t q = 0; q < 4 && i < lim; q++) {
            const uint8_t bits = READ_B8(&wb[p * 4 + q]);
            for (uint8_t m = 1; m && i < lim; m <<= 1, i++) {
                if (bits & m) sum += xp[i];
            }
        }
        positives += sum;
    }
    return positives;
#elif defined(SYMBEX_U8_SCALAR)
    for (uint16_t p = 0; p < in_words; p++) {
        const uint32_t wv = SYMBEX_READ_W32(&w[p]);
        const uint8_t* xp = &x[p * 32];
        const uint16_t lim = (uint16_t)((n_in - p * 32) < 32 ? (n_in - p * 32) : 32);
        for (uint16_t b = 0; b < lim; b++) positives += xp[b] & -(int32_t)((wv >> b) & 1UL);
    }
    return positives;
#endif
}

/*
 * First layer, mask version: no bit planes and no popcount. For each weight word, every input is added
 * through a mask (0x00 or 0xFF from its weight bit), with no branches:
 *     sum_{s_i = +1} x_i = sum_i x_i AND mask_i
 * Each weight word is read once. Exactly the same result as symbex_dense_input_u8.
 */
void symbex_dense_input_u8_mask(const uint8_t* x, int32_t* out, const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    int32_t total = 0;
    for (uint16_t i = 0; i < n_in; i++) total += x[i];
    const uint32_t* w = layer->weights;
    for (uint16_t n = 0; n < layer->out_features; n++) {
        out[n] = 2 * u8_positives(x, w, n_in, in_words) - total;
        w += in_words;
    }
}

/* ---- the same layers, writing each neuron's bit directly (no raw-score array: for chips with little RAM) ---- */

void symbex_dense_forward_bits(const uint32_t* input_bits, const int32_t* threshold, uint32_t* out_bits,
                               const struct SymbexLayerData* layer) {
    const uint16_t in_words = (layer->in_features + 31) / 32;
    const uint16_t n_out = layer->out_features;
    for (uint16_t p = 0; p < (n_out + 31) / 32; p++) out_bits[p] = 0;
    const uint32_t* w = layer->weights;
    int32_t raw[ROWS_CHUNK];                               /* chunks of ROWS_CHUNK neurons per HAL call */
    for (uint16_t n0 = 0; n0 < n_out; n0 += ROWS_CHUNK) {
        const uint16_t m = (uint16_t)((n_out - n0) < ROWS_CHUNK ? (n_out - n0) : ROWS_CHUNK);
        symbex_xnor_popcount_rows(input_bits, w + (uint32_t)n0 * in_words, in_words, layer->in_features, m, raw);
        for (uint16_t i = 0; i < m; i++) {
            const uint16_t n = (uint16_t)(n0 + i);
            if (raw[i] >= THRESHOLD(threshold, n)) out_bits[n / 32] |= 1UL << (n % 32);
        }
    }
}

void symbex_dense_input_u8_bits(const uint8_t* x, const int32_t* threshold, uint32_t* out_bits,
                                const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    const uint16_t n_out = layer->out_features;
    int32_t total = 0;
    for (uint16_t i = 0; i < n_in; i++) total += x[i];
    for (uint16_t p = 0; p < (n_out + 31) / 32; p++) out_bits[p] = 0;
    const uint32_t* w = layer->weights;
    for (uint16_t n = 0; n < n_out; n++) {
        const int32_t raw = 2 * u8_positives(x, w, n_in, in_words) - total;
        if (raw >= THRESHOLD(threshold, n)) out_bits[n / 32] |= 1UL << (n % 32);
        w += in_words;
    }
}

/*
 * First layer + threshold in one pass, for inputs that carry only their k most significant bits (the CDT's
 * [data] input_bits = k < 8): the k top bit planes of the input are built once per image (planes: caller buffer
 * of k * ceil(in_features / 32) words), and each neuron needs k x ceil(in / 32) AND + popcount:
 *     sum_{s_i = +1} x_i = sum_{bit b in the top k} 2^b popcount(plane_b AND w)
 * Bits below the top k are ignored (masked), so the result equals symbex_dense_input_u8_bits on the quantized
 * inputs (x_i with its low 8 - k bits cleared). With k = 1 each neuron costs as much as a binary layer.
 */
void symbex_dense_input_u8_planes_bits(const uint8_t* x, uint8_t k, uint32_t* planes, const int32_t* threshold,
                                       uint32_t* out_bits, const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    const uint16_t n_out = layer->out_features;
    const uint8_t low = (uint8_t)(8 - k);                      /* plane q holds bit (low + q) of every input */
    const uint8_t keep = (uint8_t)(0xFFu << low);
    for (uint16_t p = 0; p < (uint16_t)(k * in_words); p++) planes[p] = 0;
    int32_t total = 0;
    for (uint16_t i = 0; i < n_in; i++) {
        const uint8_t v = (uint8_t)(x[i] & keep);
        total += v;
        const uint32_t bit = 1UL << (i % 32);
        for (uint8_t q = 0; q < k; q++) {
            if (v & (uint8_t)(1u << (low + q))) planes[q * in_words + i / 32] |= bit;
        }
    }
    for (uint16_t p = 0; p < (n_out + 31) / 32; p++) out_bits[p] = 0;
#if defined(__AVR__) || defined(SYMBEX_PLANES_PER_NEURON)
    /* 8-bit chips: the original per-neuron loop. The whole-layer HAL call made this layer ~39% slower on the
       ATmega328P with k = 1 (measured: 11.6 vs 8.35 ms for 64 -> 512); on 32-bit chips it is ~2x faster. */
    const uint32_t* w = layer->weights;
    for (uint16_t n = 0; n < n_out; n++) {
        int32_t positives = 0;
        for (uint16_t p = 0; p < in_words; p++) {
            const uint32_t wv = SYMBEX_READ_W32(&w[p]);
            for (uint8_t q = 0; q < k; q++) {
                positives += (int32_t)symbex_hal_popcount32(planes[q * in_words + p] & wv) << (low + q);
            }
        }
        if (2 * positives - total >= THRESHOLD(threshold, n)) out_bits[n / 32] |= 1UL << (n % 32);
        w += in_words;
    }
#else
    int32_t pos[ROWS_CHUNK];                               /* chunks of ROWS_CHUNK neurons per HAL call */
    for (uint16_t n0 = 0; n0 < n_out; n0 += ROWS_CHUNK) {
        const uint16_t m = (uint16_t)((n_out - n0) < ROWS_CHUNK ? (n_out - n0) : ROWS_CHUNK);
        symbex_and_planes_rows(planes, k, low, layer->weights + (uint32_t)n0 * in_words, in_words, m, pos);
        for (uint16_t i = 0; i < m; i++) {
            const uint16_t n = (uint16_t)(n0 + i);
            if (2 * pos[i] - total >= THRESHOLD(threshold, n)) out_bits[n / 32] |= 1UL << (n % 32);
        }
    }
#endif
}
