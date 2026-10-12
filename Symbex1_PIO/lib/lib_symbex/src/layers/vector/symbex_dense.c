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
 * PENDING (not changed here): with k_bits > 1 each weight plane is compared with a DIFFERENT
 * input plane (input_planes[k * in_words]). That is only correct if the input carries k planes;
 * between layers, symbex_bitpack_from_scores writes only ONE.
 *
 * DEBUG: SYMBEX_DEBUG_CLASIFICADOR prints raw/logit_q per class in
 * symbex_clasificador_binario_argmax, with printf, not Serial (this is a plain .c file;
 * Serial is an Arduino C++ object and is not available here).
 */

#include <stdio.h>
#include "layers/vector/symbex_dense.h"
#include "hal/symbex_hal.h"
#include "core/symbex_core.h"

//#define SYMBEX_DEBUG_CLASIFICADOR

// Evaluation order: 1 = plane 0 is the most significant bit (MSB)
#define SYMBEX_PLANE_0_IS_MSB 1

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

    for (uint16_t n = 0; n < layer->out_features; n++) {
        output_buffer[n] = symbex_dense_score_neuron(input_planes, &w_ptr, in_words, layer->in_features, layer->k_bits);
    }
}

int symbex_dense_argmax(const uint32_t* input_planes, const struct SymbexLayerData* layer) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->weights;

    int best_class = 0;
    int32_t max_score = -2147483647 - 1;

    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t score = symbex_dense_score_neuron(input_planes, &w_ptr, in_words, layer->in_features, layer->k_bits);
        if (score > max_score) {
            max_score = score;
            best_class = n;
        }
    }

    return best_class;
}

int symbex_clasificador_binario_argmax(
    const uint32_t* input_planes,
    const struct SymbexLayerData* layer,
    const int64_t* escala, const int32_t* shift_bn,
    uint8_t q_act, uint8_t q_mult
) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->weights;

    uint8_t shift_reescalado = q_mult - q_act;
    int64_t redondeo = (int64_t)1 << (shift_reescalado - 1);

    int best_class = 0;
    int64_t max_logit_q = INT64_MIN;

    for (uint16_t n = 0; n < layer->out_features; n++) {
        // k_bits is always 1 here: the classifier is purely binary,
        // no multi-bit slicing.
        int32_t raw = symbex_dense_score_neuron(input_planes, &w_ptr, in_words, layer->in_features, 1);
        int64_t logit_q = (((int64_t)raw * escala[n]) + redondeo) >> shift_reescalado;
        logit_q += shift_bn[n];

        #ifdef SYMBEX_DEBUG_CLASIFICADOR
        printf("    clase %d | raw=%ld | logit_q=%ld\n", n, (long)raw, (long)logit_q);
        #endif

        if (logit_q > max_logit_q) {
            max_logit_q = logit_q;
            best_class = n;
        }
    }

    return best_class;
}

/* ---- SYMBEX v3: exported 1-bit MLPs ---- */

void symbex_dense_entrada_u8(const uint8_t* x, int32_t* out, const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    int32_t total = 0;                                  /* sum of all inputs, the same for every neuron */
    for (uint16_t i = 0; i < n_in; i++) total += x[i];
    const uint32_t* w = layer->weights;
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t positivos = 0;                          /* sum of the inputs whose weight is +1 */
        for (uint16_t p = 0; p < in_words; p++) {
            uint32_t palabra = SYMBEX_READ_W32(&w[p]);
            const uint16_t base = (uint16_t)(p * 32);
            while (palabra) {
                const uint16_t i = (uint16_t)(base + __builtin_ctz(palabra));
                if (i < n_in) positivos += x[i];
                palabra &= palabra - 1;
            }
        }
        out[n] = 2 * positivos - total;                 /* sum(+x) - sum(-x) */
        w += in_words;
    }
}

void symbex_dense_umbral(const int32_t* raw, const int32_t* umbral, uint32_t* bits, uint16_t n) {
    const uint16_t palabras = (n + 31) / 32;
    for (uint16_t p = 0; p < palabras; p++) bits[p] = 0;
    for (uint16_t j = 0; j < n; j++) {
        if (raw[j] >= umbral[j]) bits[j / 32] |= (1UL << (j % 32));
    }
}

/*
 * First layer, bit-plane version: the uint8 inputs are split into their 8 bit planes once per image
 * (planos = 8 * ceil(in_features / 32) words, provided by the caller), and each neuron then needs
 * 8 x ceil(in / 32) AND + popcount instead of visiting every input:
 *     sum_i x_i * s_i = 2 * sum_{s_i = +1} x_i - sum_i x_i,   sum_{s_i = +1} x_i = sum_k 2^k popcount(plane_k AND w)
 * Exactly the same result as symbex_dense_entrada_u8 (integers only).
 */
void symbex_dense_entrada_u8_planos(const uint8_t* x, uint32_t* planos, int32_t* out,
                                    const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    for (uint16_t p = 0; p < 8 * in_words; p++) planos[p] = 0;
    int32_t total = 0;
    for (uint16_t i = 0; i < n_in; i++) {
        uint32_t v = x[i];
        total += (int32_t)v;
        const uint32_t bit = 1UL << (i % 32);
        uint32_t* columna = &planos[i / 32];
        while (v) {                                     /* only the bits that are set */
            const uint32_t k = (uint32_t)__builtin_ctz(v);
            columna[k * in_words] |= bit;
            v &= v - 1;
        }
    }
    const uint32_t* w = layer->weights;
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint16_t p = 0; p < in_words; p++) {
            const uint32_t wv = SYMBEX_READ_W32(&w[p]);   /* each weight word is read ONCE (it may come from flash) */
            for (uint16_t k = 0; k < 8; k++) s[k] += symbex_hal_popcount32(planos[k * in_words + p] & wv);
        }
        int32_t positivos = 0;
        for (uint16_t k = 0; k < 8; k++) positivos += s[k] << k;
        out[n] = 2 * positivos - total;
        w += in_words;
    }
}

/*
 * First layer, mask version: no bit planes and no popcount. For each weight word, every input is added
 * through a mask (0x00 or 0xFF from its weight bit), with no branches:
 *     sum_{s_i = +1} x_i = sum_i x_i AND mask_i
 * Each weight word is read once. Exactly the same result as symbex_dense_entrada_u8.
 */
void symbex_dense_entrada_u8_mascara(const uint8_t* x, int32_t* out, const struct SymbexLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (n_in + 31) / 32;
    int32_t total = 0;
    for (uint16_t i = 0; i < n_in; i++) total += x[i];
    const uint32_t* w = layer->weights;
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t positivos = 0;
        for (uint16_t p = 0; p < in_words; p++) {
            const uint32_t wv = SYMBEX_READ_W32(&w[p]);
            const uint8_t* xp = &x[p * 32];
            const uint16_t lim = (uint16_t)((n_in - p * 32) < 32 ? (n_in - p * 32) : 32);
            for (uint16_t b = 0; b < lim; b++) positivos += xp[b] & -(int32_t)((wv >> b) & 1u);
        }
        out[n] = 2 * positivos - total;
        w += in_words;
    }
}
