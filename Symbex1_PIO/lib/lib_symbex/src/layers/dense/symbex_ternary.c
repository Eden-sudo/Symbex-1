/*
 * symbex_ternary.c -- ternary dense layer (see symbex_ternary.h). Reference kernels: integers only, no
 * multiplications, popcount through the HAL (symbex_hal_popcount32), weights read with SYMBEX_READ_W32 so
 * they may live in flash.
 */

#include "layers/dense/symbex_ternary.h"
#include "hal/symbex_hal.h"
#include "core/symbex_core.h"

void symbex_ternary_forward(const uint32_t* input_bits, int32_t* out, const struct SymbexTernaryLayerData* layer) {
    const uint16_t in_words = (uint16_t)((layer->in_features + 31) / 32);
    const uint32_t* s = layer->signs;
    const uint32_t* m = layer->mask;
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t matching = 0, differing = 0;
        for (uint16_t p = 0; p < in_words; p++) {
            const uint32_t mv = SYMBEX_READ_W32(&m[p]);
            const uint32_t d = input_bits[p] ^ SYMBEX_READ_W32(&s[p]);
            matching += symbex_hal_popcount32(mv & ~d);
            differing += symbex_hal_popcount32(mv & d);
        }
        out[n] = matching - differing;
        s += in_words;
        m += in_words;
    }
}

void symbex_ternary_input_u8_mask(const uint8_t* x, int32_t* out, const struct SymbexTernaryLayerData* layer) {
    const uint16_t n_in = layer->in_features;
    const uint16_t in_words = (uint16_t)((n_in + 31) / 32);
    const uint32_t* s = layer->signs;
    const uint32_t* m = layer->mask;
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t positive = 0, negative = 0;
        for (uint16_t p = 0; p < in_words; p++) {
            const uint32_t mv = SYMBEX_READ_W32(&m[p]);
            const uint32_t sv = SYMBEX_READ_W32(&s[p]);
            const uint32_t plus = mv & sv;              /* weights +1 */
            const uint32_t minus = mv & ~sv;            /* weights -1 */
            const uint8_t* xp = &x[p * 32];
            const uint16_t lim = (uint16_t)((n_in - p * 32) < 32 ? (n_in - p * 32) : 32);
            for (uint16_t b = 0; b < lim; b++) {
                positive += xp[b] & -(int32_t)((plus >> b) & 1u);
                negative += xp[b] & -(int32_t)((minus >> b) & 1u);
            }
        }
        out[n] = positive - negative;
        s += in_words;
        m += in_words;
    }
}
