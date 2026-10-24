/*
 * symbex_ternary.h -- TERNARY dense layer: weights in {-1, 0, +1}, stored as two bit planes.
 *
 * Per neuron, ceil(in_features / 32) words in each plane, bit i of word i / 32 = input i (LSB first):
 *     signs  bit = 1 -> +1, 0 -> -1   (only meaningful where the mask is 1)
 *     mask   bit = 1 -> nonzero weight, 0 -> weight 0. Padding bits are 0, so they never count.
 *
 * Multiplier-free, like the 1-bit dense layer. With a +-1 input bit a (1 -> +1):
 *     dot = popcount(mask & ~(a ^ signs)) - popcount(mask & (a ^ signs))
 * The output thresholds are the same as the 1-bit layer: use symbex_dense_threshold (symbex_dense.h).
 * Exported by tools/compiler_core/export/ternary.py (method "ternary").
 */

#ifndef SYMBEX_TERNARY_H
#define SYMBEX_TERNARY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SymbexTernaryLayerData {
    uint16_t in_features;
    uint16_t out_features;
    const uint32_t* signs;      /* out_features * ceil(in_features / 32) words */
    const uint32_t* mask;       /* same size */
};

/*
 * Hidden layer: +-1 input bits (packed like symbex_dense_threshold writes them; the padding bits may hold
 * anything, the mask hides them) x ternary weights -> int32 raw score per neuron.
 */
void symbex_ternary_forward(
    const uint32_t* input_bits,
    int32_t* out,
    const struct SymbexTernaryLayerData* layer
);

/*
 * First layer: uint8 inputs (e.g. pixels) x ternary weights, through masks (no multiplications):
 *     out[j] = sum_{w = +1} x_i - sum_{w = -1} x_i
 */
void symbex_ternary_input_u8_mask(
    const uint8_t* x,
    int32_t* out,
    const struct SymbexTernaryLayerData* layer
);

#ifdef __cplusplus
}
#endif

#endif /* SYMBEX_TERNARY_H */
