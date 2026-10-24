/*
 * symbex_bitpack.h -- packs raw integer scores (like the ones
 * returned by symbex_dense_forward) into bits, so they can be
 * used as the input for the next layer.
 */

#ifndef SYMBEX_BITPACK_H
#define SYMBEX_BITPACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Packs n raw scores into bits -- bipolar STE:
 * score >= 0 -> bit 1, else -> bit 0.
 *
 * scores: n int32_t values (raw scores of a layer).
 * out_bits: output buffer, (n+31)/32 words.
 */
void symbex_bitpack_from_scores(const int32_t* scores, uint32_t* out_bits, uint16_t n);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_BITPACK_H
