/*
 * symbex_bitpack.c -- Bitpacking utilities for binary activations.
 */

#include "ops/symbex_bitpack.h"

/**
 * Packs an array of 32-bit integer scores into a dense bit array.
 * 
 * This acts as the binarized activation function (threshold at 0):
 * - If score >= 0, the corresponding output bit is set to 1.
 * - If score < 0,  the corresponding output bit remains 0.
 * 
 * @param scores   Array of input layer scores (accumulators).
 * @param out_bits Output buffer where the packed bits will be stored.
 * @param n        Number of scores to process.
 */
void symbex_bitpack_from_scores(const int32_t* scores, uint32_t* out_bits, uint16_t n) {
    uint16_t words = (n + 31) / 32;
    
    // Initialize the output buffer with zeros
    for (uint16_t w = 0; w < words; w++) {
        out_bits[w] = 0;
    }

    // Apply threshold and pack into bits
    for (uint16_t i = 0; i < n; i++) {
        if (scores[i] >= 0) {
            out_bits[i / 32] |= (1UL << (i % 32));
        }
    }
}
