#include "ops/symbex_bitpack.h"

void symbex_bitpack_from_scores(const int32_t* scores, uint32_t* out_bits, uint16_t n) {
    uint16_t words = (n + 31) / 32;
    for (uint16_t w = 0; w < words; w++) out_bits[w] = 0;

    for (uint16_t i = 0; i < n; i++) {
        if (scores[i] >= 0) {
            out_bits[i / 32] |= (1UL << (i % 32));
        }
    }
}
