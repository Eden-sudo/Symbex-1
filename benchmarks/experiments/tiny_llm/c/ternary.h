#ifndef TINY_LLM_TERNARY_H
#define TINY_LLM_TERNARY_H
#include <stddef.h>
#include <stdint.h>

/* Row-local base-3 payload from ternary_pack.py: five low-first digits per
 * byte, weight = digit - 1, zero unused digits. Not GGML TQ1_0.
 * Returns 0 on success, -1 on invalid dimensions/length/codes/padding/pointers.
 * On error output is unchanged. Caller supplies rows int64 outputs and cols
 * int8 inputs, with no overlap between output and either input allocation.
 * No scales/bias, no heap allocation; full [-128,127] activation range. */
int ternary_matvec_base3(const uint8_t *packed, size_t packed_len,
                        size_t rows, size_t cols, const int8_t *x, int64_t *out);
#endif
