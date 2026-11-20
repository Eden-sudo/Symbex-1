#ifndef SYMBEX_GQA_ATTENTION_H
#define SYMBEX_GQA_ATTENTION_H
#include <stdint.h>

/* Standalone reference, not the tinylm.c ABI. Head width is 64.
 * Q [heads,64], K/V [kv_heads,length,64], maxima [head] / [kv_heads,length].
 * int8 values must be -127..127; maxima 1..2^20; length 1..512.
 * exp_table is the 2048-entry uint16 intsim table (entry zero must be >0).
 * Output [heads,64] is Q12. Buffers must not overlap and must have these sizes.
 * Return 0 on success, -1 on invalid input; output is untouched on rejection.
 */
int gqa_attention(int heads, int kv_heads, int length,
                  const int8_t *q, const int32_t *qm,
                  const int8_t *k, const int32_t *km,
                  const int8_t *v, const int32_t *vm,
                  const uint16_t *exp_table, int64_t *out);
#endif
