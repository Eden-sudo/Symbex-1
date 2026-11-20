#ifndef SYMBEX_GQA_ROLLING_H
#define SYMBEX_GQA_ROLLING_H
#include "gqa_attention.h"

/* Initialize the struct once with position=0 and caller-owned buffers:
 * k,v [kv_heads,window,64] int8; km,vm [kv_heads,window] int32.
 * Fields other than position are immutable; do not share state across streams.
 * cos/sin [max_positions,32] are exported Q14 tables at ABSOLUTE positions.
 * Q [heads,64], K/V [kv_heads,64] are Q12 int32, abs <= 2^19.
 * Output [heads,64] int64. All arrays must be sized correctly and disjoint.
 * This host reference allocates chronological scratch per step, not firmware-ready.
 * Return 0 success, -1 invalid input, -2 allocation failure. On failure, state
 * and output are unchanged. Reset position=0 to begin a new stream.
 */
typedef struct {
    int heads, kv_heads, window, max_positions;
    uint64_t position;
    int8_t *k, *v;
    int32_t *km, *vm;
    const int32_t *cos, *sin;
} gqa_ring;

int gqa_ring_step(gqa_ring *ring, const int32_t *q, const int32_t *k,
                  const int32_t *v, const uint16_t *exp_table, int64_t *out);
#endif
