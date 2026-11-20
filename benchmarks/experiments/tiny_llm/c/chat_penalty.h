#ifndef CHAT_PENALTY_H
#define CHAT_PENALTY_H
#include <stdint.h>
/* In-place additive presence penalty, once per distinct history ID, saturating.
 * Invalid inputs return -1 without modifying scores. history may be NULL iff
 * count=0. Buffers must not overlap. Penalty uses original int64 logit units. */
int chat_penalize(int64_t *scores, int n, const int *history, int count, int64_t penalty);
#endif
