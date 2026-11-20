#ifndef CHAT_SAMPLE_H
#define CHAT_SAMPLE_H
#include <stdint.h>
/* scores: int64 logits. temperature: units per log2 weight halving, 0 greedy.
 * k=0: all. top_p: Q16 in 1..65536, after top-k. Stable lower-ID ties.
 * Caller owns disjoint scores, state, scratch[min(k ? k : n,n)].
 * Invalid arguments return -1 without writes. Greedy preserves RNG state.
 * Non-greedy advances once and uses modulo retained mass (biased by design). */
int chat_sample(const int64_t *scores, int n, int k, uint32_t temperature,
                uint32_t top_p, uint32_t *state, int *scratch, int capacity);
#endif
