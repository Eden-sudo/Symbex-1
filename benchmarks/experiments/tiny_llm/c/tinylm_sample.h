#ifndef TINYLM_SAMPLE_H
#define TINYLM_SAMPLE_H

#include <stdint.h>

#include "tinylm_rng.h"

/* Q16 staircase: 2^(-floor(distance / shift)), not interpolation. */
static inline uint32_t tinylm_sample_exp_q(uint64_t distance, uint32_t shift)
{
    static const uint32_t table[] = {
        65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256,
        128, 64, 32, 16, 8, 4, 2, 1
    };
    uint64_t bucket;
    if (shift == 0) return 0;
    bucket = distance / shift;
    return bucket < 17 ? table[(unsigned)bucket] : 0;
}

/* Caller owns nonoverlapping scores, state and scratch; scratch has k entries. */
static inline int tinylm_sample_topk(const int64_t *scores, int vocab, int k,
                                     uint32_t shift, uint32_t *state,
                                     int *scratch_ids, int scratch_len)
{
    int count = 0, i, j, effective;
    uint64_t sum = 0, cumulative = 0, r;
    uint32_t next;
    if (!scores || !state || !scratch_ids || vocab <= 0 || k <= 0 ||
        shift == 0 || *state == 0) return -1;
    effective = k < vocab ? k : vocab;
    if (scratch_len < effective) return -1;
    for (i = 0; i < vocab; ++i) {
        int pos = 0;
        while (pos < count && scores[scratch_ids[pos]] >= scores[i]) ++pos;
        if (pos >= effective) continue;
        if (count < effective) ++count;
        for (j = count - 1; j > pos; --j) scratch_ids[j] = scratch_ids[j - 1];
        scratch_ids[pos] = i;
    }
    for (i = 0; i < count; ++i) {
        uint64_t d = (uint64_t)scores[scratch_ids[0]] -
                     (uint64_t)scores[scratch_ids[i]];
        sum += tinylm_sample_exp_q(d, shift);
    }
    next = tinylm_sample_xorshift32(*state);
    r = (uint64_t)next % sum;
    for (i = 0; i < count; ++i) {
        uint64_t d = (uint64_t)scores[scratch_ids[0]] -
                     (uint64_t)scores[scratch_ids[i]];
        cumulative += tinylm_sample_exp_q(d, shift);
        if (cumulative > r) {
            *state = next;
            return scratch_ids[i];
        }
    }
    return -1; /* Unreachable: the maximum always has positive weight. */
}
#endif
