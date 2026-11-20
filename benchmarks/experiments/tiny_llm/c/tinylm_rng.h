#ifndef TINYLM_RNG_H
#define TINYLM_RNG_H
#include <stdint.h>

/* Marsaglia xorshift32 (13,17,5), modulo 2^32 after each left shift.
 * State is caller-owned, seed range 1..UINT32_MAX; zero is rejected.
 * A failed checked advance leaves state untouched. No global RNG state. */
static inline uint32_t tinylm_sample_xorshift32(uint32_t state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
static inline int tinylm_rng_next(uint32_t *state)
{
    if (!state || !*state) return -1;
    *state = tinylm_sample_xorshift32(*state);
    return 0;
}
#endif
