#ifndef TINYLM_FACTOR_POPCOUNT_H
#define TINYLM_FACTOR_POPCOUNT_H
#include "tinylm_factor.h"

/* Portable byte SWAR, no compiler intrinsics. */
static inline unsigned littlebit_pop8(unsigned x)
{
    x -= (x >> 1) & 0x55u;
    x = (x & 0x33u) + ((x >> 2) & 0x33u);
    return (x + (x >> 4)) & 0x0fu;
}

/* Same contract/output as tinylm_factor. Additional nonoverlapping workspace:
 * planes[8*ceil(rank/8)] (sign plane, then seven magnitude planes).
 * Latents are cached once. XNOR matches signs; magnitude masks retain int8
 * values exactly, unlike a single binary activation popcount.
 * This is a performance candidate, not a measured target-board speedup. */
static inline int tinylm_factor_popcount(const int8_t *x, const uint8_t *v,
    const uint8_t *u, const int32_t *m1, const int32_t *m2,
    int n_in, int rank, int n_out, int sh1, int8_t *z, int64_t *y,
    uint8_t *planes, size_t planes_size)
{
    size_t vs, us, r, o, i, b;
    if (!x || !v || !u || !m1 || !m2 || !z || !y || !planes ||
        n_in <= 0 || rank <= 0 || n_out <= 0 ||
        n_in > INT32_MAX / 128 || rank > INT32_MAX / 127 ||
        sh1 < 0 || sh1 > 63) return -1;
    vs = (size_t)n_in / 8 + ((size_t)n_in % 8 != 0);
    us = (size_t)rank / 8 + ((size_t)rank % 8 != 0);
    if (us > SIZE_MAX / 8 || planes_size < 8 * us ||
        (size_t)rank > SIZE_MAX / vs || (size_t)n_out > SIZE_MAX / us ||
        (size_t)rank > SIZE_MAX / sizeof(*m1) ||
        (size_t)n_out > SIZE_MAX / sizeof(*m2) ||
        (size_t)n_out > SIZE_MAX / sizeof(*y)) return -1;
    for (i = 0; i < 8 * us; ++i) planes[i] = 0;
    for (r = 0; r < (size_t)rank; ++r) {
        int32_t acc = 0;
        unsigned magnitude, bit = 1u << (r % 8);
        for (i = 0; i < (size_t)n_in; ++i) {
            int32_t value = x[i];
            acc += ((v[r * vs + i / 8] >> (i % 8)) & 1) ? value : -value;
        }
        z[r] = tinylm_factor_requantize(acc, m1[r], sh1);
        if (z[r] >= 0) planes[r / 8] |= (uint8_t)bit;
        magnitude = (unsigned)(z[r] < 0 ? -(int)z[r] : z[r]);
        for (b = 0; b < 7; ++b)
            if ((magnitude >> b) & 1u) planes[(b + 1) * us + r / 8] |= (uint8_t)bit;
    }
    for (o = 0; o < (size_t)n_out; ++o) {
        int32_t acc = 0;
        for (i = 0; i < us; ++i) {
            unsigned matches = (unsigned)(~(u[o * us + i] ^ planes[i])) & 255u;
            for (b = 0; b < 7; ++b) {
                unsigned mask = planes[(b + 1) * us + i];
                int dot = 2 * (int)littlebit_pop8(matches & mask) - (int)littlebit_pop8(mask);
                acc += dot * (1 << b);
            }
        }
        y[o] = (int64_t)acc * m2[o];
    }
    return 0;
}
#endif
