#ifndef TINYLM_FACTOR_H
#define TINYLM_FACTOR_H

#include <stdint.h>
#include <stddef.h>
#include <limits.h>

/* Rows are byte padded, with bit 0 first; 1 means +1 and 0 means -1.
 * The caller supplies x[n_in], v[rank*ceil(n_in/8)],
 * u[n_out*ceil(rank/8)], m1[rank], m2[n_out], z[rank], y[n_out].
 * Typed arrays must be naturally aligned. Writable arrays must not overlap
 * each other or any input; read-only arrays may overlap each other. */
static inline int8_t tinylm_factor_requantize(int32_t acc, int32_t multiplier,
                                              int shift)
{
    int64_t p = (int64_t)acc * multiplier;
    uint64_t a = (uint64_t)(p < 0 ? -p : p);
    uint64_t q;
    if (shift == 0) {
        q = a;
    } else {
        uint64_t half = UINT64_C(1) << (shift - 1);
        uint64_t mask = (UINT64_C(1) << shift) - 1;
        q = (a >> shift) + ((a & mask) >= half);
    }
    if (q > 127) q = 127;
    return (int8_t)(p < 0 ? -(int)q : (int)q);
}

static inline int tinylm_factor(const int8_t *x, const uint8_t *v,
                                const uint8_t *u, const int32_t *m1,
                                const int32_t *m2, int n_in, int rank,
                                int n_out, int sh1, int8_t *z, int64_t *y)
{
    size_t vs, us, r, o, i;
    if (!x || !v || !u || !m1 || !m2 || !z || !y ||
        n_in <= 0 || rank <= 0 || n_out <= 0 ||
        n_in > INT32_MAX / 128 || rank > INT32_MAX / 127 ||
        sh1 < 0 || sh1 > 63) return -1;
    vs = (size_t)n_in / 8 + ((size_t)n_in % 8 != 0);
    us = (size_t)rank / 8 + ((size_t)rank % 8 != 0);
    if ((size_t)rank > SIZE_MAX / vs || (size_t)n_out > SIZE_MAX / us ||
        (size_t)rank > SIZE_MAX / sizeof(*m1) ||
        (size_t)n_out > SIZE_MAX / sizeof(*m2) ||
        (size_t)n_out > SIZE_MAX / sizeof(*y)) return -1;
    for (r = 0; r < (size_t)rank; ++r) {
        int32_t acc = 0;
        for (i = 0; i < (size_t)n_in; ++i) {
            int32_t value = x[i];
            acc += ((v[r * vs + i / 8] >> (i % 8)) & 1) ? value : -value;
        }
        z[r] = tinylm_factor_requantize(acc, m1[r], sh1);
    }
    for (o = 0; o < (size_t)n_out; ++o) {
        int32_t acc = 0;
        for (r = 0; r < (size_t)rank; ++r) {
            int32_t value = z[r];
            acc += ((u[o * us + r / 8] >> (r % 8)) & 1) ? value : -value;
        }
        y[o] = (int64_t)acc * m2[o];
    }
    return 0;
}
#endif
