/*
 * onebit_gemv.h -- 1-bit weight matrix-vector kernels for the edge-LLM probe. Portable C99, no platform headers.
 *
 * W is M rows x N columns of +-1 weights, packed LSB-first into uint32 words (bit = 1 means +1, bit = 0 means -1),
 * row-major, N a multiple of 32: row m starts at W + m * (N / 32). The output y[m] is an exact int32 dot product.
 * Every kernel returns EXACTLY the integers of gemv_ref on the same (quantized) activations.
 *
 *   kernel            activations                       what it does per weight word (32 weights)
 *   gemv_ref          int8                              unpack the bit, multiply-add (the specification)
 *   gemv_mask         int8                              sum x where the bit is set, y = 2 * S - sum(x)  (SYMBEX's
 *                                                       current first-layer approach)
 *   gemv_lut4         int8                              8 table lookups: 4 activations -> 16 precomputed sums
 *                                                       (T-MAC style); the table is built once per input vector
 *   gemv_bitserial4   4-bit, a in [0, 15], value a - 8  4 AND + 4 popcounts (one per activation bit-plane)
 *   gemv_xnor         1-bit (+-1 packed like W)         1 XOR + 1 popcount (lower bound)
 *
 * Popcount is a SWAR routine (no __builtin_popcount): the Xtensa cores of the ESP32 and ESP32-S3 have no popcount
 * instruction, and libgcc's generic routine is a slow function call.
 */
#ifndef ONEBIT_GEMV_H
#define ONEBIT_GEMV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* int16 entries of the LUT4 table for an input vector of n activations: (n / 4) groups x 16 entries */
#define ONEBIT_LUT4_ENTRIES(n) (((n) / 4) * 16)

/* uint32 words of the 4 bit-planes of an n-activation vector: plane b starts at planes + b * (n / 32) */
#define ONEBIT_BS4_WORDS(n) (4 * ((n) / 32))

/* ---- activation preparation: once per input vector, amortised over all M rows ---- */

/* LUT4: lut[g * 16 + v] = sum over j < 4 of (bit j of v ? x[4g + j] : -x[4g + j]). |entry| <= 512. */
void lut4_build(const int8_t* x, int n, int16_t* lut);

/* Bit-serial 4-bit: a[i] in [0, 15] -> planes (bit b of a[i] -> bit i % 32 of plane b, word i / 32).
   Returns sum(a), needed by gemv_bitserial4. */
int32_t bs4_pack(const uint8_t* a, int n, uint32_t* planes);

/* Offline, once per weight matrix: rowsum[m] = sum of the +-1 weights of row m = 2 * popcount(row) - n. */
void onebit_rowsums(const uint32_t* w, int m, int n, int32_t* rowsum);

/* Pack n +-1 values given as int8 (+1 / -1) into n / 32 words (+1 -> bit 1). For gemv_xnor activations. */
void xnor_pack(const int8_t* x, int n, uint32_t* bits);

/* ---- the matrix-vector products ---- */

void gemv_ref(const uint32_t* w, const int8_t* x, int32_t* y, int m, int n);

void gemv_mask(const uint32_t* w, const int8_t* x, int32_t* y, int m, int n);

/* Rows [r0, r1) only, so firmware can give each core a slice of the rows. y[r] is written for r in [r0, r1). */
void gemv_lut4_rows(const uint32_t* w, const int16_t* lut, int32_t* y, int n, int r0, int r1);
void gemv_lut4(const uint32_t* w, const int16_t* lut, int32_t* y, int m, int n);

/* y[m] = sum_i (a_i - 8) * w_i. planes and sum_a come from bs4_pack, rowsum from onebit_rowsums. */
void gemv_bitserial4(const uint32_t* w, const uint32_t* planes, int32_t sum_a, const int32_t* rowsum,
                     int32_t* y, int m, int n);

/* y[m] = n - 2 * popcount(x ^ w[m]); xbits from xnor_pack. */
void gemv_xnor(const uint32_t* w, const uint32_t* xbits, int32_t* y, int m, int n);

#ifdef __cplusplus
}
#endif

#endif
