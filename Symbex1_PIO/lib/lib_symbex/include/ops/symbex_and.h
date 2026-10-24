/*
 * symbex_and.h -- AND arithmetic for binary layers (portable). The product of two bits is an AND; the counts
 * come from the HAL (hal/symbex_hal.h): portable C everywhere, and on chips with SIMD multiply-accumulate
 * (ESP32-S3 PIE) the HAL does them with the multiplier.
 *
 * The trained network is +-1 (bit 1 = +1, bit 0 = -1). Its dot products, EXACTLY, with AND:
 *     sum_i a_i w_i (+-1)  =  4 * count(a AND w) - 2 * count(a) - 2 * count(w) + n
 *     first layer (uint8 x 1 bit):  sum_i x_i w_i (+-1)  =  2 * sum_{w_i = +1} x_i - sum_i x_i
 * count(w) is fixed per neuron (w_count, written by the exporter). The raw scores are the same, bit for bit,
 * as the XNOR / popcount path: thresholds and classifier do not change.
 *
 * Layout: weight rows of `stride` words, padding bits 0. The input buffer `a` must have `stride` readable
 * words (bits past n_in may hold anything). For the SIMD path of a HAL: stride a multiple of 4 and buffers
 * aligned to 16 bytes (otherwise the HAL falls back to C, same results).
 */
#ifndef SYMBEX_AND_H
#define SYMBEX_AND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Binary layer: raw[j] = +-1 dot product of a (n_in bits) and row j of w. w_count[j] = 1-bits of row j. */
void symbex_and_raw(const uint32_t* a, uint16_t n_in,
                    const uint32_t* w, uint16_t stride, uint16_t n_out,
                    const int32_t* w_count, int32_t* raw);

/* First layer: raw[j] = sum_i x_i * (+-1 weight). work: stride * 32 bytes, 16-byte aligned (the HAL
   prepares the inputs there once per call, in the order its kernel needs). */
void symbex_and_input_u8(const uint8_t* x, uint16_t n_in,
                         const uint32_t* w, uint16_t stride, uint16_t n_out,
                         uint8_t* work, int32_t* raw);

#ifdef __cplusplus
}
#endif

#endif /* SYMBEX_AND_H */
