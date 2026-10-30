/*
 * symbex_hal.h -- hardware INTERFACE that every board must implement.
 *
 * Implementations:
 *   hal/esp32s3/symbex_hal_impl.c  -- ESP32-S3 only (PIE SIMD)
 *   hal/rp2350/symbex_hal_impl.c   -- Raspberry Pi Pico 2 (RP2350): the first-layer kernels (symbex_u8_*) of its
 *                                     two core types: usada8 on the Cortex-M33 (DSP), bit planes + cpop on Hazard3.
 *                                     The bit counts come from hal/generic, which leaves its symbex_u8_* out there.
 *   hal/generic/symbex_hal_impl.c -- everything else (chosen at compile time: native popcount (x86, AArch64, RISC-V
 *                                     Zbb such as the RP2350's Hazard3), lut16 on the ESP32 and the RP2350's
 *                                     Cortex-M33, SWAR, lut8 on AVR)
 *
 * Every counting primitive has a *_rows form that computes a whole layer in one call (one result per weight
 * row): the per-call work (checks, table lookup setup, SIMD register setup) is paid once per layer instead of
 * once per neuron -- on small rows that fixed cost was more than half of the time. Same results as calling the
 * single-row form for every row.
 */

#ifndef SYMBEX_HAL_H
#define SYMBEX_HAL_H

#include <stdint.h>

/* RP2350 (PICO_RP2350 comes from the Pico SDK / Arduino-Pico build; SYMBEX_HAL_TEST_RP2350 builds it on a PC):
   its first-layer method, seen by hal/rp2350 (the kernels) and hal/generic (symbex_hal_name). USADA8 on the
   Cortex-M33 (DSP), PLANES elsewhere (Hazard3); -DSYMBEX_RP2350_U8_PLANES / _USADA8 force one. */
#if defined(PICO_RP2350) || defined(SYMBEX_HAL_TEST_RP2350)
  #define SYMBEX_HAL_RP2350 1
  #if defined(SYMBEX_RP2350_U8_PLANES)
    #define SYMBEX_RP2350_U8_NAME "first layer: 8 bit planes + counts"
  #elif defined(SYMBEX_RP2350_U8_USADA8) || defined(__ARM_FEATURE_SIMD32)
    #define SYMBEX_RP2350_U8_USADA8_ON 1
    #define SYMBEX_RP2350_U8_NAME "first layer: usada8"
  #else
    #define SYMBEX_RP2350_U8_NAME "first layer: 8 bit planes + counts"
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- XNOR arithmetic ---- */

/* XNOR + popcount over blocks of bits: returns 2 * matches - n_real_bits. */
int32_t symbex_xnor_popcount(const uint32_t* a, const uint32_t* b, int n_words, int n_real_bits);

/* The same for n_rows weight rows of n_words words each (row r starts at w + r * n_words):
   out[r] = symbex_xnor_popcount(a, w + r * n_words, n_words, n_real_bits). */
void symbex_xnor_popcount_rows(const uint32_t* a, const uint32_t* w, int n_words, int n_real_bits,
                               int n_rows, int32_t* out);

/* Popcount of ONE word, with the same method the HAL uses inside.
 * Use it instead of __builtin_popcount (a slow libgcc routine on the ESP32). */
int32_t symbex_hal_popcount32(uint32_t x);

/* Text with the active counting path (and whether lut16 got RAM). For logs and for the paper. */
const char* symbex_hal_name(void);

/* ---- AND arithmetic primitives (used by ops/symbex_and.c) ----
 * The product of two bits is an AND; the sums can be done by a SIMD multiplier on chips that have one and
 * no popcount (ESP32-S3: ee.vmulas.u8.accx). Every HAL implements them: portable C in hal/generic, the PIE
 * multiplier in hal/esp32s3 (when the buffers are 16-byte aligned and n_words is a multiple of 4;
 * otherwise C, same results). */

/* count(a AND b) over n_words 32-bit words */
int32_t symbex_and_popcount(const uint32_t* a, const uint32_t* b, int n_words);

/* the same for n_rows rows of `stride` words (row r at w + r * stride): out[r] = count(a AND row r) */
void symbex_and_popcount_rows(const uint32_t* a, const uint32_t* w, int stride, int n_rows, int32_t* out);

/* the differences, directly: out[r] = count(a XOR row r) over all `stride` words (rows of `stride` words at
   w + r * stride; bits that are 0 on both sides, like padding, add nothing). On the ESP32-S3 it runs on PIE when a
   and w are 16-byte aligned and stride is a multiple of 4. */
void symbex_xor_popcount_rows(const uint32_t* a, const uint32_t* w, int stride, int n_rows, int32_t* out);

/* 1 if counting the XOR directly is this chip's fastest way to get the differences; 0 if the AND form (count of a AND
   w, corrected with stored popcounts) is faster. Measured with tools/count_bench.py: XOR with PIE (ESP32-S3) and with
   a popcount instruction or SWAR; AND with the 64 KB table (ESP32: its results reuse fewer table entries). */
int symbex_hal_prefers_xor(void);

/* The same two counts over only the first n_words words of every row (rows still `stride` words apart): a window
   padded to a multiple of 4 words for the PIE loads costs nothing extra where there is no PIE. On the ESP32-S3 (PIE)
   the count runs over n_words rounded up to a multiple of 4: the words between must be 0 in BOTH a and the rows
   (layers/conv/symbex_bool pads windows and weights with zeros). */
void symbex_and_popcount_rows_n(const uint32_t* a, const uint32_t* w, int n_words, int stride, int n_rows, int32_t* out);
void symbex_xor_popcount_rows_n(const uint32_t* a, const uint32_t* w, int n_words, int stride, int n_rows, int32_t* out);

/* writes n_words * 32 bytes to workspace (16-byte aligned): the n_inputs inputs in the order
   symbex_u8_sum_bits needs, zero padded */
void symbex_u8_prepare(const uint8_t* x, int n_inputs, uint8_t* workspace, int n_words);

/* sum of the prepared inputs whose weight bit (in w, n_words words) is 1 */
int32_t symbex_u8_sum_bits(const uint8_t* workspace, const uint32_t* w, int n_words);

/* the same for n_rows rows of `stride` words: out[r] = symbex_u8_sum_bits(workspace, w + r * stride, stride).
   n_inputs: the real inputs (<= stride * 32); the padding past them is not visited where the layout allows it. */
void symbex_u8_sum_bits_rows(const uint8_t* workspace, int n_inputs, const uint32_t* w, int stride, int n_rows,
                             int32_t* out);

#if defined(CONFIG_IDF_TARGET_ESP32S3)
/* ESP32-S3 only (PIE): out[r] = sum_{i < n} x[i] * w[r * stride + i]. x: 16-byte aligned, zero padded to a multiple
   of 16 bytes; the rows any alignment (read in whole 16-byte blocks, up to 31 bytes past each row's end). */
void symbex_s8_dot_rows(const int8_t* x, int n, const int8_t* w, int stride, int n_rows, int32_t* out);
#endif

/* Weighted AND counts over bit planes (first layer with the input split into k bit planes):
   out[r] = sum_{q < k} count(planes[q * n_words ...] AND row r) << (low + q), rows of n_words words.
   Each weight word is read once for the k planes. */
void symbex_and_planes_rows(const uint32_t* planes, int k, int low, const uint32_t* w, int n_words,
                            int n_rows, int32_t* out);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_HAL_H
