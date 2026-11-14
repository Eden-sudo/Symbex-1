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

/* SYMBEX_FAST_CODE: marks the inner loops of an engine (today the vertical one, with the int8 stem and the pooled
   classifier it calls). With -DSYMBEX_CODE_IN_RAM they run from RAM instead of flash: the RP2350 (Pico SDK
   .time_critical sections) and the ESP32 boards (IRAM). Code read from flash goes through a small cache, so where a
   loop lands in flash changes its speed (measured on the Pico 2's Hazard3: the same engine at 60 or 70 ms after an
   unrelated edit); in RAM the time no longer depends on it. Costs the RAM of that code. RP2350 host tests use the
   section only for ELF objects; on other hosts, elsewhere, or without the flag, it is empty. */
#if defined(SYMBEX_CODE_IN_RAM) && (defined(PICO_RP2350) || (defined(SYMBEX_HAL_TEST_RP2350) && defined(__ELF__)))
  #define SYMBEX_FAST_CODE __attribute__((section(".time_critical.symbex")))
#elif defined(SYMBEX_CODE_IN_RAM) && defined(ESP_PLATFORM)
  #define SYMBEX_FAST_CODE __attribute__((section(".iram1.symbex")))
#else
  #define SYMBEX_FAST_CODE
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
/* ESP32-S3 only (PIE): out[r] = sum_{i < n} x[i] * w[r * stride + i], n >= 1. x: 16-byte aligned, zero padded to a
   multiple of 16 bytes; the rows any alignment (read in aligned 16-byte blocks, only those holding bytes of the row). */
void symbex_s8_dot_rows(const int8_t* x, int n, const int8_t* w, int stride, int n_rows, int32_t* out);

/* ESP32-S3 only (PIE): a whole 3 x 3 int8 stem at once (symbex_bool_stem_i8's layer, stride 1 or 2, padding 1), 16
   output channels per instruction: out bit o of pixel (oy, ox) = sum over the window of (pixel - 128) * w[o] >= thr[o],
   out HWC as symbex_bool_stem_i8 writes it. 0 = done; -1 = not this kernel's, nothing written (another kernel size,
   in_w > SYMBEX_S8_STEM3_MAX_W, or a channel whose sums could leave QACC's 20 bits: |thr| + 128 * sum |w| > 524287;
   channels whose bit is constant are set directly and never refuse). Stack: 3 padded image rows + ~0.5 KB. */
#define SYMBEX_S8_STEM3_MAX_W 128
int symbex_s8_stem3_bits(const uint8_t* img, int in_h, int in_w, int stride, int out_h, int out_w,
                         const int8_t* weights, int row_bytes, const int32_t* thresholds, int out_ch, uint32_t* out);
/* the same layer at stride 1 with its bits straight in channel planes (the vertical engine's layout, 16 pixels per
   instruction): bit p of plane c (planes + c * plane_stride, little-endian words) = pixel p = y * in_w + x of
   channel c; out_h = in_h, out_w = in_w. 0 = done; -1 = nothing written (in_w not a multiple of 16 or past
   SYMBEX_S8_STEM3_MAX_W, or the 20-bit check above). Stack: 3 image rows of 9 * in_w bytes + ~2 KB. */
int symbex_s8_stem3_planes(const uint8_t* img, int in_h, int in_w, const int8_t* weights, int row_bytes,
                           const int32_t* thresholds, int out_ch, uint32_t* planes, int plane_stride);
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
