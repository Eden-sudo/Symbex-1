/*
 * symbex_swar.h -- first-layer sums with SWAR (SIMD Within A Register): the generic method for any 32-bit chip
 * without a kernel of its own (ESP32, PC, ARM, RISC-V). Header-only (static inline), so every caller gets it
 * inlined in its own loop.
 *
 * What it computes: the sum of the uint8 inputs whose weight bit is 1, for one weight row:
 *     sum_{i < n_in, bit i of w = 1} x[i]
 * How: 4 inputs per 32-bit operation. 4 weight bits pick a 32-bit mask from a 16-entry table (0xFF in the bytes
 * whose bit is 1), the 4 input bytes are read as one word and ANDed with it, and the result is accumulated in two
 * registers of 16-bit lanes (bytes 0 and 2 / bytes 1 and 3), so 4 sums advance at once. A 16-bit lane takes 257
 * additions of 255 before it could overflow; the lanes are emptied every 16 weight words (128 additions).
 * About 8 operations per 4 inputs, against about 3 per input with one mask per input. Exact: the same integer.
 *
 * The 4 inputs are read as ONE aligned 32-bit load: x must be 4-byte aligned (image buffers and the workspace of
 * symbex_u8_prepare are). If it is not, the plain loop is used: an unaligned 32-bit read is 4 byte loads on Xtensa
 * and made this method slower than the plain loop (measured on the ESP32 and the S3).
 * Little-endian chips only (byte k of the loaded word = input 4g + k); on a big-endian target the plain loop is used.
 * Weights are read with SYMBEX_READ_W32 (flash on AVR; AVR itself has its own 8-bit path and does not use this).
 */
#ifndef SYMBEX_SWAR_H
#define SYMBEX_SWAR_H

#include <stdint.h>
#include <string.h>
#include "core/symbex_core.h"

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  #define SYMBEX_SWAR_LE 1
#endif

/* a 32-bit word that may alias the uint8 inputs (no strict-aliasing assumption) */
typedef uint32_t __attribute__((may_alias)) symbex_swar_word_t;

/* byte masks for 4 weight bits: bit k set -> byte k = 0xFF */
static const uint32_t symbex_swar_spread4[16] = {
    0x00000000u, 0x000000FFu, 0x0000FF00u, 0x0000FFFFu, 0x00FF0000u, 0x00FF00FFu, 0x00FFFF00u, 0x00FFFFFFu,
    0xFF000000u, 0xFF0000FFu, 0xFF00FF00u, 0xFF00FFFFu, 0xFFFF0000u, 0xFFFF00FFu, 0xFFFFFF00u, 0xFFFFFFFFu};

/* the plain method: one input at a time, through a 0 / -1 mask from its weight bit (no branches) */
static inline int32_t symbex_scalar_u8_masked_sum(const uint8_t* x, const uint32_t* w, uint16_t n_in) {
    int32_t total = 0;
    const uint16_t in_words = (uint16_t)((n_in + 31) / 32);
    for (uint16_t p = 0; p < in_words; p++) {
        const uint32_t wv = SYMBEX_READ_W32(&w[p]);
        const uint8_t* xp = &x[(uint32_t)p * 32];
        const uint16_t lim = (uint16_t)((n_in - p * 32) < 32 ? (n_in - p * 32) : 32);
        for (uint16_t b = 0; b < lim; b++) total += xp[b] & -(int32_t)((wv >> b) & 1u);
    }
    return total;
}

/* sum of x[i] for i < n_in whose bit is set in the weight row w (ceil(n_in / 32) words). x must have n_in bytes;
   4-byte aligned for the SWAR path (otherwise the plain method). */
static inline int32_t symbex_swar_u8_masked_sum(const uint8_t* x, const uint32_t* w, uint16_t n_in) {
#if !defined(SYMBEX_SWAR_LE)
    return symbex_scalar_u8_masked_sum(x, w, n_in);
#else
    if (((uintptr_t)x & 3u) != 0) return symbex_scalar_u8_masked_sum(x, w, n_in);
    int32_t total = 0;
    const uint16_t full_words = n_in / 32;
    uint32_t lo = 0, hi = 0;                       /* 16-bit lanes: bytes 0,2 in lo; bytes 1,3 in hi */
    uint16_t pending = 0;
    for (uint16_t p = 0; p < full_words; p++) {
        uint32_t wv = SYMBEX_READ_W32(&w[p]);
        const symbex_swar_word_t* xp = (const symbex_swar_word_t*)&x[(uint32_t)p * 32];   /* aligned: checked above */
        for (uint8_t g = 0; g < 8; g++, wv >>= 4) {
            const uint32_t v = xp[g] & symbex_swar_spread4[wv & 15u];
            lo += v & 0x00FF00FFu;
            hi += (v >> 8) & 0x00FF00FFu;
        }
        if (++pending == 16) {                     /* 16 words x 8 additions = 128 per lane: empty the lanes */
            total += (int32_t)((lo & 0xFFFFu) + (lo >> 16) + (hi & 0xFFFFu) + (hi >> 16));
            lo = hi = 0;
            pending = 0;
        }
    }
    total += (int32_t)((lo & 0xFFFFu) + (lo >> 16) + (hi & 0xFFFFu) + (hi >> 16));
    const uint16_t rest = n_in % 32;               /* last partial word: one input at a time, never past n_in */
    if (rest) {
        const uint32_t wv = SYMBEX_READ_W32(&w[full_words]);
        const uint8_t* xp = &x[(uint32_t)full_words * 32];
        for (uint8_t b = 0; b < rest; b++) total += xp[b] & -(int32_t)((wv >> b) & 1u);
    }
    return total;
#endif
}

#endif /* SYMBEX_SWAR_H */
