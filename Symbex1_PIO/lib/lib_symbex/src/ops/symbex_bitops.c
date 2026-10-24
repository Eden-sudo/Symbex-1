/*
 * symbex_bitops.c -- Bitwise operations implementation.
 *
 * The logic is identical to the original symbex_conv.cpp, but uses explicit
 * parameters instead of a layer struct, and relies on SYMBEX_READ_W32
 * from core/symbex_core.h (AVR/PROGMEM portability is handled there).
 */

#include "ops/symbex_bitops.h"
#include "hal/symbex_hal.h"
#include "core/symbex_core.h"
#include <string.h>

uint32_t symbex_extract_bits(const uint32_t* plane, uint32_t bit_start, uint8_t n_bits) {
    uint32_t word_idx = bit_start / 32;
    uint8_t bit_offset = bit_start % 32;

    if (bit_offset + n_bits <= 32) {
        return (SYMBEX_READ_W32(&plane[word_idx]) >> bit_offset) & ((n_bits == 32) ? 0xFFFFFFFFUL : ((1UL << n_bits) - 1));
    } else {
        uint32_t part_low = SYMBEX_READ_W32(&plane[word_idx]) >> bit_offset;
        uint8_t bits_high = n_bits - (32 - bit_offset);
        uint32_t part_high = SYMBEX_READ_W32(&plane[word_idx + 1]) & ((1UL << bits_high) - 1);
        return part_low | (part_high << (32 - bit_offset));
    }
}

void symbex_insert_bits(uint32_t* plane, uint32_t bit_start_local, uint8_t n_bits, uint32_t value) {
    uint32_t word_idx = bit_start_local / 32;
    uint8_t bit_offset = bit_start_local % 32;
    uint32_t mask_low = (n_bits >= (uint8_t)(32 - bit_offset))
        ? (0xFFFFFFFFUL << bit_offset)
        : (((1UL << n_bits) - 1) << bit_offset);

    // Clear first (AND with inverted mask), THEN insert (OR).
    // This works regardless of the buffer's initial value (previously it assumed an all-zero start, which
    // caused a total no-op if patch_buffer started with 0xFFFFFFFF).
    // The inserted bits are masked too: a value with bits above n_bits can no longer overwrite the neighbors.
    plane[word_idx] = (plane[word_idx] & ~mask_low) | ((value << bit_offset) & mask_low);

    if (bit_offset + n_bits > 32) {
        uint8_t bits_high = n_bits - (32 - bit_offset);
        uint32_t mask_high = (1UL << bits_high) - 1;
        plane[word_idx + 1] = (plane[word_idx + 1] & ~mask_high) | ((value >> (32 - bit_offset)) & mask_high);
    }
}

/* count(a AND b). Delegates to the HAL (symbex_and_popcount), so it gets the SIMD path of the chip when there
   is one (ESP32-S3 PIE, with 16-byte aligned buffers and n_words a multiple of 4) and portable C otherwise. */
int32_t symbex_popcount_and(const uint32_t* a, const uint32_t* b, uint16_t n_words) {
    return symbex_and_popcount(a, b, (int)n_words);
}

void symbex_load_line_buffer(
    const uint32_t* source_input,
    uint32_t* sram_buffer,
    uint16_t in_ch, uint16_t in_h, uint16_t in_w,
    uint16_t kernel, uint16_t stride, uint16_t padding,
    int16_t oh
) {
    int16_t ih_start = oh * stride - padding;
    uint16_t words_per_row = (in_w + 31) / 32;

    for (uint16_t ic = 0; ic < in_ch; ic++) {
        for (uint16_t kh = 0; kh < kernel; kh++) {
            int16_t ih = ih_start + kh;
            uint32_t sram_word_idx = (ic * kernel * words_per_row) + (kh * words_per_row);

            if (ih >= 0 && ih < in_h) {
                uint32_t src_word_idx = (ic * in_h * words_per_row) + (ih * words_per_row);
                memcpy(&sram_buffer[sram_word_idx], &source_input[src_word_idx], words_per_row * sizeof(uint32_t));
            } else {
                // Zero-fill for convolution padding
                memset(&sram_buffer[sram_word_idx], 0, words_per_row * sizeof(uint32_t));
            }
        }
    }
}
