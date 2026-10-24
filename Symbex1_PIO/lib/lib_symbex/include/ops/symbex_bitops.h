/*
 * symbex_bitops.h -- low-level, generic bitwise operations,
 * used by ANY spatial layer (conv1d, conv2d, pool...).
 *
 * Migrated from symbex_conv.cpp (the "El Mesero" and "El Ayudante"
 * functions) -- preserved because it handles the GENERAL case of
 * arbitrary image widths (unlike symbex_extract_window in conv2d,
 * which assumes a row fits into a single 32-bit word).
 *
 * DESIGN NOTE: unlike the original version, load_line_buffer NO LONGER
 * takes a full layer struct -- it takes discrete parameters so that
 * ops/ does not depend on any specific layer type.
 */

#ifndef SYMBEX_BITOPS_H
#define SYMBEX_BITOPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extracts an N-bit range (up to 32) starting at bit_start,
 * even if that range CROSSES a 32-bit word boundary.
 *
 * plane: source bit block.
 * bit_start: starting position, in bits, within plane.
 * n_bits: how many bits to extract (1 to 32).
 * Returns: the extracted n_bits, right-aligned.
 */
uint32_t symbex_extract_bits(const uint32_t* plane, uint32_t bit_start, uint8_t n_bits);

/*
 * Inserts an N-bit value into plane, starting at bit_start_local.
 * CLEARS those bits first (AND with inverted mask), then sets them
 * (OR). This works regardless of the plane's initial value (it
 * previously assumed an all-zero start; with patch_buffer starting
 * at 0xFFFFFFFF this was an actual bug, now fixed).
 * Also handles crossing a word boundary.
 *
 * plane: destination bit block (modified in place).
 * bit_start_local: starting position, in bits, within plane.
 * n_bits: how many bits of the value are valid (1 to 32).
 * value: the bits to insert, right-aligned.
 */
void symbex_insert_bits(uint32_t* plane, uint32_t bit_start_local, uint8_t n_bits, uint32_t value);

/*
 * Counts how many bits are 1 in both a and b (popcount of a AND b),
 * over full n_words. Used to correct the padding popcount: counts
 * how many weights are 1 exactly in the positions marked as padding
 * by a mask.
 */
int32_t symbex_popcount_and(const uint32_t* a, const uint32_t* b, uint16_t n_words);

/*
 * Loads a workspace buffer (SRAM) with the 'kernel' rows needed
 * to process the output row 'oh', copying from a larger source
 * (e.g., the full binarized image). Zero-fills rows that fall
 * out of bounds (convolution padding).
 *
 * source_input: source bit block (the full image).
 * sram_buffer: destination buffer, must have room for
 *   in_ch * kernel * ((in_w+31)/32) words.
 * in_ch, in_h, in_w: input image dimensions.
 * kernel, stride, padding: convolution parameters.
 * oh: OUTPUT row to process (not input).
 */
void symbex_load_line_buffer(
    const uint32_t* source_input,
    uint32_t* sram_buffer,
    uint16_t in_ch, uint16_t in_h, uint16_t in_w,
    uint16_t kernel, uint16_t stride, uint16_t padding,
    int16_t oh
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_BITOPS_H
