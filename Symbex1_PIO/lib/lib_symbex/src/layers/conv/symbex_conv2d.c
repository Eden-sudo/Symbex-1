/*
 * symbex_conv2d.c -- HWC FORMAT IN BITS.
 *
 * Binary maps store the channels of each pixel together:
 *   pixel (y, x) starts at word (y * W + x) * PW,  PW = ceil(C / 32)
 *   channel c is bit (c % 32) of word (c / 32) of that pixel
 *
 * The extracted patch is ordered as [kh][kw][c]: local bit = (kh*K + kw)*C + c
 * -> weights must be exported in that exact order (plain.py, transpose).
 *
 * Building the window now involves K*K copies of C bits (one per pixel), 
 * read straight from the map -- no line_buffer, no bit-by-bit extract/insert.
 *
 * Padding logic is identical to before: bit=0 in the patch and bit=1 in the 
 * mask, with the exact same correction applied in symbex_conv2d_process_position.
 *
 * DEBUG: compile with -DSYMBEX_DEBUG_STEM_POS to print the raw score of the stem
 * (oc=0, oh=5, ow=5). It is no longer active by default.
 */

#include <stdio.h>
#include <string.h>
#include "layers/conv/symbex_conv2d.h"
#include "hal/symbex_hal.h"
#include "ops/symbex_bitops.h"
#include <stdbool.h>

// OR `value` (up to 32 bits, already masked) into `dst` starting at bit `off`.
static inline void or_bits(uint32_t* dst, uint32_t off, uint32_t value) {
    uint32_t w = off >> 5;
    uint32_t s = off & 31;
    dst[w] |= value << s;
    if (s != 0) {
        uint32_t high = value >> (32 - s);
        if (high != 0) dst[w + 1] |= high;
    }
}

static inline uint32_t mask_n(uint32_t n) {
    return (n >= 32) ? 0xFFFFFFFFUL : ((1UL << n) - 1);
}

// Copies the first n_bits from src (word-aligned) to dst starting at bit off.
static inline void put_bits(uint32_t* dst, uint32_t off, const uint32_t* src, uint32_t n_bits) {
    for (uint32_t i = 0; n_bits > 0; i++) {
        uint32_t n = (n_bits > 32) ? 32 : n_bits;
        or_bits(dst, off, src[i] & mask_n(n));
        off += n;
        n_bits -= n;
    }
}

// Sets n_bits to 1 in dst starting at bit off.
static inline void put_ones(uint32_t* dst, uint32_t off, uint32_t n_bits) {
    while (n_bits > 0) {
        uint32_t n = (n_bits > 32) ? 32 : n_bits;
        or_bits(dst, off, mask_n(n));
        off += n;
        n_bits -= n;
    }
}

void symbex_conv2d_process_position(
    const uint32_t* patch_buffer,
    const uint32_t* padding_mask,
    const int32_t* identity_q_act,
    int32_t* out_values_q_act,
    const struct SymbexConv2DLayer* layer)
{
    uint16_t window_elems = layer->in_ch * layer->kernel * layer->kernel;
    uint16_t patch_words = (((window_elems + 31) / 32) + 3) & ~3;

    int32_t n_padding = 0;
    for (uint16_t i = 0; i < patch_words; i++) {
        n_padding += symbex_hal_popcount32(padding_mask[i]);
    }

    #ifdef SYMBEX_DEBUG_STEM_POS
    static int pos_counter = 0;
    bool is_target_pos = (layer->in_ch == 24 && pos_counter == 165);  // oh=5, ow=5 (5*32+5)
    if (layer->in_ch == 24) pos_counter++;
    #endif

    const uint32_t* w_ptr = layer->weights;

    for (uint16_t oc = 0; oc < layer->out_ch; oc++) {
        int32_t raw_measured = symbex_xnor_popcount(patch_buffer, w_ptr, patch_words, window_elems);

        int32_t raw;
        if (n_padding == 0) {
            raw = raw_measured;
        } else {
            int32_t padding_matches = symbex_popcount_and(padding_mask, w_ptr, patch_words);
            raw = raw_measured - n_padding + 2 * padding_matches;
        }

        w_ptr += patch_words;

        #ifdef SYMBEX_DEBUG_STEM_POS
        if (is_target_pos && oc == 0) {
            printf("    [debug] stem oc=0 oh=5 ow=5 raw=%ld\n", (long)raw);
        }
        #endif

        out_values_q_act[oc] = symbex_dsp_apply_layer(raw, identity_q_act[oc], oc, &layer->dsp);
    }
}

// int32 CHW -> bits HWC. Reads sequentially (channel by channel);
// writes hit a small map that lives in cache.
void symbex_dsp_binarize_map(
    const int32_t* values_q_act, uint32_t* out_bits,
    const int32_t* beta_rsign_q_act,
    uint16_t ch, uint16_t h, uint16_t w)
{
    uint16_t pw = (ch + 31) / 32;
    uint32_t hw = (uint32_t)h * w;
    memset(out_bits, 0, hw * pw * sizeof(uint32_t));
    for (uint16_t c = 0; c < ch; c++) {
        const int32_t* row = &values_q_act[(uint32_t)c * hw];
        const int32_t threshold = beta_rsign_q_act[c];
        uint32_t* dest = &out_bits[c / 32];
        const uint32_t bit = 1UL << (c % 32);
        for (uint32_t p = 0; p < hw; p++) {
            if (symbex_dsp_binarize(row[p], threshold)) dest[p * pw] |= bit;
        }
    }
}

// float CHW -> bits HWC (threshold 0).
void symbex_binarize_image(const float* input_map, uint32_t* out_bits,
                           uint16_t ch, uint16_t h, uint16_t w) {
    uint16_t pw = (ch + 31) / 32;
    uint32_t hw = (uint32_t)h * w;

    for (uint32_t p = 0; p < hw; p++) {
        for (uint16_t i = 0; i < pw; i++) {
            uint16_t c0 = i * 32;
            uint16_t c1 = (uint16_t)((c0 + 32 < ch) ? c0 + 32 : ch);
            uint32_t word = 0;
            for (uint16_t c = c0; c < c1; c++) {
                if (input_map[c * hw + p] >= 0.0f) word |= (1UL << (c - c0));
            }
            out_bits[p * pw + i] = word;
        }
    }
}

// WARNING: the first argument is the FULL HWC MAP (before the HWC format it was a line_buffer).
void symbex_conv2d_extract_window(const uint32_t* hwc_map, uint32_t* patch_buffer, uint32_t* padding_mask,
                                  uint16_t in_ch, uint16_t in_h, uint16_t in_w, uint16_t kernel,
                                  uint16_t stride, uint16_t padding, uint16_t oh, uint16_t ow) {
    uint16_t window_elems = in_ch * kernel * kernel;
    uint16_t patch_words = (((window_elems + 31) / 32) + 3) & ~3;
    uint16_t pw = (in_ch + 31) / 32;
    int16_t ih0 = (int16_t)(oh * stride) - (int16_t)padding;
    int16_t iw0 = (int16_t)(ow * stride) - (int16_t)padding;

    memset(patch_buffer, 0, patch_words * sizeof(uint32_t));
    memset(padding_mask, 0, patch_words * sizeof(uint32_t));

    uint32_t off = 0;
    for (uint16_t kh = 0; kh < kernel; kh++) {
        int16_t ih = ih0 + kh;
        bool valid_row = (ih >= 0 && ih < in_h);
        for (uint16_t kw = 0; kw < kernel; kw++) {
            int16_t iw = iw0 + kw;
            if (valid_row && iw >= 0 && iw < in_w) {
                put_bits(patch_buffer, off, &hwc_map[((uint32_t)ih * in_w + iw) * pw], in_ch);
            } else {
                put_ones(padding_mask, off, in_ch);   // padding: bit 0 in patch, 1 in mask
            }
            off += in_ch;
        }
    }

    // Patch tail (after window_elems, up to the 4-word multiple):
    // set to 1, same as before (weights have 0 there -> XNOR yields 0, adds nothing).
    put_ones(patch_buffer, window_elems, (uint32_t)patch_words * 32 - window_elems);
}
