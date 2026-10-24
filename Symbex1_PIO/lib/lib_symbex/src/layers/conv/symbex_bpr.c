/*
 * symbex_bpr.c -- RGB (uint8, HWC) -> 24 bit planes, HWC FORMAT.
 *
 * Same plane order as before: plane = color * 8 + (7 - bit), meaning the
 * most significant bit of each color comes first. In HWC, this means the
 * word for each pixel is simply:
 *     reverse_bits(R) | reverse_bits(G) << 8 | reverse_bits(B) << 16
 */

#include "layers/conv/symbex_bpr.h"

static inline uint32_t reverse_bits8(uint32_t v) {
    v = ((v & 0xF0u) >> 4) | ((v & 0x0Fu) << 4);
    v = ((v & 0xCCu) >> 2) | ((v & 0x33u) << 2);
    v = ((v & 0xAAu) >> 1) | ((v & 0x55u) << 1);
    return v;
}

void symbex_rgb_to_24bits(const uint8_t* img_rgb, uint32_t* out_bits, uint16_t h, uint16_t w) {
    uint32_t n = (uint32_t)h * w;
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t* px = &img_rgb[p * 3];
        out_bits[p] = reverse_bits8(px[0]) | (reverse_bits8(px[1]) << 8) | (reverse_bits8(px[2]) << 16);
    }
}
