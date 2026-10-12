/*
 * symbex_bpr.c -- RGB (uint8, HWC) -> 24 planos de bits, FORMATO HWC.
 *
 * Mismo orden de planos que antes: plano = color*8 + (7 - bit), o sea el
 * bit mas significativo de cada color primero. En HWC eso significa que la
 * palabra de cada pixel es simplemente:
 *     invertir_bits(R) | invertir_bits(G) << 8 | invertir_bits(B) << 16
 */

#include "layers/spatial/symbex_bpr.h"

static inline uint32_t invertir_bits8(uint32_t v) {
    v = ((v & 0xF0u) >> 4) | ((v & 0x0Fu) << 4);
    v = ((v & 0xCCu) >> 2) | ((v & 0x33u) << 2);
    v = ((v & 0xAAu) >> 1) | ((v & 0x55u) << 1);
    return v;
}

void symbex_rgb_a_24bits(const uint8_t* imagen_rgb, uint32_t* out_bits, uint16_t h, uint16_t w) {
    uint32_t n = (uint32_t)h * w;
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t* px = &imagen_rgb[p * 3];
        out_bits[p] = invertir_bits8(px[0]) | (invertir_bits8(px[1]) << 8) | (invertir_bits8(px[2]) << 16);
    }
}
