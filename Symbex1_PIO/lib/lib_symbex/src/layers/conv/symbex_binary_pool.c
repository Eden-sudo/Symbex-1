/*
 * symbex_binary_pool.c -- 2x2 binary pooling, HWC FORMAT.
 * Same rule as before: the output bit is 1 if at least 2 out of the 4 input bits are 1.
 * In HWC this is done word by word, processing 32 channels at once:
 *     at least 2 of {a,b,c,d}  =  (a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d)
 */

#include "layers/conv/symbex_binary_pool.h"

void symbex_binary_pool_2x2(const uint32_t* bits_in, uint32_t* bits_out, uint16_t ch, uint16_t h, uint16_t w) {
    uint16_t pw = (ch + 31) / 32;
    uint16_t out_h = h / 2;
    uint16_t out_w = w / 2;

    for (uint16_t y = 0; y < out_h; y++) {
        for (uint16_t x = 0; x < out_w; x++) {
            const uint32_t* p00 = &bits_in[((uint32_t)(2 * y) * w + 2 * x) * pw];
            const uint32_t* p01 = p00 + pw;
            const uint32_t* p10 = &bits_in[((uint32_t)(2 * y + 1) * w + 2 * x) * pw];
            const uint32_t* p11 = p10 + pw;
            uint32_t* dest = &bits_out[((uint32_t)y * out_w + x) * pw];
            
            for (uint16_t i = 0; i < pw; i++) {
                uint32_t a = p00[i], b = p01[i], c = p10[i], d = p11[i];
                dest[i] = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d);
            }
        }
    }
}
