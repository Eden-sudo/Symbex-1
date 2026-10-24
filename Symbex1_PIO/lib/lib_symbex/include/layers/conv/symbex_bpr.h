/*
 * symbex_bpr.h -- "Binary Pixel Representation" (BPR) encoding:
 * decomposes an RGB image into 24 bit planes (8 per channel, MSB
 * first), the first step of the entire network -- C equivalent of
 * rgb_to_24bits (Python), extensively validated today.
 *
 * Answers XNOR-Net's original objection about "too few
 * channels to make binarizing the first layer worth it" -- 3
 * RGB channels are converted into 24 planes, enough width for
 * XNOR+popcount to make real sense.
 */

#ifndef SYMBEX_BPR_H
#define SYMBEX_BPR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * img_rgb: interleaved, [h][w][3] -- 3 bytes per pixel (R, G, B),
 *   typical camera format, NOT PyTorch's [C,H,W].
 * out_bits: 24 packed planes (order: R bit7..bit0, G bit7..bit0,
 *   B bit7..bit0), ready to feed the network's stem.
 */
void symbex_rgb_to_24bits(const uint8_t* img_rgb, uint32_t* out_bits, uint16_t h, uint16_t w);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_BPR_H
