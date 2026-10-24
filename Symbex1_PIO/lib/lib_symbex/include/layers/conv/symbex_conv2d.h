/*
 * symbex_conv2d.h -- binarized 2D convolution layer, with the
 * corrected DSP formula (real residual sum) AND exact edge padding
 * correction (uniform bit=0 row+col, corrected per channel via
 * padding_mask -- previous default "padding=1" did not match
 * PyTorch's real zero-padding).
 *
 * IMPORTANT CHANGE regarding previous version: no longer writes
 * bits directly -- writes Q_ACT VALUES (fixed-point). Converting to
 * bits is a separate step (symbex_dsp_binarize_map).
 *
 * REMOVED: symbex_conv2d_prepare_row (empty since the HWC format: the window is read straight from the map
 * by symbex_conv2d_extract_window).
 */

#ifndef SYMBEX_CONV2D_H
#define SYMBEX_CONV2D_H

#include <stdint.h>
#include "ops/symbex_mac_1bit.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SymbexConv2DLayer {
    SymbexDspParams dsp;
    const uint32_t* weights;
    uint16_t in_ch, in_h, in_w;
    uint16_t out_ch, kernel, stride, padding;
};

/*
 * Processes ONE position (oh, ow), ALL output channels.
 * padding_mask: same size as patch_buffer, 1 in each bit that
 * is padding (row or column outside image), 0 in real ones --
 * all-zero for interior positions. Used to correct the
 * exact popcount per channel.
 */
void symbex_conv2d_process_position(
    const uint32_t* patch_buffer,
    const uint32_t* padding_mask,
    const int32_t* identity_q_act,
    int32_t* out_values_q_act,
    const struct SymbexConv2DLayer* layer
);

void symbex_dsp_binarize_map(
    const int32_t* values_q_act, uint32_t* out_bits,
    const int32_t* beta_rsign_q_act,
    uint16_t ch, uint16_t h, uint16_t w
);

void symbex_binarize_image(const float* input_map, uint32_t* out_bits,
                           uint16_t ch, uint16_t h, uint16_t w);

/*
 * padding_mask: OUTPUT buffer, same size as patch_buffer --
 * this function fills it marking which positions are padding.
 * in_h and oh are NEW -- necessary to detect ROW padding
 * here as well (previously only column padding was detected).
 */
void symbex_conv2d_extract_window(const uint32_t* hwc_map, uint32_t* patch_buffer, uint32_t* padding_mask,
                                  uint16_t in_ch, uint16_t in_h, uint16_t in_w, uint16_t kernel,
                                  uint16_t stride, uint16_t padding, uint16_t oh, uint16_t ow);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_CONV2D_H
