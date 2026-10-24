/*
 * symbex_network_conv.h -- chains residual BLOCKS (conv1+conv2,
 * each with its own identity addition), not flat layers.
 *
 * REDESIGN relative to the previous version: that version assumed a
 * simple chain of independent layers -- no such thing exists here, each
 * block has TWO halves with TWO internal residual additions (see
 * the real forward pass of BasicBlockBiReal, confirmed today).
 *
 * UPDATED: now it DOES support blocks with a downsample path (RSign +
 * 2x2 binary pool + 3x3 conv + affine BN, without RPReLU) -- validated in
 * Python against the real DownsampleBinario module (2.46% differences
 * above a strict tolerance, consistent with standard fixed-point rounding,
 * not a missing term).
 *
 * LIMITS: the library uses fixed-size work buffers per position. Every block is
 * checked once, when it is added (symbex_network_conv_add_block), so inference
 * never runs past them -- see SYMBEX_CONV_MAX_* below.
 */

#ifndef SYMBEX_NETWORK_CONV_H
#define SYMBEX_NETWORK_CONV_H

#include <stdint.h>
#include <stdbool.h>
#include "layers/conv/symbex_conv2d.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SYMBEX_NETWORK_CONV_MAX_BLOCKS 16

/* Largest convolution window, in 32-bit words (in_ch * K * K bits, rounded up to a multiple of 4 words):
 * 64 words = 2048 bits, i.e. up to 227 input channels with a 3x3 kernel. The patch_buffer passed to
 * symbex_network_conv_run* must hold at least this many words. */
#define SYMBEX_CONV_MAX_PATCH_WORDS 64
/* Most output channels of one convolution. */
#define SYMBEX_CONV_MAX_OUT_CH 256

struct SymbexResidualBlock {
    struct SymbexConv2DLayer conv1;
    struct SymbexConv2DLayer conv2;
    const int32_t* beta_rsign1;   // Q_ACT, one per block input channel
    const int32_t* beta_rsign2;   // Q_ACT, one per conv1 output channel
    bool has_downsample;

    // Downsample fields -- only valid if has_downsample == true.
    // Chain: binarize(beta_rsign_downsample) -> 2x2 binary pool ->
    // 3x3 conv (downsample_weights) -> affine BN (downsample_dsp), without
    // RPReLU -- the result IS the identity for the main block.
    const uint32_t* downsample_weights;
    const int32_t* beta_rsign_downsample;
    SymbexDspParamsSimple downsample_dsp;
    uint16_t downsample_kernel;   // usually 3
    uint16_t downsample_padding;  // usually 1 (kernel/2)
};

struct SymbexNetworkConv {
    const struct SymbexResidualBlock* blocks[SYMBEX_NETWORK_CONV_MAX_BLOCKS];
    int block_count;
};

void symbex_network_conv_init(struct SymbexNetworkConv* net);

/*
 * Adds a block. Returns false -- and does NOT add it -- if the network is full or the block is invalid:
 *   - a convolution window over SYMBEX_CONV_MAX_PATCH_WORDS words, or more than SYMBEX_CONV_MAX_OUT_CH
 *     output channels (conv1, conv2 and the downsample);
 *   - kernel or stride 0, or conv2 not matching conv1's output (channels and size);
 *   - fixed-point formats outside what the DSP formula needs: q_act >= 1 and q_mult > q_act;
 *   - a downsample whose output size differs from conv1's (the two are added together).
 */
bool symbex_network_conv_add_block(struct SymbexNetworkConv* net, const struct SymbexResidualBlock* block);

/*
 * Runs the image (already in bits format, output from the binarized stem)
 * through all the blocks. x_initial_q_act: Q_ACT input values to the 
 * first block (stem output, converted to Q_ACT).
 * out_final_q_act: output buffer, Q_ACT values of the LAST block
 * (ready to be converted to float and passed to the classifier).
 *
 * Returns true if successful. false only if block_count == 0.
 *
 * buffer_a, buffer_b, buffer_identity: Q_ACT workspace buffers,
 * each large enough for the largest activation map in the network.
 * line_buffer, patch_buffer: same as before (bits); patch_buffer must hold
 * at least SYMBEX_CONV_MAX_PATCH_WORDS words.
 */
bool symbex_network_conv_run(
    const struct SymbexNetworkConv* net,
    const int32_t* x_initial_q_act,
    int32_t* buffer_a, int32_t* buffer_b,
    uint32_t* bits_buffer,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* out_final_q_act
);

/*
 * Processes ONE full residual block (with downsample if applicable),
 * writing the result into dest_q_act.
 *
 * debug_identity1 / debug_conv1_out: OPTIONAL (pass NULL if not
 * needed) -- if not NULL, copies identity1 (the real "shortcut" 
 * used for conv1: downsample(x) or straight x) and the output of 
 * conv1 (=identity2), respectively. Expected size:
 * conv1.out_ch * h1 * w1 (h1/w1 = OUTPUT resolution of conv1).
 * Reuses the ALREADY VALIDATED computation inside the function, 
 * without manually reproducing it elsewhere (avoids transcription bugs).
 */
void symbex_network_conv_process_block(
    const struct SymbexResidualBlock* block,
    const int32_t* x_current_q_act,
    int32_t* dest_q_act,
    int32_t* temp_buffer,
    uint32_t* bits_buffer,
    int32_t* identity_downsample_buffer,
    uint32_t* bits_full_downsample_buffer,
    uint32_t* bits_pooled_downsample_buffer,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* debug_identity1, int32_t* debug_conv1_out
);

/*
 * Extended version -- required if the network has blocks with
 * downsample. The 3 extra buffers are ONLY for the downsample path
 * (RSign+pool+conv3x3+BN):
 *   identity_downsample_buffer: Q_ACT, size of the largest downsample 
 *     identity in the network (conv1 out_ch x its h x its w)
 *   bits_full_downsample_buffer: bits, FULL input resolution to the 
 *     downsample (before pool)
 *   bits_pooled_downsample_buffer: bits, REDUCED resolution (half)
 *
 * symbex_network_conv_run (the simple version) calls this passing
 * NULL for these 3 -- it only works if NO block has a downsample.
 */
bool symbex_network_conv_run_ext(
    const struct SymbexNetworkConv* net,
    const int32_t* x_initial_q_act,
    int32_t* buffer_a, int32_t* buffer_b,
    uint32_t* bits_buffer,
    int32_t* identity_downsample_buffer,
    uint32_t* bits_full_downsample_buffer,
    uint32_t* bits_pooled_downsample_buffer,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* out_final_q_act
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_NETWORK_CONV_H
