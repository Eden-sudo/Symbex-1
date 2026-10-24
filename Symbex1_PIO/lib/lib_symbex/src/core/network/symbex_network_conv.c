/*
 * symbex_network_conv.c -- BITS IN HWC FORMAT (see symbex_conv2d.c).
 *
 * Changes from the previous version:
 *   - binarize_map_q_act writes HWC bits (channels of each pixel together).
 *   - line_buffer is no longer used: extract_window reads directly from the bit map.
 *     The line_buffer parameter is kept in the signatures to avoid breaking
 *     callers, but it is ignored.
 *   - every block is validated when it is added (block_is_valid): the per-position work buffers have
 *     fixed sizes (SYMBEX_CONV_MAX_PATCH_WORDS, SYMBEX_CONV_MAX_OUT_CH), so a block that does not fit
 *     is rejected instead of writing past them during inference.
 * The int32 maps (x, identity, outputs) remain in CHW format, exactly as before.
 */

#include "core/network/symbex_network_conv.h"
#include "layers/conv/symbex_binary_pool.h"
#include "hal/symbex_hal.h"
#include "ops/symbex_bitops.h"
#include <stddef.h>
#include <string.h>

static uint16_t out_dim(uint16_t in_dim, uint16_t kernel, uint16_t stride, uint16_t padding) {
    return (in_dim + 2 * padding - kernel) / stride + 1;
}

static uint32_t patch_words_for(uint32_t in_ch, uint32_t kernel) {
    uint32_t elems = in_ch * kernel * kernel;
    return (((elems + 31) / 32) + 3) & ~3u;
}

// q formats the DSP formula needs: rounding shifts by (q_mult - q_act - 1) and (q_act - 1).
static bool q_formats_ok(uint8_t q_act, uint8_t q_mult) {
    return q_act >= 1 && q_mult > q_act && q_mult < 63;
}

static bool conv_is_valid(const struct SymbexConv2DLayer* L) {
    if (L->kernel == 0 || L->stride == 0 || L->in_ch == 0 || L->out_ch == 0 || L->weights == NULL) return false;
    if (L->in_h + 2 * L->padding < L->kernel || L->in_w + 2 * L->padding < L->kernel) return false;
    if (patch_words_for(L->in_ch, L->kernel) > SYMBEX_CONV_MAX_PATCH_WORDS) return false;
    if (L->out_ch > SYMBEX_CONV_MAX_OUT_CH) return false;
    return q_formats_ok(L->dsp.q_act, L->dsp.q_mult);
}

static bool block_is_valid(const struct SymbexResidualBlock* b) {
    if (b == NULL || b->beta_rsign1 == NULL || b->beta_rsign2 == NULL) return false;
    if (!conv_is_valid(&b->conv1) || !conv_is_valid(&b->conv2)) return false;
    uint16_t h1 = out_dim(b->conv1.in_h, b->conv1.kernel, b->conv1.stride, b->conv1.padding);
    uint16_t w1 = out_dim(b->conv1.in_w, b->conv1.kernel, b->conv1.stride, b->conv1.padding);
    // conv2 reads conv1's output and adds it as its identity: same channels and size in and out
    if (b->conv2.in_ch != b->conv1.out_ch || b->conv2.in_h != h1 || b->conv2.in_w != w1) return false;
    if (b->conv2.out_ch != b->conv1.out_ch) return false;
    if (out_dim(h1, b->conv2.kernel, b->conv2.stride, b->conv2.padding) != h1 ||
        out_dim(w1, b->conv2.kernel, b->conv2.stride, b->conv2.padding) != w1) return false;
    if (!b->has_downsample) {
        // the identity of conv1 is the block input itself
        return b->conv1.out_ch == b->conv1.in_ch && h1 == b->conv1.in_h && w1 == b->conv1.in_w;
    }
    if (b->downsample_weights == NULL || b->beta_rsign_downsample == NULL || b->downsample_kernel == 0) return false;
    if (patch_words_for(b->conv1.in_ch, b->downsample_kernel) > SYMBEX_CONV_MAX_PATCH_WORDS) return false;
    if (!q_formats_ok(b->downsample_dsp.q_act, b->downsample_dsp.q_mult)) return false;
    uint16_t ph = b->conv1.in_h / 2, pw = b->conv1.in_w / 2;
    if (ph + 2 * b->downsample_padding < b->downsample_kernel || pw + 2 * b->downsample_padding < b->downsample_kernel) return false;
    // the downsample output is added to conv1's output: same size
    return out_dim(ph, b->downsample_kernel, 1, b->downsample_padding) == h1 &&
           out_dim(pw, b->downsample_kernel, 1, b->downsample_padding) == w1;
}

void symbex_network_conv_init(struct SymbexNetworkConv* net) {
    net->block_count = 0;
}

bool symbex_network_conv_add_block(struct SymbexNetworkConv* net, const struct SymbexResidualBlock* block) {
    if (net->block_count >= SYMBEX_NETWORK_CONV_MAX_BLOCKS) return false;
    if (!block_is_valid(block)) return false;
    net->blocks[net->block_count++] = block;
    return true;
}

// int32 CHW -> bits HWC. It iterates channel by channel to READ sequentially
// (input is CHW); writes hit a small map (<= 4 KB) that lives in cache,
// so jumping around is not a performance issue.
static void binarize_map_q_act(const int32_t* values, uint32_t* bits_out,
                               const int32_t* beta, uint16_t ch, uint16_t h, uint16_t w) {
    uint16_t pw = (ch + 31) / 32;
    uint32_t hw = (uint32_t)h * w;
    memset(bits_out, 0, hw * pw * sizeof(uint32_t));
    for (uint16_t c = 0; c < ch; c++) {
        const int32_t* row = &values[(uint32_t)c * hw];
        const int32_t threshold = beta[c];
        uint32_t* dest = &bits_out[c / 32];
        const uint32_t bit = 1UL << (c % 32);
        for (uint32_t p = 0; p < hw; p++) {
            if (symbex_dsp_binarize(row[p], threshold)) dest[p * pw] |= bit;
        }
    }
}

static void process_full_layer(
    const uint32_t* input_bits, const int32_t* identity_q_act,
    int32_t* out_q_act, const struct SymbexConv2DLayer* layer,
    uint32_t* line_buffer, uint32_t* patch_buffer)
{
    (void)line_buffer;
    uint16_t out_h = out_dim(layer->in_h, layer->kernel, layer->stride, layer->padding);
    uint16_t out_w = out_dim(layer->in_w, layer->kernel, layer->stride, layer->padding);

    for (uint16_t oh = 0; oh < out_h; oh++) {
        for (uint16_t ow = 0; ow < out_w; ow++) {
            uint32_t padding_mask[SYMBEX_CONV_MAX_PATCH_WORDS];
            symbex_conv2d_extract_window(input_bits, patch_buffer, padding_mask,
                                         layer->in_ch, layer->in_h, layer->in_w, layer->kernel,
                                         layer->stride, layer->padding, oh, ow);

            int32_t identity_pos[SYMBEX_CONV_MAX_OUT_CH];
            int32_t out_pos[SYMBEX_CONV_MAX_OUT_CH];
            for (uint16_t oc = 0; oc < layer->out_ch; oc++) {
                identity_pos[oc] = identity_q_act[(oc * out_h + oh) * out_w + ow];
            }

            symbex_conv2d_process_position(patch_buffer, padding_mask, identity_pos, out_pos, layer);

            for (uint16_t oc = 0; oc < layer->out_ch; oc++) {
                out_q_act[(oc * out_h + oh) * out_w + ow] = out_pos[oc];
            }
        }
    }
}

static void process_downsample(
    const int32_t* x_q_act, const struct SymbexResidualBlock* block,
    uint16_t in_ch, uint16_t in_h, uint16_t in_w,
    uint32_t* bits_full_ds, uint32_t* bits_pooled,
    uint32_t* patch_buffer,
    int32_t* out_identity_q_act)
{
    binarize_map_q_act(x_q_act, bits_full_ds, block->beta_rsign_downsample,
                       in_ch, in_h, in_w);

    symbex_binary_pool_2x2(bits_full_ds, bits_pooled, in_ch, in_h, in_w);

    uint16_t pooled_h = in_h / 2;
    uint16_t pooled_w = in_w / 2;
    uint16_t out_ch = block->conv1.out_ch;
    uint16_t kernel = block->downsample_kernel;
    uint16_t padding = block->downsample_padding;

    uint16_t window_elems = in_ch * kernel * kernel;
    uint16_t patch_words = (((window_elems + 31) / 32) + 3) & ~3;

    uint16_t out_h = out_dim(pooled_h, kernel, 1, padding);
    uint16_t out_w = out_dim(pooled_w, kernel, 1, padding);

    for (uint16_t oh = 0; oh < out_h; oh++) {
        for (uint16_t ow = 0; ow < out_w; ow++) {
            uint32_t padding_mask[SYMBEX_CONV_MAX_PATCH_WORDS];
            symbex_conv2d_extract_window(bits_pooled, patch_buffer, padding_mask,
                                         in_ch, pooled_h, pooled_w, kernel, 1, padding, oh, ow);

            int32_t n_padding = 0;
            for (uint16_t i = 0; i < patch_words; i++) n_padding += symbex_hal_popcount32(padding_mask[i]);

            const uint32_t* w_ptr = block->downsample_weights;
            for (uint16_t oc = 0; oc < out_ch; oc++) {
                int32_t raw_measured = symbex_xnor_popcount(patch_buffer, w_ptr, patch_words, window_elems);
                int32_t raw = raw_measured;
                if (n_padding > 0) {
                    int32_t padding_matches = symbex_popcount_and(padding_mask, w_ptr, patch_words);
                    raw = raw_measured - n_padding + 2 * padding_matches;
                }
                w_ptr += patch_words;
                out_identity_q_act[(oc * out_h + oh) * out_w + ow] =
                    symbex_dsp_apply_downsample(raw, oc, &block->downsample_dsp);
            }
        }
    }
}

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
) {
    const int32_t* identity_conv1 = x_current_q_act;
    if (block->has_downsample) {
        static uint32_t patch_buffer_ds[SYMBEX_CONV_MAX_PATCH_WORDS];
        process_downsample(x_current_q_act, block,
                           block->conv1.in_ch, block->conv1.in_h, block->conv1.in_w,
                           bits_full_downsample_buffer, bits_pooled_downsample_buffer,
                           patch_buffer_ds, identity_downsample_buffer);
        identity_conv1 = identity_downsample_buffer;
    }

    uint16_t h1 = out_dim(block->conv1.in_h, block->conv1.kernel, block->conv1.stride, block->conv1.padding);
    uint16_t w1 = out_dim(block->conv1.in_w, block->conv1.kernel, block->conv1.stride, block->conv1.padding);

    // This copy must happen BEFORE process_full_layer (temp_buffer
    // and identity_conv1 can be the same pointer).
    if (debug_identity1 != NULL) {
        uint32_t n = (uint32_t)block->conv1.out_ch * h1 * w1;
        for (uint32_t k = 0; k < n; k++) debug_identity1[k] = identity_conv1[k];
    }

    binarize_map_q_act(x_current_q_act, bits_buffer, block->beta_rsign1,
                       block->conv1.in_ch, block->conv1.in_h, block->conv1.in_w);

    process_full_layer(bits_buffer, identity_conv1, temp_buffer, &block->conv1, line_buffer, patch_buffer);

    if (debug_conv1_out != NULL) {
        uint32_t n = (uint32_t)block->conv1.out_ch * h1 * w1;
        for (uint32_t k = 0; k < n; k++) debug_conv1_out[k] = temp_buffer[k];
    }

    binarize_map_q_act(temp_buffer, bits_buffer, block->beta_rsign2,
                       block->conv1.out_ch, h1, w1);

    process_full_layer(bits_buffer, temp_buffer, dest_q_act, &block->conv2, line_buffer, patch_buffer);
}

bool symbex_network_conv_run(
    const struct SymbexNetworkConv* net,
    const int32_t* x_initial_q_act,
    int32_t* buffer_a, int32_t* buffer_b,
    uint32_t* bits_buffer,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* out_final_q_act
) {
    return symbex_network_conv_run_ext(net, x_initial_q_act, buffer_a, buffer_b,
                                       bits_buffer, NULL, NULL, NULL,
                                       line_buffer, patch_buffer, out_final_q_act);
}

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
) {
    if (net->block_count == 0) return false;

    for (int i = 0; i < net->block_count; i++) {
        if (net->blocks[i]->has_downsample &&
            (identity_downsample_buffer == NULL || bits_full_downsample_buffer == NULL || bits_pooled_downsample_buffer == NULL)) {
            return false;
        }
    }

    int32_t* x_current = buffer_a;
    if (x_initial_q_act != buffer_a) {
        const struct SymbexConv2DLayer* first_conv1 = &net->blocks[0]->conv1;
        uint32_t size = (uint32_t)first_conv1->in_ch * first_conv1->in_h * first_conv1->in_w;
        for (uint32_t k = 0; k < size; k++) buffer_a[k] = x_initial_q_act[k];
    }

    int32_t* dest_conv1 = buffer_b;

    for (int i = 0; i < net->block_count; i++) {
        const struct SymbexResidualBlock* block = net->blocks[i];
        bool is_last_block = (i == net->block_count - 1);
        int32_t* dest = is_last_block ? out_final_q_act : x_current;

        symbex_network_conv_process_block(
            block, x_current, dest, dest_conv1,
            bits_buffer, identity_downsample_buffer,
            bits_full_downsample_buffer, bits_pooled_downsample_buffer,
            line_buffer, patch_buffer, NULL, NULL
        );
    }

    return true;
}
