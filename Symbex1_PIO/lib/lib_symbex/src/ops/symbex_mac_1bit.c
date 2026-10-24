/*
 * symbex_mac_1bit.c -- DSP operations for 1-bit MAC layers.
 *
 * Bit-exact arithmetic matching the Python V4 simulator (validated on the 
 * FULL CIFAR-10 test set: 74.63% simulator vs 74.68% real PyTorch, 
 * 10,000 images -- 0.05 points difference).
 *
 * Uses "round-to-nearest" on every shift instead of truncation. Supports
 * both RPReLU_Shift (power of 2, uses bit shifting) and standard RPReLU 
 * (general multiplication using slope_q). The stem of the fully binarized 
 * model uses standard RPReLU.
 *
 * INT32 + BRANCHLESS RPReLU OPTIMIZATIONS:
 *   1. Int32 scaling: The raw*scale product is 32x32->64, not 64x64.
 *      While similar on x86-64, on 32-bit MCUs (like ESP32-S3) this is the 
 *      difference between a couple of instructions and a full 64x64 routine.
 *   2. Branchless: Computes both branches and selects using a conditional move
 *      (e.g., cmov on x86). The sign of 'diff' is highly unpredictable, and 
 *      a branch misprediction costs ~15 cycles on modern CPUs.
 *
 * The Q_ACT values (y, diff, output) fit in int32 as long as the real 
 * value is < 32768, maintaining the int32 return type.
 */

#include "ops/symbex_mac_1bit.h"

int32_t symbex_dsp_apply_layer(int32_t raw_popcount, int32_t identity_q_act, uint16_t channel, const SymbexDspParams* params) {
    uint8_t shift_rescale = params->q_mult - params->q_act;
    int64_t round_mult = (int64_t)1 << (shift_rescale - 1);

    int32_t y = (int32_t)((((int64_t)raw_popcount * params->scale[channel]) + round_mult) >> shift_rescale);
    y += params->shift_bn[channel];
    y += identity_q_act;

    int32_t diff = y - params->gamma_rprelu[channel];
    int64_t neg_branch; // 64 bits as before: with log2_beta >= 0, the shift can overflow int32

    if (params->use_shift) {
        // This branch IS well predicted: use_shift is constant per layer.
        // Magnitude is always computed on a >= 0 value (0 if diff > 0).
        // Left-shifting a negative number is undefined behavior in C.
        int64_t negated = (diff > 0) ? 0 : -(int64_t)diff;
        int8_t log2b = params->log2_beta[channel];
        int64_t magnitude = (log2b >= 0) ? (negated << log2b) : (negated >> (-log2b));
        neg_branch = (params->sign_beta[channel] >= 0) ? -magnitude : magnitude;
    } else {
        // Standard RPReLU: diff * slope_q, round-to-nearest.
        // We ALWAYS add half (regardless of sign) -- the arithmetic shift
        // acts as 'floor', so adding half first creates 'round-to-nearest'
        // for ANY sign.
        int64_t round_act = (int64_t)1 << (params->q_act - 1);
        neg_branch = (((int64_t)diff * params->slope_q[channel]) + round_act) >> params->q_act;
    }

    int64_t output = ((diff > 0) ? (int64_t)diff : neg_branch) + params->zeta[channel];
    return (int32_t)output;
}

bool symbex_dsp_binarize(int32_t val_q_act, int32_t beta_rsign_q_act) {
    return val_q_act >= beta_rsign_q_act;
}

int32_t symbex_dsp_apply_downsample(int32_t raw_popcount, uint16_t channel, const SymbexDspParamsSimple* params) {
    uint8_t shift_rescale = params->q_mult - params->q_act;
    int64_t round_val = (int64_t)1 << (shift_rescale - 1);
    int32_t y = (int32_t)((((int64_t)raw_popcount * params->scale[channel]) + round_val) >> shift_rescale);
    return y + params->shift_bn[channel];
}
