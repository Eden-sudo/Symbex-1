/*
 * symbex_mac_1bit.h -- Binarized MAC with fixed-point arithmetic (DSP),
 * validated end-to-end in Python against real PyTorch
 * (84.80% vs 86.20% accuracy, 500 CIFAR-10 images).
 *
 * REPLACES the older version (SYMBEX_MODO_SIMPLE/TRIPLE) -- that
 * version DID NOT include the residual addition (identity) that the real
 * model's forward pass does, which caused a ~27-38% error per layer
 * when rigorously tested (100+ images, not just 1).
 *
 * Formula, identical for ANY conv (1 or 2) in any block:
 *
 *   y = (raw_popcount * scale) >> (Q_MULT - Q_ACT)
 *   y += shift_bn
 *   y += identity                     <- the real addition, at runtime
 *   diff = y - gamma_rprelu
 *   if diff <= 0: output = shift(diff, log2_beta, sign_beta) + zeta
 *   else:         output = diff + zeta
 *
 * "output" remains in Q_ACT format -- it is used DIRECTLY as the
 * "identity" for the next layer. The bit to feed the NEXT
 * convolution is obtained separately, by comparing the output against the beta
 * of the following RSign (see symbex_dsp_binarize).
 *
 * CHANGE (int32 version): 'scale' goes from int64 to int32, in both
 * SymbexDspParams and SymbexDspParamsSimple. With Q_MULT=20, any
 * |scale| < 2048 fits perfectly (the exporter verifies this).
 * Multiplications are still 64-bit, but 32x32->64. Result is
 * bit-exact identical to the int64 version.
 */

#ifndef SYMBEX_MAC_1BIT_H
#define SYMBEX_MAC_1BIT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DSP parameters for ONE layer (conv1 or conv2 of a block).
 * One value per output channel in each array -- except q_act/q_mult, 
 * which are global to the entire network (same format in all layers).
 */
typedef struct {
    const int32_t* scale;           // Q_MULT -- combined alpha*scale_bn (previously int64)
    const int32_t* shift_bn;        // Q_ACT
    const int32_t* gamma_rprelu;    // Q_ACT -- RPReLU branch threshold (bias1)
    const int32_t* zeta;            // Q_ACT (bias2)
    bool use_shift;                 // true: RPReLU_Shift (power of 2, uses shifting).
                                    // false: standard RPReLU (multiplication, uses slope_q)
    const int8_t* log2_beta;        // only if use_shift==true -- exponent,
                                    // NEGATIVE if beta < 1 (the common case)
    const int8_t* sign_beta;        // only if use_shift==true -- +1 or -1
    const int32_t* slope_q;         // only if use_shift==false -- slope in Q_ACT
    uint8_t q_act;                  // fractional bits for activations
    uint8_t q_mult;                 // fractional bits for the multiplier
} SymbexDspParams;

/*
 * Computes the output of ONE layer, for the given 'channel', given the
 * raw popcount of that position and the corresponding identity (Q_ACT) 
 * to add. Returns the output in Q_ACT format -- USE THIS as the
 * identity for the next layer, or pass it to symbex_dsp_binarize
 * to get the bit that feeds the next convolution.
 */
int32_t symbex_dsp_apply_layer(int32_t raw_popcount, int32_t identity_q_act, uint16_t channel, const SymbexDspParams* params);

/*
 * Converts a Q_ACT value (output of symbex_dsp_apply_layer, or the
 * input of a block) into a bit -- comparing against the beta (Q_ACT)
 * of the corresponding RSign. Computes sign(value - beta).
 */
bool symbex_dsp_binarize(int32_t val_q_act, int32_t beta_rsign_q_act);

/*
 * SIMPLIFIED version, for the downsample -- affine BN only, NO
 * RPReLU (the downsample doesn't have that branch). 
 * y = (raw*scale) >> shift + shift_bn -- this is directly the 
 * "identity" to add in the main block, no extra steps needed.
 */
typedef struct {
    const int32_t* scale;           // previously int64
    const int32_t* shift_bn;
    uint8_t q_act;
    uint8_t q_mult;
} SymbexDspParamsSimple;

int32_t symbex_dsp_apply_downsample(int32_t raw_popcount, uint16_t channel, const SymbexDspParamsSimple* params);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_MAC_1BIT_H
