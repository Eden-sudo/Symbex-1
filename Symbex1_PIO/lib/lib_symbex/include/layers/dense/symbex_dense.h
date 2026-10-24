/*
 * symbex_dense.h -- multi-bit DENSE (vector) layer.
 *
 * Migrated from symbex_bitslice.h/.cpp ("dense" describes the TYPE of layer, "bitslice" described an
 * implementation detail). Same internal logic, now using symbex_xnor_popcount (hal/) instead of the
 * old unaccelerated SYMBEX_BIT_MAC.
 *
 * SYMBEX v3 additions (for exported 1-bit MLPs):
 *   symbex_dense_input_u8        first layer: uint8 inputs (e.g. pixels) x 1-bit weights -> int32 raw scores
 *   symbex_dense_input_u8_planes same, with the inputs split into 8 bit planes
 *   symbex_dense_input_u8_mask   same, adding every input through a mask from its weight bit
 *   symbex_dense_threshold       raw scores -> bits with ONE integer threshold per neuron (bit = raw >= threshold);
 *                                BatchNorm, the weight scale, ReLU and the next layer's binarization threshold
 *                                are all folded into that integer by the exporter
 *   symbex_dense_forward_bits    binary layer + threshold in one pass: each neuron's bit is written directly,
 *   symbex_dense_input_u8_bits   first layer + threshold in one pass   (no raw-score array: for little RAM)
 *   symbex_dense_input_u8_planes_bits  same, for inputs with only k bits per value (k planes x AND + popcount)
 *
 * Weights and thresholds may live in flash: they are read with SYMBEX_READ_W32 (pgm_read_dword on AVR).
 */

#ifndef SYMBEX_DENSE_H
#define SYMBEX_DENSE_H

#include "core/symbex_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Full inference of a multi-bit dense layer. Writes the raw accumulated vote of EACH neuron into
 * output_buffer; no threshold or activation is applied, the caller decides that.
 *
 * input_planes: input bit planes (AoS).
 * output_buffer: one int32_t per neuron.
 * layer: topology and weights.
 */
void symbex_dense_forward(
    const uint32_t* input_planes,
    int32_t* output_buffer,
    const struct SymbexLayerData* layer
);

/*
 * Same as symbex_dense_forward, but returns the index of the winning neuron directly (final
 * classifier, no full buffer needed).
 */
int symbex_dense_argmax(
    const uint32_t* input_planes,
    const struct SymbexLayerData* layer
);

/*
 * Final binary classifier: binary weights (XNOR+popcount) + RSign on the input activation (already
 * binary, from the previous layer) + BatchNorm1d NOT folded -- unlike every other layer, this is the
 * LAST one, there is no following RSign to fold the BatchNorm into, so the real logit (not just a bit)
 * is needed to compare the classes with argmax.
 *
 * Fixed point, like the rest of the engine -- NO floats (Q_MULT for the scale, Q_ACT for shift_bn,
 * same arithmetic as symbex_dsp_apply_layer, with round-to-nearest).
 */
int symbex_binary_classifier_argmax(
    const uint32_t* input_planes,
    const struct SymbexLayerData* layer,
    const int64_t* scale, const int32_t* shift_bn,
    uint8_t q_act, uint8_t q_mult
);

/*
 * First layer of an exported MLP: uint8 inputs x 1-bit weights.
 * out[j] = sum_i x[i] * (+1 if bit (j, i) is set, -1 otherwise), exact in int32.
 * layer->k_bits must be 1; weights are packed like any 1-bit dense layer
 * (ceil(in_features / 32) words per neuron, bit i of word i / 32 = input i).
 */
void symbex_dense_input_u8(
    const uint8_t* x,
    int32_t* out,
    const struct SymbexLayerData* layer
);

/*
 * Same result as symbex_dense_input_u8, faster: the inputs are split once into their 8 bit planes
 * (planes: caller-provided buffer of 8 * ceil(in_features / 32) words) and each neuron does
 * 8 x ceil(in / 32) AND + popcount instead of visiting every input.
 */
void symbex_dense_input_u8_planes(
    const uint8_t* x,
    uint32_t* planes,
    int32_t* out,
    const struct SymbexLayerData* layer
);

/*
 * Same result as symbex_dense_input_u8, with no bit planes and no popcount: every input is added through
 * a 0x00 / 0xFF mask built from its weight bit (no branches). Each weight word is read once.
 */
void symbex_dense_input_u8_mask(
    const uint8_t* x,
    int32_t* out,
    const struct SymbexLayerData* layer
);

/*
 * Raw scores -> bits: bit j = (raw[j] >= threshold[j]), packed LSB-first in 32-bit words; the padding
 * bits of the last word are 0. Use INT32_MIN as a threshold for an always-1 bit, INT32_MAX for always-0.
 */
void symbex_dense_threshold(
    const int32_t* raw,
    const int32_t* threshold,
    uint32_t* bits,
    uint16_t n
);

/*
 * symbex_dense_forward (k_bits = 1) + symbex_dense_threshold in one pass: bit n of out_bits =
 * (raw score of neuron n >= threshold[n]). Same bits, without the int32 array of raw scores
 * (out_features * 4 bytes of RAM saved). out_bits: ceil(out_features / 32) words, padding bits 0.
 */
void symbex_dense_forward_bits(
    const uint32_t* input_bits,
    const int32_t* threshold,
    uint32_t* out_bits,
    const struct SymbexLayerData* layer
);

/*
 * symbex_dense_input_u8_mask + symbex_dense_threshold in one pass (first layer, uint8 inputs).
 * Same bits, without the int32 array of raw scores.
 */
void symbex_dense_input_u8_bits(
    const uint8_t* x,
    const int32_t* threshold,
    uint32_t* out_bits,
    const struct SymbexLayerData* layer
);

/*
 * First layer + threshold for inputs quantized to their k most significant bits (1 <= k <= 8): only the k top
 * bit planes are used (planes: caller buffer of k * ceil(in_features / 32) words), k AND + popcount per weight
 * word. Bits below the top k are ignored. Same bits as symbex_dense_input_u8_bits on the quantized inputs.
 */
void symbex_dense_input_u8_planes_bits(
    const uint8_t* x,
    uint8_t k,
    uint32_t* planes,
    const int32_t* threshold,
    uint32_t* out_bits,
    const struct SymbexLayerData* layer
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_DENSE_H
