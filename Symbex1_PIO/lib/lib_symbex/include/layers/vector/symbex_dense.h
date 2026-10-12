/*
 * symbex_dense.h -- multi-bit DENSE (vector) layer.
 *
 * Migrated from symbex_bitslice.h/.cpp ("dense" describes the TYPE of layer, "bitslice" described an
 * implementation detail). Same internal logic, now using symbex_xnor_popcount (hal/) instead of the
 * old unaccelerated SYMBEX_BIT_MAC.
 *
 * SYMBEX v3 additions (for exported 1-bit MLPs):
 *   symbex_dense_entrada_u8  first layer: uint8 inputs (e.g. pixels) x 1-bit weights -> int32 raw scores
 *   symbex_dense_entrada_u8_planos   the same, with the inputs split into 8 bit planes
 *   symbex_dense_entrada_u8_mascara  the same, adding every input through a mask from its weight bit
 *   symbex_dense_umbral      raw scores -> bits with ONE integer threshold per neuron (bit = raw >= threshold);
 *                            BatchNorm, the weight scale, ReLU and the next layer's binarization threshold
 *                            are all folded into that integer by the exporter
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
 * same arithmetic as symbex_dsp_aplicar_capa, with round-to-nearest).
 */
int symbex_clasificador_binario_argmax(
    const uint32_t* input_planes,
    const struct SymbexLayerData* layer,
    const int64_t* escala, const int32_t* shift_bn,
    uint8_t q_act, uint8_t q_mult
);

/*
 * First layer of an exported MLP: uint8 inputs x 1-bit weights.
 * out[j] = sum_i x[i] * (+1 if bit (j, i) is set, -1 otherwise), exact in int32.
 * layer->k_bits must be 1; weights are packed like any 1-bit dense layer
 * (ceil(in_features / 32) words per neuron, bit i of word i / 32 = input i).
 */
void symbex_dense_entrada_u8(
    const uint8_t* x,
    int32_t* out,
    const struct SymbexLayerData* layer
);

/*
 * Same result as symbex_dense_entrada_u8, faster: the inputs are split once into their 8 bit planes
 * (planos: caller-provided buffer of 8 * ceil(in_features / 32) words) and each neuron does
 * 8 x ceil(in / 32) AND + popcount instead of visiting every input.
 */
void symbex_dense_entrada_u8_planos(
    const uint8_t* x,
    uint32_t* planos,
    int32_t* out,
    const struct SymbexLayerData* layer
);

/*
 * Same result as symbex_dense_entrada_u8, with no bit planes and no popcount: every input is added through
 * a 0x00 / 0xFF mask built from its weight bit (no branches). Each weight word is read once.
 */
void symbex_dense_entrada_u8_mascara(
    const uint8_t* x,
    int32_t* out,
    const struct SymbexLayerData* layer
);

/*
 * Raw scores -> bits: bit j = (raw[j] >= umbral[j]), packed LSB-first in 32-bit words; the padding
 * bits of the last word are 0. Use INT32_MIN as a threshold for an always-1 bit, INT32_MAX for always-0.
 */
void symbex_dense_umbral(
    const int32_t* raw,
    const int32_t* umbral,
    uint32_t* bits,
    uint16_t n
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_DENSE_H
