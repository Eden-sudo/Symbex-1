/*
 * symbex_types.h -- topology structs shared by dense (vector) layers
 * (dense, gated).
 *
 * MIGRATION NOTE: the SymbexConvLayerData struct from the original
 * file has been REMOVED here -- it is completely superseded by
 * SymbexConv2DLayer (layers/conv/symbex_conv2d.h), which uses
 * folded integer thresholds instead of floating-point
 * scale_factors/bias_factors.
 */

#ifndef SYMBEX_TYPES_H
#define SYMBEX_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Dense (vector) layer, multi-bit quantization (k_bits: 1, 4, 8, or other)
struct SymbexLayerData {
    uint16_t in_features;
    uint16_t out_features;
    uint16_t k_bits;
    const uint32_t* weights;
};

// Dense layer with block-conditional activation ("mixture
// of experts" style -- a "director" scores blocks, only the
// K best ones are activated)
struct SymbexGatedLayerData {
    uint16_t in_features;
    uint16_t out_features;
    uint16_t block_size;
    uint16_t k_bits;
    const uint32_t* gate_weights;
    const uint32_t* core_weights;
};

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_TYPES_H
