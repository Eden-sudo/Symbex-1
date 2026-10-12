/*
 * symbex_types.h -- structs de topologia compartidas por las capas
 * vectoriales (densa, gated).
 *
 * NOTA DE MIGRACION: la struct SymbexConvLayerData del archivo
 * original fue RETIRADA aqui -- esta completamente superada por
 * SymbexConv2DLayer (layers/spatial/symbex_conv2d.h), que usa
 * umbrales enteros plegados en vez de scale_factors/bias_factors
 * flotantes.
 */

#ifndef SYMBEX_TYPES_H
#define SYMBEX_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Capa densa (vectorial), cuantizacion multi-bit (k_bits: 1, 4, 8, u otro)
struct SymbexLayerData {
    uint16_t in_features;
    uint16_t out_features;
    uint16_t k_bits;
    const uint32_t* weights;
};

// Capa densa con activacion condicional por bloques (tipo "mixture
// of experts" -- un "director" puntua bloques, se activan solo los
// K mejores)
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
