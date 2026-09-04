#ifndef SYMBEX_TYPES_H
#define SYMBEX_TYPES_H

#include <stdint.h>

/**
 * @struct SymbexLayerData
 * @brief Inert container for a standard layer in AoS format.
 * Maintains cache locality (32 bytes per neuron) by requiring 
 * that the bit-planes of the same neuron remain contiguous.
 */
struct SymbexLayerData {
    uint16_t in_features;
    uint16_t out_features;
    uint16_t k_bits;
    const uint32_t* weights;  
};

/**
 * @struct SymbexGatedLayerData
 * @brief Inert container for the hybrid topology (Block-Gated).
 */
struct SymbexGatedLayerData {
    uint16_t in_features;
    uint16_t out_features;
    uint16_t block_size;
    uint16_t k_bits;
    const uint32_t* gate_weights;
    const uint32_t* core_weights;
};

#endif
