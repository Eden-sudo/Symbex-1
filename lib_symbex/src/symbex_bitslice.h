/**
 * @file symbex_bitslice.h
 * @brief Pure C inference engine for Multi-Bit networks.
 *  
 * Exposes the OOP-free procedural API to propagate quantized tensors
 * using AoS memory and explicit Loop Unrolling.
 */

#ifndef SYMBEX_BITSLICE_H
#define SYMBEX_BITSLICE_H

#include "symbex_types.h"
#include "symbex_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Full inference (Multi-Bit). Writes the raw accumulated votes for all neurons.
 *  
 * @param input_planes  Pointer to the input memory block (32-bit AoS).
 * @param output_buffer Pointer to the buffer where each neuron's score will be accumulated.
 * @param layer         Pointer to the inert structure containing topology and weights.
 */
void symbex_forward_bitslice(
    const uint32_t* input_planes,  
    int32_t* output_buffer,  
    const struct SymbexLayerData* layer
);

/**
 * @brief Optimized inference (Multi-Bit). Returns only the ID of the winning class.
 *  
 * Used as the final layer (Classifier) to avoid writing to the full buffer 
 * and keep the maximum search within CPU registers.
 *  
 * @param input_planes  Pointer to the input memory block.
 * @param layer         Pointer to the inert structure containing topology and weights.
 * @return int          Index (ID) of the neuron with the highest accumulated vote count.
 */
int symbex_argmax_bitslice(
    const uint32_t* input_planes,  
    const struct SymbexLayerData* layer
);

#ifdef __cplusplus
}
#endif

#endif
