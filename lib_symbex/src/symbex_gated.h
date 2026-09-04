#ifndef SYMBEX_GATED_H
#define SYMBEX_GATED_H

#include "symbex_types.h"
#include "symbex_core.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Phase 1: Conditional gate evaluation. Determines the active blocks.
 *  
 * @param input_planes  Pointer to the input memory block.
 * @param active_blocks Output boolean array indicating the activation state per block.
 * @param layer         Pointer to the Gated layer structure.
 * @param k_active      Maximum number of blocks to activate.
 */
void symbex_evaluate_gates(
    const uint32_t* input_planes,  
    bool* active_blocks,  
    const struct SymbexGatedLayerData* layer,
    uint16_t k_active
);

/**
 * @brief Phase 2: Conditional inference. Exclusively processes blocks with an active state.
 *
 * @param input_planes  Pointer to the input memory block.
 * @param output_buffer Pointer to the output buffer where packed bits will be written.
 * @param active_blocks Boolean array indicating the activation state per block.
 * @param layer         Pointer to the Gated layer structure.
 */
void symbex_forward_gated(
    const uint32_t* input_planes,  
    uint32_t* output_buffer,  
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
);

/**
 * @brief Phase 2 (Alternative): Conditional inference with final classification (Argmax).
 */
int symbex_argmax_gated(
    const uint32_t* input_planes,  
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
);

#ifdef __cplusplus
}
#endif

#endif
