/*
 * symbex_gated.h -- DENSE layer with block-conditional
 * activation ("mixture of experts" style). A "director" sub-network
 * scores neuron blocks, only the K best ones are activated, the
 * rest are skipped (real conditional compute, not cosmetic).
 *
 * Migrated from symbex_gated.h/.cpp -- same logic, now uses
 * symbex_xnor_popcount (hal/) instead of the old SYMBEX_BIT_MAC.
 */

#ifndef SYMBEX_GATED_H
#define SYMBEX_GATED_H

#include "core/symbex_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Phase 1: "director" evaluation. Scores each block, marks the
 * K best ones as active.
 *
 * active_blocks: output array, one bool per block.
 * k_active: how many blocks to activate at most.
 */
void symbex_gated_evaluate(
    const uint32_t* input_planes,
    bool* active_blocks,
    const struct SymbexGatedLayerData* layer,
    uint16_t k_active
);

/*
 * Phase 2: conditional inference. Only processes blocks marked
 * active -- inactive ones are skipped (advances weight pointer,
 * does not compute). Packs output bit by bit (bipolar STE:
 * score > 0 -> bit 1).
 */
void symbex_gated_forward(
    const uint32_t* input_planes,
    uint32_t* output_buffer,
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
);

/*
 * Phase 2 (alternative): conditional inference with direct final
 * classification (argmax), to be used as the last layer.
 */
int symbex_gated_argmax(
    const uint32_t* input_planes,
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_GATED_H
