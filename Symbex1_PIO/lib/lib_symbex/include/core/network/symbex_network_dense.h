/*
 * symbex_network_dense.h -- chains layers/dense/ layers (dense,
 * gated) in sequence.
 *
 * Each layer is tagged with its type (SYMBEX_DENSE_LAYER_DENSE or
 * SYMBEX_DENSE_LAYER_GATED) -- the signatures of dense and gated are different
 * from each other (gated needs to evaluate active blocks first), so
 * this is a tagged union bounded to these two types, not a
 * generic interface for any layer (see the design note in
 * symbex_network_conv.h on why we did NOT attempt to unify everything).
 */

#ifndef SYMBEX_NETWORK_DENSE_H
#define SYMBEX_NETWORK_DENSE_H

#include <stdint.h>
#include <stdbool.h>
#include "core/symbex_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SYMBEX_NETWORK_DENSE_MAX_LAYERS 16
#define SYMBEX_NETWORK_DENSE_MAX_BLOCKS 64

enum SymbexDenseLayerType {
    SYMBEX_DENSE_LAYER_DENSE,
    SYMBEX_DENSE_LAYER_GATED
};

struct SymbexNetworkDenseLayer {
    enum SymbexDenseLayerType type;
    const struct SymbexLayerData* dense;        // used if type == SYMBEX_DENSE_LAYER_DENSE
    const struct SymbexGatedLayerData* gated;   // used if type == SYMBEX_DENSE_LAYER_GATED
    uint16_t k_active;                          // GATED only: blocks to activate
};

struct SymbexNetworkDense {
    struct SymbexNetworkDenseLayer layers[SYMBEX_NETWORK_DENSE_MAX_LAYERS];
    int layer_count;
};

void symbex_network_dense_init(struct SymbexNetworkDense* net);

bool symbex_network_dense_add_dense(struct SymbexNetworkDense* net, const struct SymbexLayerData* layer);

bool symbex_network_dense_add_gated(struct SymbexNetworkDense* net, const struct SymbexGatedLayerData* layer, uint16_t k_active);

/*
 * Runs input_planes (already packed bits -- if the source is an
 * image or floating-point vector, binarize it before calling this)
 * through ALL layers, in order. The LAST layer uses direct argmax
 * (does not pack, returns the winning class).
 *
 * buffer_a, buffer_b: ping-pong for intermediate layers -- must
 *   be large enough for the network's largest out_features/32.
 * scores_buffer: working buffer for raw scores of a dense layer
 *   before packing -- large enough for the network's largest out_features.
 * active_blocks_buffer: working buffer for gated -- large enough
 *   for SYMBEX_NETWORK_DENSE_MAX_BLOCKS.
 *
 * Returns: the index of the winning class (output of the last layer).
 */
int symbex_network_dense_run(
    const struct SymbexNetworkDense* net,
    const uint32_t* input_planes,
    uint32_t* buffer_a, uint32_t* buffer_b,
    int32_t* scores_buffer, bool* active_blocks_buffer
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_NETWORK_DENSE_H
