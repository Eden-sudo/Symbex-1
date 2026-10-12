/*
 * symbex_network_vector.c -- implementacion. Encadena capas dense/
 * gated, empaquetando las puntuaciones crudas de dense entre capas
 * (gated ya empaqueta internamente). La ultima capa usa argmax
 * directo, sin empaquetar -- es la decision final de clasificacion.
 */

#include "core/network/symbex_network_vector.h"
#include "layers/vector/symbex_dense.h"
#include "layers/vector/symbex_gated.h"
#include "ops/symbex_bitpack.h"

void symbex_network_vector_init(struct SymbexNetworkVector* net) {
    net->layer_count = 0;
}

bool symbex_network_vector_add_dense(struct SymbexNetworkVector* net, const struct SymbexLayerData* layer) {
    if (net->layer_count >= SYMBEX_NETWORK_VECTOR_MAX_LAYERS) return false;
    struct SymbexNetworkVectorLayer* slot = &net->layers[net->layer_count++];
    slot->type = SYMBEX_VECTOR_DENSE;
    slot->dense = layer;
    slot->gated = 0;
    slot->k_active = 0;
    return true;
}

bool symbex_network_vector_add_gated(struct SymbexNetworkVector* net, const struct SymbexGatedLayerData* layer, uint16_t k_active) {
    if (net->layer_count >= SYMBEX_NETWORK_VECTOR_MAX_LAYERS) return false;
    struct SymbexNetworkVectorLayer* slot = &net->layers[net->layer_count++];
    slot->type = SYMBEX_VECTOR_GATED;
    slot->dense = 0;
    slot->gated = layer;
    slot->k_active = k_active;
    return true;
}

int symbex_network_vector_run(
    const struct SymbexNetworkVector* net,
    const uint32_t* input_planes,
    uint32_t* buffer_a, uint32_t* buffer_b,
    int32_t* scores_buffer, bool* active_blocks_buffer
) {
    if (net->layer_count == 0) return 0;

    const uint32_t* current_input = input_planes;
    uint32_t* current_output = buffer_a;

    for (int i = 0; i < net->layer_count; i++) {
        const struct SymbexNetworkVectorLayer* capa = &net->layers[i];
        bool es_ultima = (i == net->layer_count - 1);

        if (capa->type == SYMBEX_VECTOR_DENSE) {
            if (es_ultima) {
                return symbex_dense_argmax(current_input, capa->dense);
            }
            symbex_dense_forward(current_input, scores_buffer, capa->dense);
            symbex_bitpack_from_scores(scores_buffer, current_output, capa->dense->out_features);
        } else {
            symbex_gated_evaluate(current_input, active_blocks_buffer, capa->gated, capa->k_active);
            if (es_ultima) {
                return symbex_gated_argmax(current_input, active_blocks_buffer, capa->gated);
            }
            symbex_gated_forward(current_input, current_output, active_blocks_buffer, capa->gated);
        }

        // Ping-pong entre capas intermedias
        current_input = current_output;
        current_output = (current_output == buffer_a) ? buffer_b : buffer_a;
    }

    return 0; // No deberia llegar aca -- la ultima capa siempre retorna arriba
}
