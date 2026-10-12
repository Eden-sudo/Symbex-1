/*
 * symbex_gated.c -- implementacion de la capa gated (mixture of
 * experts, activacion condicional por bloques).
 *
 * CAMBIO DE MIGRACION: mismo cambio que symbex_dense -- cada plano
 * de bits se procesa con UNA llamada a symbex_xnor_popcount (PIE)
 * en vez de palabra por palabra con el viejo SYMBEX_BIT_MAC.
 */

#include "layers/vector/symbex_gated.h"
#include "hal/symbex_hal.h"
#include "core/symbex_core.h"
#include <string.h>

#define SYMBEX_PLANE_0_IS_MSB 1
#define SYMBEX_MAX_BLOCKS 64

// Misma logica de puntuacion que symbex_dense -- se duplica aca (en
// vez de compartirla) porque gated tiene su propio manejo de avance
// de puntero por bloque; se puede unificar mas adelante si hace falta.
static int32_t symbex_gated_score_neuron(const uint32_t* input_planes, const uint32_t** w_ptr_io, uint16_t in_words, uint16_t k_bits) {
    const uint32_t* w_ptr = *w_ptr_io;
    int32_t score = 0;

    switch (k_bits) {
        case 1:
            score += symbex_xnor_popcount(input_planes, w_ptr, in_words, in_words * 32);
            w_ptr += in_words;
            break;

#if SYMBEX_PLANE_0_IS_MSB
        case 8:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, in_words * 32) << 7; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, in_words * 32) << 6; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, in_words * 32) << 5; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, in_words * 32) << 4; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[4 * in_words], w_ptr, in_words, in_words * 32) << 3; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[5 * in_words], w_ptr, in_words, in_words * 32) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[6 * in_words], w_ptr, in_words, in_words * 32) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[7 * in_words], w_ptr, in_words, in_words * 32) << 0; w_ptr += in_words;
            break;
        case 4:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, in_words * 32) << 3; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, in_words * 32) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, in_words * 32) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, in_words * 32) << 0; w_ptr += in_words;
            break;
        default:
            for (uint16_t k = 0; k < k_bits; k++) {
                score += symbex_xnor_popcount(&input_planes[k * in_words], w_ptr, in_words, in_words * 32) << (k_bits - 1 - k);
                w_ptr += in_words;
            }
            break;
#else
        case 8:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, in_words * 32) << 0; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, in_words * 32) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, in_words * 32) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, in_words * 32) << 3; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[4 * in_words], w_ptr, in_words, in_words * 32) << 4; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[5 * in_words], w_ptr, in_words, in_words * 32) << 5; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[6 * in_words], w_ptr, in_words, in_words * 32) << 6; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[7 * in_words], w_ptr, in_words, in_words * 32) << 7; w_ptr += in_words;
            break;
        case 4:
            score += symbex_xnor_popcount(&input_planes[0 * in_words], w_ptr, in_words, in_words * 32) << 0; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[1 * in_words], w_ptr, in_words, in_words * 32) << 1; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[2 * in_words], w_ptr, in_words, in_words * 32) << 2; w_ptr += in_words;
            score += symbex_xnor_popcount(&input_planes[3 * in_words], w_ptr, in_words, in_words * 32) << 3; w_ptr += in_words;
            break;
        default:
            for (uint16_t k = 0; k < k_bits; k++) {
                score += symbex_xnor_popcount(&input_planes[k * in_words], w_ptr, in_words, in_words * 32) << k;
                w_ptr += in_words;
            }
            break;
#endif
    }

    *w_ptr_io = w_ptr;
    return score;
}

void symbex_gated_evaluate(
    const uint32_t* input_planes,
    bool* active_blocks,
    const struct SymbexGatedLayerData* layer,
    uint16_t k_active
) {
    uint16_t num_blocks = layer->out_features / layer->block_size;
    if (num_blocks > SYMBEX_MAX_BLOCKS) num_blocks = SYMBEX_MAX_BLOCKS;

    uint16_t in_words = (layer->in_features + 31) / 32;
    int16_t gate_scores[SYMBEX_MAX_BLOCKS];
    uint8_t block_indices[SYMBEX_MAX_BLOCKS];

    const uint32_t* w_ptr = layer->gate_weights;

    // Evaluacion del director: un puntaje por bloque
    for (uint16_t b = 0; b < num_blocks; b++) {
        gate_scores[b] = (int16_t)symbex_xnor_popcount(input_planes, w_ptr, in_words, in_words * 32);
        w_ptr += in_words;
        block_indices[b] = (uint8_t)b;
        active_blocks[b] = false;
    }

    // Insertion sort para seleccionar los Top-K bloques
    for (uint16_t i = 1; i < num_blocks; i++) {
        uint8_t key_idx = block_indices[i];
        int16_t key_val = gate_scores[key_idx];
        int16_t j = i - 1;
        while (j >= 0 && gate_scores[block_indices[j]] < key_val) {
            block_indices[j + 1] = block_indices[j];
            j--;
        }
        block_indices[j + 1] = key_idx;
    }

    // Activacion logica de los bloques seleccionados
    for (uint16_t i = 0; i < k_active; i++) {
        if (i < num_blocks) {
            active_blocks[block_indices[i]] = true;
        }
    }
}

void symbex_gated_forward(
    const uint32_t* input_planes,
    uint32_t* output_buffer,
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->core_weights;
    uint32_t weights_per_neuron = (uint32_t)layer->k_bits * in_words;

    uint16_t out_words = (layer->out_features + 31) / 32;
    memset(output_buffer, 0, out_words * sizeof(uint32_t));
    uint8_t* out_bytes = (uint8_t*)output_buffer;

    for (uint16_t n = 0; n < layer->out_features; n++) {
        uint16_t block_id = n / layer->block_size;

        // Salto condicional de memoria para bloques inactivos
        if (!active_blocks[block_id]) {
            w_ptr += weights_per_neuron;
            continue;
        }

        int32_t neuron_score = symbex_gated_score_neuron(input_planes, &w_ptr, in_words, layer->k_bits);

        // Empaquetado bit a bit (bipolar STE: puntaje > 0 -> 1)
        if (neuron_score > 0) {
            out_bytes[n / 8] |= (1 << (7 - (n % 8)));
        }
    }
}

int symbex_gated_argmax(
    const uint32_t* input_planes,
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->core_weights;
    uint32_t weights_per_neuron = (uint32_t)layer->k_bits * in_words;

    int best_class = 0;
    int32_t max_score = -2147483647 - 1;

    for (uint16_t n = 0; n < layer->out_features; n++) {
        uint16_t block_id = n / layer->block_size;

        if (!active_blocks[block_id]) {
            w_ptr += weights_per_neuron;
            continue;
        }

        int32_t neuron_score = symbex_gated_score_neuron(input_planes, &w_ptr, in_words, layer->k_bits);

        if (neuron_score > max_score) {
            max_score = neuron_score;
            best_class = n;
        }
    }

    return best_class;
}
