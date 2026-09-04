#include "symbex_gated.h"
#include <string.h>

#if defined(__AVR__)
    #include <avr/pgmspace.h>
    #define SYMBEX_READ_W32(addr) pgm_read_dword(addr)
#else
    #define SYMBEX_READ_W32(addr) (*(const uint32_t*)(addr))
#endif

#define SYMBEX_PLANE_0_IS_MSB 1
#define SYMBEX_MAX_BLOCKS 64 

void symbex_evaluate_gates(
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

    // Gate evaluation (Director)
    for (uint16_t b = 0; b < num_blocks; b++) {
        int32_t score = 0;
        for (uint16_t w = 0; w < in_words; w++) {
            score += SYMBEX_BIT_MAC(input_planes[w], SYMBEX_READ_W32(w_ptr++));
        }
        gate_scores[b] = score;
        block_indices[b] = b;
        active_blocks[b] = false; 
    }

    // Insertion sort for Top-K block selection
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

    // Logical activation of selected blocks
    for (uint16_t i = 0; i < k_active; i++) {
        if (i < num_blocks) {
            active_blocks[block_indices[i]] = true;
        }
    }
}

void symbex_forward_gated(
    const uint32_t* input_planes, 
    uint32_t* output_buffer, 
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->core_weights;
    uint32_t weights_per_neuron = layer->k_bits * in_words; 
    
    // Buffer initialization and byte alignment
    uint16_t out_words = (layer->out_features + 31) / 32;
    memset(output_buffer, 0, out_words * sizeof(uint32_t));
    uint8_t* out_bytes = (uint8_t*)output_buffer;

    for (uint16_t n = 0; n < layer->out_features; n++) {
        uint16_t block_id = n / layer->block_size;
        
        // Conditional memory skip for inactive blocks (Early Exit)
        if (!active_blocks[block_id]) {
            w_ptr += weights_per_neuron; 
            continue;
        }

        int32_t neuron_score = 0;
        
        switch (layer->k_bits) {
            case 1:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[w], SYMBEX_READ_W32(w_ptr++));
                }
                break;
                
#if SYMBEX_PLANE_0_IS_MSB
            case 8:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 7;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 6;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 5;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 4;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(4 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(5 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(6 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(7 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                }
                break;
            case 4:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                }
                break;
            default: 
                for (uint16_t w = 0; w < in_words; w++) {
                    for (uint16_t k = 0; k < layer->k_bits; k++) {
                        neuron_score += SYMBEX_BIT_MAC(input_planes[(k * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << (layer->k_bits - 1 - k);
                    }
                }
                break;
#else
            case 8:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(4 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 4;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(5 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 5;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(6 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 6;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(7 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 7;
                }
                break;
            case 4:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                }
                break;
            default: 
                for (uint16_t w = 0; w < in_words; w++) {
                    for (uint16_t k = 0; k < layer->k_bits; k++) {
                        neuron_score += SYMBEX_BIT_MAC(input_planes[(k * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << k;
                    }
                }
                break;
#endif
        }
        
        // Bit-level packing (Bipolar STE: x > 0 -> 1)
        if (neuron_score > 0) {
            out_bytes[n / 8] |= (1 << (7 - (n % 8)));
        }
    }
}

int symbex_argmax_gated(
    const uint32_t* input_planes, 
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->core_weights;
    uint32_t weights_per_neuron = layer->k_bits * in_words;
    
    int best_class = 0;
    int32_t max_score = -2147483647 - 1; 

    for (uint16_t n = 0; n < layer->out_features; n++) {
        uint16_t block_id = n / layer->block_size;
        
        if (!active_blocks[block_id]) {
            w_ptr += weights_per_neuron; 
            continue;
        }

        int32_t neuron_score = 0;
        
        switch (layer->k_bits) {
            case 1:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[w], SYMBEX_READ_W32(w_ptr++));
                }
                break;
                
#if SYMBEX_PLANE_0_IS_MSB
            case 8:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 7;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 6;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 5;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 4;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(4 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(5 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(6 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(7 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                }
                break;
            case 4:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                }
                break;
            default:
                for (uint16_t w = 0; w < in_words; w++) {
                    for (uint16_t k = 0; k < layer->k_bits; k++) {
                        neuron_score += SYMBEX_BIT_MAC(input_planes[(k * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << (layer->k_bits - 1 - k);
                    }
                }
                break;
#else
            case 8:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(4 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 4;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(5 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 5;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(6 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 6;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(7 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 7;
                }
                break;
            case 4:
                for (uint16_t w = 0; w < in_words; w++) {
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(0 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 0;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(1 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 1;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(2 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 2;
                    neuron_score += SYMBEX_BIT_MAC(input_planes[(3 * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << 3;
                }
                break;
            default:
                for (uint16_t w = 0; w < in_words; w++) {
                    for (uint16_t k = 0; k < layer->k_bits; k++) {
                        neuron_score += SYMBEX_BIT_MAC(input_planes[(k * in_words) + w], SYMBEX_READ_W32(w_ptr++)) << k;
                    }
                }
                break;
#endif
        }
        
        if (neuron_score > max_score) {
            max_score = neuron_score;
            best_class = n;
        }
    }
    
    return best_class;
}
