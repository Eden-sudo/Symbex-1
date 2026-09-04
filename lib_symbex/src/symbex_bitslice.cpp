#include "symbex_bitslice.h"

// Memory read macro selection based on microcontroller architecture
#if defined(__AVR__)
    #include <avr/pgmspace.h>
    #define SYMBEX_READ_W32(addr) pgm_read_dword(addr)
#else
    #define SYMBEX_READ_W32(addr) (*(const uint32_t*)(addr))
#endif

// Evaluation order flag: 1 = Plane 0 is the Most Significant Bit (MSB)
#define SYMBEX_PLANE_0_IS_MSB 1

void symbex_forward_bitslice(const uint32_t* input_planes, int32_t* output_buffer, const struct SymbexLayerData* layer) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->weights;
    
    for (uint16_t n = 0; n < layer->out_features; n++) {
        int32_t neuron_score = 0;
        
        // External branching to avoid branch prediction penalties in the inner loop
        switch (layer->k_bits) {
            case 1:
                // High-speed fast-path for pure binary inference (1-bit)
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
                // Dynamic fallback for arbitrary quantizations
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
        
        output_buffer[n] = neuron_score;
    }
}

int symbex_argmax_bitslice(const uint32_t* input_planes, const struct SymbexLayerData* layer) {
    uint16_t in_words = (layer->in_features + 31) / 32;
    const uint32_t* w_ptr = layer->weights;
    
    int best_class = 0;
    int32_t max_score = -2147483647 - 1;  

    for (uint16_t n = 0; n < layer->out_features; n++) {
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
        
        // Winning class record update
        if (neuron_score > max_score) {
            max_score = neuron_score;
            best_class = n;
        }
    }
    
    return best_class;
}
