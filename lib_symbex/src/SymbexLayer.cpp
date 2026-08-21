#include "SymbexLayer.h"
#include <string.h>

static inline uint8_t symbex_popcount(uint8_t val) {
    uint8_t count = 0;
    for (uint8_t i = 0; i < 8; i++) {
        if ((val >> i) & 1) count++;
    }
    return count;
}

SymbexLayer::SymbexLayer(uint16_t in_f, uint16_t out_f, uint8_t m, uint8_t k, const SymbexSubLayer* subs) {
    in_features = in_f;
    out_features = out_f;
    M = m;
    k_bits = k;
    sub_layers = subs;
}

void SymbexLayer::forward(const uint8_t* input_buffer, uint8_t* output_buffer) {
    // Limpiamos el buffer de salida porque lo llenaremos con OR bit a bit
    memset(output_buffer, 0, (out_features + 7) / 8);

    // Iteramos por neurona de salida (Seguro para el Stack SRAM)
    for (uint16_t n = 0; n < out_features; n++) {
        int32_t total_votes = 0;

        for (uint8_t m = 0; m < M; m++) {
            const SymbexSubLayer& sub = sub_layers[m];
            int32_t accumulator = 0;

            for (uint16_t i = 0; i < in_features; i += 8) {
                uint8_t in_byte = input_buffer[i / 8];
                uint16_t weight_idx = (n * ((in_features + 7) / 8)) + (i / 8);
                
                // Planos de bits dinámicos
                for (uint8_t k = 0; k < k_bits; k++) {
                    uint8_t w = SYMBEX_READ_BYTE(&sub.bit_planes[k][weight_idx]);
                    uint8_t xnor_val = ~(in_byte ^ w);
                    int8_t scale = 1 << (k_bits - 1 - k);
                    accumulator += scale * ((2 * symbex_popcount(xnor_val)) - 8);
                }
                
                // Outliers con signo corregido mediante MSB (w0)
                uint8_t outl_mask = SYMBEX_READ_BYTE(&sub.outliers[weight_idx]);
                if (outl_mask != 0) {
                    uint8_t w0 = SYMBEX_READ_BYTE(&sub.bit_planes[0][weight_idx]);
                    for (uint8_t bit = 0; bit < 8; bit++) {
                        if ((outl_mask >> (7 - bit)) & 1) {
                            int8_t mag = (int8_t)SYMBEX_READ_BYTE(&sub.outlier_mag[n]);
                            uint8_t weight_bit = (w0 >> (7 - bit)) & 1;
                            uint8_t input_bit  = (in_byte >> (7 - bit)) & 1;
                            int8_t sign = (weight_bit == input_bit) ? 1 : -1;
                            accumulator += sign * mag;
                        }
                    }
                }
            }
            total_votes += accumulator;
        }

        // Binarización directa (solo activamos el bit si es positivo)
        if (total_votes > 0) {
            uint8_t byte_idx = n / 8;
            uint8_t bit_idx = 7 - (n % 8);
            output_buffer[byte_idx] |= (1 << bit_idx);
        }
    }
}

int SymbexLayer::argmax(const uint8_t* input_buffer) {
    int best_class = 0;
    // Iniciamos con el valor más bajo posible para enteros de 32 bits
    int32_t max_votes = -2147483647 - 1; 

    // Misma estructura segura, pero guardamos el campeón en lugar de binarizar
    for (uint16_t n = 0; n < out_features; n++) {
        int32_t total_votes = 0;

        for (uint8_t m = 0; m < M; m++) {
            const SymbexSubLayer& sub = sub_layers[m];
            int32_t accumulator = 0;

            for (uint16_t i = 0; i < in_features; i += 8) {
                uint8_t in_byte = input_buffer[i / 8];
                uint16_t weight_idx = (n * ((in_features + 7) / 8)) + (i / 8);
                
                for (uint8_t k = 0; k < k_bits; k++) {
                    uint8_t w = SYMBEX_READ_BYTE(&sub.bit_planes[k][weight_idx]);
                    uint8_t xnor_val = ~(in_byte ^ w);
                    int8_t scale = 1 << (k_bits - 1 - k);
                    accumulator += scale * ((2 * symbex_popcount(xnor_val)) - 8);
                }
                
                uint8_t outl_mask = SYMBEX_READ_BYTE(&sub.outliers[weight_idx]);
                if (outl_mask != 0) {
                    uint8_t w0 = SYMBEX_READ_BYTE(&sub.bit_planes[0][weight_idx]);
                    for (uint8_t bit = 0; bit < 8; bit++) {
                        if ((outl_mask >> (7 - bit)) & 1) {
                            int8_t mag = (int8_t)SYMBEX_READ_BYTE(&sub.outlier_mag[n]);
                            uint8_t weight_bit = (w0 >> (7 - bit)) & 1;
                            uint8_t input_bit  = (in_byte >> (7 - bit)) & 1;
                            int8_t sign = (weight_bit == input_bit) ? 1 : -1;
                            accumulator += sign * mag;
                        }
                    }
                }
            }
            total_votes += accumulator;
        }

        // ¿Superó al campeón actual?
        if (total_votes > max_votes) {
            max_votes = total_votes;
            best_class = n;
        }
    }
    return best_class;
}
