#include "SymbexLayer.h"

SymbexLayer::SymbexLayer(uint16_t in, uint16_t out, uint8_t M, const SymbexSubLayer* subs) {
    in_features = in;
    out_features = out;
    M_factor = M;
    sub_layers = subs;
}

void SymbexLayer::process_layer(const uint8_t* input, uint8_t* output) {
    uint16_t input_bytes = (in_features + 7) / 8;
    uint16_t output_bytes = (out_features + 7) / 8;
    
    for (uint16_t i = 0; i < output_bytes; i++) {
        output[i] = 0;
    }

    for (uint16_t n = 0; n < out_features; n++) {
        int32_t total_votes = 0;

        for (uint8_t m = 0; m < M_factor; m++) {
            const SymbexSubLayer& sub = sub_layers[m];
            int32_t accumulator = 0;

            for (uint16_t b = 0; b < input_bytes; b++) {
                uint16_t weight_idx = n * input_bytes + b;
                uint8_t in_byte = input[b];

                uint8_t w0 = SYMBEX_READ_BYTE(&sub.weight_bit0[weight_idx]);
                uint8_t w1 = SYMBEX_READ_BYTE(&sub.weight_bit1[weight_idx]);
                uint8_t w2 = SYMBEX_READ_BYTE(&sub.weight_bit2[weight_idx]);
                uint8_t outl = SYMBEX_READ_BYTE(&sub.outliers[weight_idx]);

                // w0 es el MSB (Peso 4)
                uint8_t xnor0 = ~(in_byte ^ w0);
                accumulator += 4 * ((2 * symbex_popcount(xnor0)) - 8);

                // w1 es el MID (Peso 2)
                uint8_t xnor1 = ~(in_byte ^ w1);
                accumulator += 2 * ((2 * symbex_popcount(xnor1)) - 8);

                // w2 es el LSB (Peso 1)
                uint8_t xnor2 = ~(in_byte ^ w2);
                accumulator += 1 * ((2 * symbex_popcount(xnor2)) - 8);

                if (outl) {
                    // El signo del outlier es el MSB (w0). Filtramos con la máscara outl.
                    uint8_t match_outl = ~(in_byte ^ w0) & outl;
                    // Mismos votos bipolares: 2 * (coincidencias) - (total de outliers activos en el byte)
                    int8_t outl_contrib = (2 * symbex_popcount(match_outl)) - symbex_popcount(outl);
                    accumulator += outl_contrib * (int8_t)SYMBEX_READ_BYTE(&sub.outlier_magnitudes[n]);
                }
            }
            total_votes += accumulator;
        }

        if (total_votes > 0) {
            output[n / 8] |= (1 << (7 - (n % 8)));
        }
    }
}

int SymbexLayer::process_layer_argmax(const uint8_t* input) {
    uint16_t input_bytes = (in_features + 7) / 8;
    int max_vote = -2147483647; 
    int best_class = 0;

    for (uint16_t n = 0; n < out_features; n++) {
        int32_t total_votes = 0;

        for (uint8_t m = 0; m < M_factor; m++) {
            const SymbexSubLayer& sub = sub_layers[m];
            int32_t accumulator = 0;

            for (uint16_t b = 0; b < input_bytes; b++) {
                uint16_t weight_idx = n * input_bytes + b;
                uint8_t in_byte = input[b];

                uint8_t w0 = SYMBEX_READ_BYTE(&sub.weight_bit0[weight_idx]);
                uint8_t w1 = SYMBEX_READ_BYTE(&sub.weight_bit1[weight_idx]);
                uint8_t w2 = SYMBEX_READ_BYTE(&sub.weight_bit2[weight_idx]);
                uint8_t outl = SYMBEX_READ_BYTE(&sub.outliers[weight_idx]);

                uint8_t xnor0 = ~(in_byte ^ w0);
                accumulator += 4 * ((2 * symbex_popcount(xnor0)) - 8);

                uint8_t xnor1 = ~(in_byte ^ w1);
                accumulator += 2 * ((2 * symbex_popcount(xnor1)) - 8);

                uint8_t xnor2 = ~(in_byte ^ w2);
                accumulator += 1 * ((2 * symbex_popcount(xnor2)) - 8);

                if (outl) {
                    uint8_t match_outl = ~(in_byte ^ w0) & outl;
                    int8_t outl_contrib = (2 * symbex_popcount(match_outl)) - symbex_popcount(outl);
                    accumulator += outl_contrib * (int8_t)SYMBEX_READ_BYTE(&sub.outlier_magnitudes[n]);
                }
            }
            total_votes += accumulator;
        }

        if (total_votes > max_vote) {
            max_vote = total_votes;
            best_class = n;
        }
    }
    return best_class;
}
