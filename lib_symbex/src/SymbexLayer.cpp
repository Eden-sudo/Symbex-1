#include "../include/SymbexLayer.h"

SymbexLayer::SymbexLayer(int in, int out, 
                         const uint8_t* msb, const uint8_t* mid, const uint8_t* lsb, 
                         const uint8_t* outl_mask, const int16_t* outl_mags, 
                         const int16_t* thresh) {
    in_features = in;
    out_features = out;
    weights_msb = msb;
    weights_mid = mid;
    weights_lsb = lsb;
    weights_outlier = outl_mask;
    outlier_magnitudes = outl_mags;
    thresholds = thresh;
}

static inline int count_set_bits(uint8_t n) {
    int count = 0;
    while (n) {
        n &= (n - 1);
        count++;
    }
    return count;
}

void SymbexLayer::process_layer(const uint8_t* input, uint8_t* output) {
    int input_bytes = in_features / 8;
    int output_bytes = (out_features + 7) / 8;
    
    for (int i = 0; i < output_bytes; i++) {
        output[i] = 0;
    }

    for (int n = 0; n < out_features; n++) {
        int16_t accumulator = 0;
        
        for (int b = 0; b < input_bytes; b++) {
            uint8_t in_val = input[b];
            int weight_idx = n * input_bytes + b;

            // 1. VIA PRINCIPAL (K=3 Normal)
            uint8_t xnor_msb = ~(in_val ^ weights_msb[weight_idx]);
            uint8_t xnor_mid = ~(in_val ^ weights_mid[weight_idx]);
            uint8_t xnor_lsb = ~(in_val ^ weights_lsb[weight_idx]);

            int16_t bipol_msb = (count_set_bits(xnor_msb) * 2) - 8;
            int16_t bipol_mid = (count_set_bits(xnor_mid) * 2) - 8;
            int16_t bipol_lsb = (count_set_bits(xnor_lsb) * 2) - 8;

            accumulator += (bipol_msb << 2) + (bipol_mid << 1) + bipol_lsb;
            
            // 2. VIA SECUNDARIA (Abstraccion de Outliers)
            // Evaluamos si los datos de entrada coinciden con los patrones atipicos
            uint8_t xnor_outl = ~(in_val ^ weights_outlier[weight_idx]);
            int outl_matches = count_set_bits(xnor_outl);
            
            // Si hay un nivel de coincidencia alto con el ruido atipico, inyectamos la magnitud
            if (outl_matches > 4) { 
                // Sumamos la fuerza del outlier a esta neurona
                accumulator += outlier_magnitudes[n]; 
            }
        }

        if (accumulator >= thresholds[n]) {
            output[n / 8] |= (1 << (7 - (n % 8))); 
        }
    }
}
