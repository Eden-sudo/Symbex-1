#include "../include/SymbexNetwork.h"

// ---------------------------------------------------------
// Helpers Matemáticos (Privados al archivo)
// ---------------------------------------------------------
inline uint8_t popcount8_layer(uint8_t n) {
    uint8_t count = 0;
    while (n) {
        count++;
        n &= (n - 1);
    }
    return count;
}

inline int8_t compute_mac(uint8_t in_bits, uint8_t w_bits) {
    uint8_t xnor_res = ~(in_bits ^ w_bits);
    return (int8_t)(2 * popcount8_layer(xnor_res)) - 8;
}

// ---------------------------------------------------------
// Implementación de SymbexLayer
// ---------------------------------------------------------
SymbexLayer::SymbexLayer(uint16_t inputs, uint16_t neurons, 
                         const uint8_t* msb, const uint8_t* mid, const uint8_t* lsb, 
                         const int16_t* th) {
    num_inputs = inputs;
    num_neurons = neurons;
    weights_msb = msb;
    weights_mid = mid;
    weights_lsb = lsb;
    thresholds = th;
}

void SymbexLayer::process_layer(const uint8_t* input_state, uint8_t* output_state) {
    uint16_t total_input_bytes = num_inputs / 8;
    uint16_t total_output_bytes = (num_neurons + 7) / 8; // Redondeo hacia arriba

    // Limpiamos el estado de salida para evitar basura en la memoria
    for (uint16_t b = 0; b < total_output_bytes; b++) {
        output_state[b] = 0;
    }

    for (uint16_t neuron = 0; neuron < num_neurons; neuron++) {
        int16_t accumulator = 0;

        for (uint16_t i = 0; i < total_input_bytes; i++) {
            uint8_t in_bits = input_state[i];
            uint16_t weight_idx = (neuron * total_input_bytes) + i;

            int8_t sum_msb = compute_mac(in_bits, weights_msb[weight_idx]);
            int8_t sum_mid = compute_mac(in_bits, weights_mid[weight_idx]);
            int8_t sum_lsb = compute_mac(in_bits, weights_lsb[weight_idx]);

            // Reconstrucción del Bit-Slicing (K=3)
            accumulator += (sum_msb << 2) + (sum_mid << 1) + sum_lsb;
        }

        // Activación: Si supera el umbral, encendemos el bit en el arreglo de salida
        if (accumulator >= thresholds[neuron]) {
            output_state[neuron / 8] |= (1 << (neuron % 8));
        }
    }
}
