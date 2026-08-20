#include "../include/Symbex1.h"
#include <iostream>
#include <bitset>

// ---------------------------------------------------------
// Constructor e Inicialización
// ---------------------------------------------------------
SymbexEngine::SymbexEngine() {
    // Inicializamos el puntero en nulo por seguridad
    hidden_layer = nullptr; 
}

void SymbexEngine::load_network(const SymbexLayer* layer_config) {
    hidden_layer = layer_config;
}

// ---------------------------------------------------------
// Helpers Matemáticos (Ocultos al usuario)
// ---------------------------------------------------------

// Algoritmo de Brian Kernighan para contar bits rápidos
// Solo itera la cantidad de veces equivalente a los bits en '1'
inline uint8_t popcount8(uint8_t n) {
    uint8_t count = 0;
    while (n) {
        count++;
        n &= (n - 1); // Apaga el bit menos significativo que esté encendido
    }
    return count;
}

int8_t SymbexEngine::compute_mac_1bit(uint8_t input_bits, uint8_t weight_bits) {
    uint8_t xnor_result = ~(input_bits ^ weight_bits);
    uint8_t matches = popcount8(xnor_result);
    int8_t mac_value = (int8_t)(2 * matches) - 8;

// --- MODO VERBOSE ---
#ifdef SYMBEX_VERBOSE
    std::cout << "    [MAC] IN: " << std::bitset<8>(input_bits) 
              << " | W: " << std::bitset<8>(weight_bits) 
              << " -> XNOR: " << std::bitset<8>(xnor_result) 
              << " | Matches: " << (int)matches 
              << " | Val: " << (int)mac_value << "\n";
#endif
// --------------------

    return mac_value;
}

// ---------------------------------------------------------
// Motor de Inferencia Principal
// ---------------------------------------------------------
uint8_t SymbexEngine::predict(const uint8_t* input_state) {
    // Si no hay red cargada, devolvemos 0 por seguridad
    if (!hidden_layer) return 0; 

    uint8_t output_byte = 0; // Aquí colapsaremos el resultado final
    
    // Calculamos cuántos bytes componen la entrada
    uint16_t total_input_bytes = hidden_layer->num_inputs / 8;

    // Iteramos sobre cada neurona de salida (hasta un máximo de 8 neuronas 
    // para formar un byte perfecto para el diccionario LUT)
    for (uint8_t neuron = 0; (neuron < hidden_layer->num_neurons) && (neuron < 8); neuron++) {
        
        int16_t accumulator = 0; // Acumulador de alta precisión

        // Procesamos todas las conexiones de entrada hacia esta neurona
        for (uint16_t i = 0; i < total_input_bytes; i++) {
            uint8_t in_bits = input_state[i];
            
            // Calculamos la posición exacta en el arreglo lineal de la memoria
            uint16_t weight_idx = (neuron * total_input_bytes) + i;

            // Extraemos los cálculos de las 3 capas superpuestas
            int8_t sum_msb = compute_mac_1bit(in_bits, hidden_layer->weights_msb[weight_idx]);
            int8_t sum_mid = compute_mac_1bit(in_bits, hidden_layer->weights_mid[weight_idx]);
            int8_t sum_lsb = compute_mac_1bit(in_bits, hidden_layer->weights_lsb[weight_idx]);

            // MAGIA DE BIT-SLICING: 
            // Reconstruimos el valor sin usar ni una sola multiplicación decimal.
            // MSB vale x4 (<<2), MID vale x2 (<<1), LSB vale x1.
            accumulator += (sum_msb << 2) + (sum_mid << 1) + sum_lsb;
        }

        // Función de activación final (Thresholding)
        // Si la energía acumulada supera el umbral de la neurona, la disparamos
        if (accumulator >= hidden_layer->thresholds[neuron]) {
            output_byte |= (1 << neuron); // Encendemos el bit correspondiente
        }
    }

    // Retornamos el "código de barras" de 8 bits listo para el mundo físico
    return output_byte;
}
