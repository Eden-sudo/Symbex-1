#include "../include/SymbexNetwork.h"

// ---------------------------------------------------------
// Implementación de SymbexNetwork
// ---------------------------------------------------------
SymbexNetwork::SymbexNetwork() {
    total_layers = 0;
    for (uint8_t i = 0; i < MAX_LAYERS; i++) {
        layers[i] = nullptr;
    }
}

bool SymbexNetwork::add_layer(SymbexLayer* layer) {
    // Agregamos la capa solo si no excedemos el límite estricto de memoria
    if (total_layers < MAX_LAYERS) {
        layers[total_layers] = layer;
        total_layers++;
        return true;
    }
    return false; 
}

uint8_t SymbexNetwork::predict(const uint8_t* sensor_input) {
    if (total_layers == 0) return 0; // Red vacía

    // Buffers locales estáticos (Ping-Pong).
    // Soportan capas ocultas de hasta 256 neuronas (32 bytes)
    // Esto evita usar malloc() en el microcontrolador.
    uint8_t buffer_A[32] = {0}; 
    uint8_t buffer_B[32] = {0};

    const uint8_t* current_input = sensor_input;
    uint8_t* current_output = buffer_A;

    for (uint8_t i = 0; i < total_layers; i++) {
        // Ejecutamos la matemática pesada de la capa actual
        layers[i]->process_layer(current_input, current_output);
        
        // La salida de esta capa se convierte en la entrada de la siguiente
        current_input = current_output;
        
        // Alternamos el rol de los buffers para la siguiente iteración
        if (current_output == buffer_A) {
            current_output = buffer_B;
        } else {
            current_output = buffer_A;
        }
    }

    // El resultado final es el primer byte del último buffer modificado.
    // Este byte es el que leerá la tabla LUT de hardware.
    return current_input[0]; 
}
