#include "SymbexNetwork.h"

//===================================
// CONSTRUCTOR Y GESTIÓN DE TOPOLOGÍA
//===================================

SymbexNetwork::SymbexNetwork() {
    // Inicializa el contador en cero para garantizar un pipeline limpio al inicio.
    layer_count = 0;
}

bool SymbexNetwork::add_layer(SymbexLayer* layer) {
    // Mecanismo de seguridad: Evita desbordamiento de memoria (Buffer Overflow) 
    // bloqueando la adición si se supera el límite físico del arreglo de capas.
    if (layer_count >= MAX_LAYERS) return false;
    
    // Registra el puntero de la capa y avanza el índice.
    layers[layer_count++] = layer;
    return true;
}

//=====================================
// ORQUESTADOR DE INFERENCIA SECUENCIAL
//=====================================

int SymbexNetwork::classify(const uint8_t* input) {
    // Abortar si la red está vacía (evita accesos a memoria nula).
    if (layer_count == 0) return -1;

    // Inicialización de punteros para el Ping-Pong Buffering.
    // Evita asignar memoria dinámica en tiempo de ejecución.
    const uint8_t* current_input = input;
    uint8_t* current_output = buffer_A;

    // 1. PROPAGACIÓN EN CAPAS OCULTAS
    // Itera hasta la penúltima capa. La última capa se procesa distinto (argmax).
    for (uint8_t i = 0; i < layer_count - 1; i++) {
        
        // Ejecuta la propagación matemática y binariza el resultado en current_output.
        layers[i]->forward(current_input, current_output);
        
        // Intercambio de buffers (Ping-Pong): 
        // La salida actual se convierte en la entrada de la siguiente capa.
        current_input = current_output;
        
        // Alternar el puntero de salida para reciclar el espacio de memoria (SRAM).
        if (current_output == buffer_A) {
            current_output = buffer_B;
        } else {
            current_output = buffer_A;
        }
    }

    // 2. CAPA FINAL (DECISIÓN ARITMÉTICA)
    // La capa de salida evalúa el último buffer y retorna el ID ganador (argmax),
    // prescindiendo de generar un nuevo arreglo de bits.
    return layers[layer_count - 1]->argmax(current_input);
}

