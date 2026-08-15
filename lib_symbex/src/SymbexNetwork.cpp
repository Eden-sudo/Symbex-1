#include "../include/SymbexNetwork.h"

// ------------------------------------------------------------------
// Constructor: Inicializa la red de forma segura
// ------------------------------------------------------------------
SymbexNetwork::SymbexNetwork() {
    layer_count = 0;
    for(int i = 0; i < MAX_LAYERS; i++) {
        layers[i] = nullptr; // Limpieza de punteros por seguridad
    }
}

// ------------------------------------------------------------------
// add_layer: Conecta una nueva capa a la arquitectura secuencial
// ------------------------------------------------------------------
bool SymbexNetwork::add_layer(SymbexLayer* layer) {
    if (layer_count < MAX_LAYERS) {
        layers[layer_count] = layer;
        layer_count++;
        return true;
    }
    return false; // Retorna falso si se supera el límite de memoria estática
}

// ------------------------------------------------------------------
// predict: Ejecuta el flujo de datos usando Ping-Pong Buffering
// ------------------------------------------------------------------
uint8_t SymbexNetwork::predict(const uint8_t* input) {
    if (layer_count == 0) return 0; // Red vacía

    const uint8_t* current_input = input;
    uint8_t* current_output = buffer_A;

    // La información rebota entre el buffer_A y el buffer_B en cada capa
    for (int i = 0; i < layer_count; i++) {
        // La capa matemática hace el trabajo pesado (XNOR, <<, POPCOUNT)
        layers[i]->process_layer(current_input, current_output);
        
        // El resultado actual se convierte en la entrada de la siguiente iteración
        current_input = current_output;
        
        // Intercambio de buffers (Ping-Pong)
        if (current_output == buffer_A) {
            current_output = buffer_B;
        } else {
            current_output = buffer_A;
        }
    }

    // Como la red colapsa en un solo byte (Clasificación/Símbolo),
    // retornamos el primer índice del buffer que actuó como última salida.
    return current_input[0];
}

// ------------------------------------------------------------------
// generate_trajectory: El generador en cadena (Expansión Simbólica)
// ------------------------------------------------------------------
void SymbexNetwork::generate_trajectory(uint8_t* state_buffer, int input_size_bytes, int steps, uint8_t bit_mask, uint8_t* output_trajectory) {
    
    for(int i = 0; i < steps; i++) {
        // 1. Inferencia: La red evalúa el estado actual y genera el "símbolo bruto"
        uint8_t raw_symbol = this->predict(state_buffer);
        
        // 2. Expansión Simbólica: Se aplica la máscara lógica (Decodificación de Hardware)
        // Usamos XOR (^) como ejemplo matemático, pero puede adaptarse a reglas específicas de hardware
        uint8_t complex_action = raw_symbol ^ bit_mask; 
        
        // 3. Almacenamiento: Guardamos la decisión decodificada en la trayectoria final
        output_trajectory[i] = complex_action;
        
        // 4. Retroalimentación (Sliding Window): Actualizamos el estado interno
        // Movemos todos los bytes históricos un bloque a la izquierda (el más viejo se descarta)
        for(int j = 0; j < input_size_bytes - 1; j++) {
            state_buffer[j] = state_buffer[j + 1];
        }
        
        // Inyectamos la última acción generada en la ranura más reciente del buffer
        state_buffer[input_size_bytes - 1] = complex_action;
    }
}
