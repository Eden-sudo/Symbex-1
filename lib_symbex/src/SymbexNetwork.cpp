#include "SymbexNetwork.h"

SymbexNetwork::SymbexNetwork() {
    layer_count = 0;
}

bool SymbexNetwork::add_layer(SymbexLayer* layer) {
    if (layer_count >= MAX_LAYERS) return false;
    layers[layer_count++] = layer;
    return true;
}

int SymbexNetwork::classify(const uint8_t* input) {
    if (layer_count == 0) return -1;

    const uint8_t* current_input = input;
    uint8_t* current_output = buffer_A;

    // Procesar todas las capas ocultas usando el Ping-Pong Buffer
    for (uint8_t i = 0; i < layer_count - 1; i++) {
        // Usamos el nuevo método forward()
        layers[i]->forward(current_input, current_output);
        
        current_input = current_output;
        if (current_output == buffer_A) {
            current_output = buffer_B;
        } else {
            current_output = buffer_A;
        }
    }

    // La capa final utiliza el nuevo método argmax() de forma segura y directa
    return layers[layer_count - 1]->argmax(current_input);
}
