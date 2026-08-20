#include "SymbexNetwork.h"

SymbexNetwork::SymbexNetwork() {
    layer_count = 0;
}

bool SymbexNetwork::add_layer(SymbexLayer* layer) {
    if (layer_count >= MAX_LAYERS) return false;
    layers[layer_count++] = layer;
    return true;
}

uint8_t SymbexNetwork::predict(const uint8_t* input) {
    if (layer_count == 0) return 0;
    
    const uint8_t* current_input = input;
    uint8_t* current_output = buffer_A;

    for (int i = 0; i < layer_count; i++) {
        layers[i]->process_layer(current_input, current_output);
        current_input = current_output;
        current_output = (current_output == buffer_A) ? buffer_B : buffer_A;
    }
    
    return current_input[0];
}

int SymbexNetwork::classify(const uint8_t* input) {
    if (layer_count == 0) return 0;
    
    const uint8_t* current_input = input;
    uint8_t* current_output = buffer_A;

    // Procesar todas las capas ocultas usando el Ping-Pong Buffer
    for (int i = 0; i < layer_count - 1; i++) {
        layers[i]->process_layer(current_input, current_output);
        current_input = current_output;
        current_output = (current_output == buffer_A) ? buffer_B : buffer_A;
    }

    // La capa final utiliza process_layer_argmax delegando el factor M internamente
    return layers[layer_count - 1]->process_layer_argmax(current_input);
}
