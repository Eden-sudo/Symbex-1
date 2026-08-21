#ifndef SYMBEX_NETWORK_H
#define SYMBEX_NETWORK_H

#include <stdint.h>
#include "SymbexLayer.h"

// Definimos los límites de memoria estática (SRAM) para los microcontroladores
#define MAX_LAYERS 8
#define BUFFER_SIZE 128

class SymbexNetwork {
private:
    SymbexLayer* layers[MAX_LAYERS];
    uint8_t layer_count;
    
    // Buffers de propagación (Ping-Pong) para Feed-Forward
    uint8_t buffer_A[BUFFER_SIZE];
    uint8_t buffer_B[BUFFER_SIZE];
    
    // Buffer preparado para el futuro modo autorregresivo (Generación de secuencias)
    uint8_t state_buffer[BUFFER_SIZE]; 

public:
    SymbexNetwork();
    
    bool add_layer(SymbexLayer* layer);
    
    // Inferencia clásica
    int classify(const uint8_t* input);
};

#endif // SYMBEX_NETWORK_H
