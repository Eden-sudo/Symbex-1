#ifndef SYMBEX_NETWORK_H
#define SYMBEX_NETWORK_H

#include <stdint.h>
#include "SymbexLayer.h"

// Límites estáticos diseñados para microcontroladores (Ajustables según necesidad)
#define MAX_LAYERS 4
#define BUFFER_SIZE 32 // Suficiente para redes embebidas, sin usar RAM dinámica

class SymbexNetwork {
private:
    SymbexLayer* layers[MAX_LAYERS];
    int layer_count;
    
    // Ping-Pong Buffers: Alternan memoria para evitar fragmentación de la SRAM
    uint8_t buffer_A[BUFFER_SIZE];
    uint8_t buffer_B[BUFFER_SIZE];

public:
    // Constructor
    SymbexNetwork();

    // Orquestación
    bool add_layer(SymbexLayer* layer);

    // Motor de Inferencia (Devuelve un byte crudo o 'símbolo base')
    uint8_t predict(const uint8_t* input);

    // Motor de Clasificación (Sin M_factor externo)
    int classify(const uint8_t* input);
};

#endif // SYMBEX_NETWORK_H
