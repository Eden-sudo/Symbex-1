#ifndef SYMBEX_NETWORK_H
#define SYMBEX_NETWORK_H

#include <stdint.h>

// ---------------------------------------------------------
// CLASE 1: La Capa Individual
// ---------------------------------------------------------
class SymbexLayer {
public:
    uint16_t num_inputs;
    uint16_t num_neurons;
    
    // Las 3 matrices lógicas del Bit-Slicing
    const uint8_t* weights_msb;
    const uint8_t* weights_mid;
    const uint8_t* weights_lsb;
    const int16_t* thresholds;

    // Constructor para inicializar la capa fácilmente
    SymbexLayer(uint16_t inputs, uint16_t neurons, 
                const uint8_t* msb, const uint8_t* mid, const uint8_t* lsb, 
                const int16_t* th);

    // Ejecuta la matemática de esta capa específica
    void process_layer(const uint8_t* input_state, uint8_t* output_state);
};

// ---------------------------------------------------------
// CLASE 2: El Gestor de la Red (El Chasis)
// ---------------------------------------------------------
#define MAX_LAYERS 5 // Límite estático para no fragmentar la RAM del micro

class SymbexNetwork {
private:
    SymbexLayer* layers[MAX_LAYERS]; // Arreglo de punteros a nuestras capas
    uint8_t total_layers;

public:
    SymbexNetwork();

    // Permite agregar capas una tras otra (ej. entrada -> oculta -> salida)
    bool add_layer(SymbexLayer* layer);

    // La función maestra: toma el sensor, lo pasa por todas las capas
    // y devuelve el byte final para el diccionario LUT
    uint8_t predict(const uint8_t* sensor_input);
};

#endif // SYMBEX_NETWORK_H
