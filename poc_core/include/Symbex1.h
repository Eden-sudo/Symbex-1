/*
 * SYMBEX-1 (Symbolic Bit Expansion 1-bit Engine)
 * Motor de Inferencia Binarizada con Cuantización Residual (K=3)
 */

#ifndef SYMBEX1_H
#define SYMBEX1_H

#include <stdint.h>

// 1. Estructura que define una capa oculta (Las 3 hojas semitransparentes)
struct SymbexLayer {
    uint16_t num_inputs;    // Cantidad de bits que entran a la capa
    uint16_t num_neurons;   // Cantidad de neuronas en esta capa
    
    // Punteros a las matrices lógicas (Pesos binarizados)
    // En un microcontrolador real, estos apuntarán a la memoria Flash
    const uint8_t* weights_msb; 
    const uint8_t* weights_mid;
    const uint8_t* weights_lsb;
    
    // Umbrales de activación para cada neurona (El límite para disparar un 1 o 0)
    const int16_t* thresholds; 
};

// 2. La Clase Principal (El Motor)
class SymbexEngine {
private:
    // Punteros a las capas de nuestra red
    const SymbexLayer* hidden_layer;
    
    // Función matemática interna (XNOR + POPCOUNT)
    // Devuelve el acumulado de coincidencias entre la entrada y los pesos
    int8_t compute_mac_1bit(uint8_t input_bits, uint8_t weight_bits);

public:
    // Constructor de la clase
    SymbexEngine();

    // Carga la estructura de la red desde el archivo generado por Python
    void load_network(const SymbexLayer* layer_config);

    // El corazón del sistema: Ejecuta la inferencia
    // Toma un arreglo de bytes (sensores) y devuelve el byte final (decisión)
    uint8_t predict(const uint8_t* input_state);
};

#endif // SYMBEX1_H
