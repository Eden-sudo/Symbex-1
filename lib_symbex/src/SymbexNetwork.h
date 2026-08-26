#ifndef SYMBEX_NETWORK_H
#define SYMBEX_NETWORK_H

#include <stdint.h>
#include "SymbexLayer.h"

// =================================================================
// LÍMITES DE MEMORIA ESTÁTICA (Prevención de Out-Of-Memory en SRAM)
// =================================================================

/** @brief Límite máximo de capas que puede ensamblar el orquestador. */
#define MAX_LAYERS 8

/** 
 * @brief Tamaño de los buffers de paso en bytes. 
 * Un tamaño de 128 soporta el empaquetado de hasta 1024 neuronas por capa (128 * 8 bits). 
 */
#define BUFFER_SIZE 128

/**
 * @class SymbexNetwork
 * @brief Orquestador del pipeline de inferencia. 
 * Ensambla la topología de la red y gestiona el flujo de memoria estática 
 * para procesar las activaciones entre capas.
 */
class SymbexNetwork {
private:
    /** @brief Arreglo estático que almacena los punteros a cada capa de la red. */
    SymbexLayer* layers[MAX_LAYERS];
    
    /** @brief Número actual de capas registradas en el pipeline. */
    uint8_t layer_count;
    
    // =========================
    // GESTIÓN DE MEMORIA (SRAM)
    // =========================
    
    /** @brief Buffer primario para el almacenamiento temporal de activaciones (Ping-Pong). */
    uint8_t buffer_A[BUFFER_SIZE];
    
    /** @brief Buffer secundario para la sobreescritura cruzada entre capas (Ping-Pong). */
    uint8_t buffer_B[BUFFER_SIZE];
    
    /** @brief Buffer de reserva para inyectar estados previos en el bucle autoregresivo (Chaining). */
    uint8_t state_buffer[BUFFER_SIZE];  

public:
    /**
     * @brief Constructor de la red. Inicializa el contador de capas en cero de forma segura.
     */
    SymbexNetwork();
    
    /**
     * @brief Añade una nueva capa al pipeline de forma secuencial.
     * @param layer Puntero a una instancia de SymbexLayer pre-configurada.
     * @return true si la capa se agregó exitosamente. false si se superó MAX_LAYERS.
     */
    bool add_layer(SymbexLayer* layer);
    
    /**
     * @brief Ejecuta el ciclo de inferencia completo (Forward Pass).
     * Pasa los datos crudos por todas las capas ocultas y evalúa la capa de salida.
     * @param input Arreglo de bytes con los datos iniciales (ej. lectura de sensores).
     * @return El índice aritmético (ID) de la neurona de salida ganadora (El Símbolo a decodificar).
     */
    int classify(const uint8_t* input);
};

#endif // SYMBEX_NETWORK_H
