#ifndef SYMBEX_LAYER_H
#define SYMBEX_LAYER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __AVR__
#include <avr/pgmspace.h>
#define SYMBEX_READ_BYTE(addr) pgm_read_byte(addr)
#else
#define SYMBEX_READ_BYTE(addr) (*(addr))
#ifndef PROGMEM
#define PROGMEM
#endif
#endif

#define MAX_K_BITS 4  
#define MAX_BLOCKS 32 // Límite de bloques para no saturar la RAM del Arduino

struct SymbexSubLayer {
    const uint8_t* bit_planes[MAX_K_BITS];
    const uint8_t* outliers;
    const int8_t* outlier_mag;
};

class SymbexLayer {
public:
    // [!] Constructor actualizado con parámetros por defecto para compatibilidad
    SymbexLayer(uint16_t in_f, uint16_t out_f, uint8_t m, uint8_t k, const SymbexSubLayer* subs, uint8_t blk_size = 0, uint8_t k_act = 0);
    
    /**
     * @brief Procesa la capa oculta y empaqueta el resultado en bits.
     * @param input_buffer Arreglo de bits de entrada (datos del sensor o capa anterior).
     * @param output_buffer Arreglo donde se guardarán los bits resultantes (>0 = 1, <=0 = 0).
     */
    void forward(const uint8_t* input_buffer, uint8_t* output_buffer);
    
    /**
     * @brief Evalúa la capa final y decide qué neurona ganó.
     * @param input_buffer Arreglo de bits de entrada.
     * @return El índice (ID) de la neurona con la puntuación más alta.
     */
    int argmax(const uint8_t* input_buffer);

private:
    /** @brief Número de características de entrada (bits que recibe la capa). */
    uint16_t in_features;
    /** @brief Número de características de salida (neuronas de esta capa). */
    uint16_t out_features;
    /** @brief Factor de expansión topológica. Cuántas copias tiene cada neurona. */
    uint8_t M;
    /** @brief Niveles de cuantización del Bit-Slicing (ej. K=3 para MSB, MID, LSB). */
    uint8_t k_bits;
    /** @brief Puntero al arreglo de sub-capas (matrices lógicas) almacenadas en memoria Flash. */
    const SymbexSubLayer* sub_layers;

    // --- VARIABLES NUEVAS PARA BLOCK-GATING ---
    uint8_t block_size;
    uint8_t k_active;

    /**
     * @brief Evalúa el Director y define el Top-K de bloques vivos.
     */
    void evaluate_gates(const uint8_t* input_buffer, bool* active_blocks);

    /**
     * @brief Función interna. Calcula el puntaje bruto de una sola neurona.
     * @param n Índice de la neurona a evaluar.
     * @param input_buffer Arreglo de bits de entrada.
     * @return Sumatoria total de votos (aciertos - fallos).
     */
    inline int32_t compute_neuron_votes(uint16_t n, const uint8_t* input_buffer);
};

#endif // SYMBEX_LAYER_H
