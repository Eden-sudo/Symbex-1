#ifndef SYMBEX_BLOCK_GATED_LAYER_H
#define SYMBEX_BLOCK_GATED_LAYER_H

#include <stdint.h>
#include <string.h>

class SymbexBlockGatedLayer {
public:
    uint16_t in_features;
    uint16_t out_features;
    uint16_t block_size;
    uint16_t num_blocks;
    uint16_t k_active;

    const uint8_t* gate_weights;
    const uint8_t* core_weights;

    SymbexBlockGatedLayer(uint16_t in_f, uint16_t out_f, uint16_t b_size, uint16_t k_act, 
                          const uint8_t* g_w, const uint8_t* c_w);

    void forward(const uint8_t* __restrict input_buffer, uint8_t* __restrict output_buffer);
    int argmax(const uint8_t* __restrict input_buffer);

    // [NUEVO] Variantes MAC (Desempaquetado a int8 + Multiplicación Nativa)
    void forward_mac(const uint8_t* __restrict input_buffer, uint8_t* __restrict output_buffer);
    int argmax_mac(const uint8_t* __restrict input_buffer);
};

#endif
