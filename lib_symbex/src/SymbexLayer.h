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

struct SymbexSubLayer {
    const uint8_t* bit_planes[MAX_K_BITS];
    const uint8_t* outliers;
    const int8_t* outlier_mag;
};

class SymbexLayer {
public:
    uint16_t in_features;
    uint16_t out_features;
    uint8_t M;
    uint8_t k_bits;
    const SymbexSubLayer* sub_layers;

    SymbexLayer(uint16_t in_f, uint16_t out_f, uint8_t m, uint8_t k, const SymbexSubLayer* subs);
    
    // Inferencia intermedia (binarizada)
    void forward(const uint8_t* input_buffer, uint8_t* output_buffer);
    
    // Inferencia final (devuelve la clase ganadora)
    int argmax(const uint8_t* input_buffer);
};

#endif // SYMBEX_LAYER_H
