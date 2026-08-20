#ifndef SYMBEX_LAYER_H
#define SYMBEX_LAYER_H

#include <stdint.h>
#include "symbex_config.h"

struct SymbexSubLayer {
    const uint8_t* weight_bit0;
    const uint8_t* weight_bit1;
    const uint8_t* weight_bit2;
    const uint8_t* outliers;
    const int8_t* outlier_magnitudes; // Unificado estrictamente a int8_t
};

class SymbexLayer {
public:
    uint16_t in_features;
    uint16_t out_features;
    uint8_t M_factor;
    const SymbexSubLayer* sub_layers;

    SymbexLayer(uint16_t in, uint16_t out, uint8_t M, const SymbexSubLayer* subs);
    
    void process_layer(const uint8_t* input, uint8_t* output);
    int process_layer_argmax(const uint8_t* input);
};

#endif
