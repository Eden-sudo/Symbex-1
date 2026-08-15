#ifndef SYMBEX_LAYER_H
#define SYMBEX_LAYER_H

#include <stdint.h>

class SymbexLayer {
private:
    int in_features;
    int out_features;
    
    // Via Principal (El nucleo normal K=3)
    const uint8_t* weights_msb;
    const uint8_t* weights_mid;
    const uint8_t* weights_lsb;
    
    // Via Secundaria (Capa de Abstraccion de Outliers)
    const uint8_t* weights_outlier; 
    const int16_t* outlier_magnitudes; // El valor de impacto del pico
    
    const int16_t* thresholds;

public:
    // Constructor actualizado
    SymbexLayer(int in, int out, 
                const uint8_t* msb, const uint8_t* mid, const uint8_t* lsb, 
                const uint8_t* outl_mask, const int16_t* outl_mags, 
                const int16_t* thresh);
    
    void process_layer(const uint8_t* input, uint8_t* output);
};

#endif // SYMBEX_LAYER_H
