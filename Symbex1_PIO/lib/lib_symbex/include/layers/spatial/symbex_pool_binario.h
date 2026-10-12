/*
 * symbex_pool_binario.h -- reduce a la mitad la resolucion espacial
 * (pooling 2x2), operando 100% sobre bits. "Voto de mayoria" de 4
 * pixeles, para imitar AvgPool2d + BinarizadorBiReal del downsample
 * (DownsampleBinario en Python).
 */

#ifndef SYMBEX_POOL_BINARIO_H
#define SYMBEX_POOL_BINARIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void symbex_pool_binario_2x2(const uint32_t* bits_in, uint32_t* bits_out, uint16_t ch, uint16_t h, uint16_t w);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_POOL_BINARIO_H
