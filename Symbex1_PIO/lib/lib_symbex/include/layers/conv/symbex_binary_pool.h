/*
 * symbex_binary_pool.h -- halves spatial resolution
 * (2x2 pooling), operating 100% on bits. 4-pixel "majority vote",
 * to mimic AvgPool2d + BiRealBinarizer of downsample
 * (BinaryDownsample in Python).
 */

#ifndef SYMBEX_BINARY_POOL_H
#define SYMBEX_BINARY_POOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void symbex_binary_pool_2x2(const uint32_t* bits_in, uint32_t* bits_out, uint16_t ch, uint16_t h, uint16_t w);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_BINARY_POOL_H
