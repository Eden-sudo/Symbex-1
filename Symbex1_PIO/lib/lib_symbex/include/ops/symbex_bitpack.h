/*
 * symbex_bitpack.h -- empaqueta puntuaciones enteras crudas (como
 * las que devuelve symbex_dense_forward) en bits, para poder usarlas
 * como entrada de la proxima capa.
 */

#ifndef SYMBEX_BITPACK_H
#define SYMBEX_BITPACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Empaqueta n puntuaciones crudas en bits -- bipolar STE:
 * puntuacion >= 0 -> bit 1, sino bit 0.
 *
 * scores: n valores int32_t (puntuaciones crudas de una capa).
 * out_bits: buffer de salida, (n+31)/32 palabras.
 */
void symbex_bitpack_from_scores(const int32_t* scores, uint32_t* out_bits, uint16_t n);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_BITPACK_H
