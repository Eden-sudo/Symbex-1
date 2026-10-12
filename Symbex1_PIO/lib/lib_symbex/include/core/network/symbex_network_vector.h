/*
 * symbex_network_vector.h -- encadena capas layers/vector/ (densa,
 * gated) en secuencia.
 *
 * Cada capa se etiqueta con su tipo (SYMBEX_VECTOR_DENSE o
 * SYMBEX_VECTOR_GATED) -- las firmas de dense y gated son distintas
 * entre si (gated necesita evaluar bloques activos primero), asi que
 * esto es una union etiquetada acotada a estos dos tipos, no una
 * interfaz generica para cualquier capa (ver la nota de diseño en
 * symbex_network_conv.h sobre por que NO se intento unificar todo).
 */

#ifndef SYMBEX_NETWORK_VECTOR_H
#define SYMBEX_NETWORK_VECTOR_H

#include <stdint.h>
#include <stdbool.h>
#include "core/symbex_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SYMBEX_NETWORK_VECTOR_MAX_LAYERS 16
#define SYMBEX_NETWORK_VECTOR_MAX_BLOCKS 64

enum SymbexVectorLayerType {
    SYMBEX_VECTOR_DENSE,
    SYMBEX_VECTOR_GATED
};

struct SymbexNetworkVectorLayer {
    enum SymbexVectorLayerType type;
    const struct SymbexLayerData* dense;        // usado si type == SYMBEX_VECTOR_DENSE
    const struct SymbexGatedLayerData* gated;    // usado si type == SYMBEX_VECTOR_GATED
    uint16_t k_active;                            // solo para GATED: bloques a activar
};

struct SymbexNetworkVector {
    struct SymbexNetworkVectorLayer layers[SYMBEX_NETWORK_VECTOR_MAX_LAYERS];
    int layer_count;
};

void symbex_network_vector_init(struct SymbexNetworkVector* net);

bool symbex_network_vector_add_dense(struct SymbexNetworkVector* net, const struct SymbexLayerData* layer);

bool symbex_network_vector_add_gated(struct SymbexNetworkVector* net, const struct SymbexGatedLayerData* layer, uint16_t k_active);

/*
 * Corre input_planes (bits ya empaquetados -- si el origen es una
 * imagen o vector flotante, binarizarlo antes de llamar esto) a
 * traves de TODAS las capas, en orden. La ULTIMA capa usa argmax
 * directo (no empaqueta, devuelve la clase ganadora).
 *
 * buffer_a, buffer_b: ping-pong para las capas intermedias -- deben
 *   ser lo bastante grandes para la mayor out_features/32 de la red.
 * scores_buffer: buffer de trabajo para las puntuaciones crudas de
 *   una capa dense antes de empaquetar -- lo bastante grande para
 *   la mayor out_features de la red.
 * active_blocks_buffer: buffer de trabajo para gated -- lo bastante
 *   grande para SYMBEX_NETWORK_VECTOR_MAX_BLOCKS.
 *
 * Devuelve: el indice de la clase ganadora (salida de la ultima capa).
 */
int symbex_network_vector_run(
    const struct SymbexNetworkVector* net,
    const uint32_t* input_planes,
    uint32_t* buffer_a, uint32_t* buffer_b,
    int32_t* scores_buffer, bool* active_blocks_buffer
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_NETWORK_VECTOR_H
