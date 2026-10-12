/*
 * symbex_gated.h -- capa DENSA con activacion condicional por
 * bloques (tipo "mixture of experts"). Una sub-red "directora"
 * puntua bloques de neuronas, se activan solo los K mejores, el
 * resto se saltea (computo condicional real, no cosmetico).
 *
 * Migrado de symbex_gated.h/.cpp -- misma logica, ahora usa
 * symbex_xnor_popcount (hal/) en vez del viejo SYMBEX_BIT_MAC.
 */

#ifndef SYMBEX_GATED_H
#define SYMBEX_GATED_H

#include "core/symbex_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fase 1: evaluacion del "director". Puntua cada bloque, marca como
 * activos los K mejores.
 *
 * active_blocks: array de salida, un bool por bloque.
 * k_active: cuantos bloques activar como maximo.
 */
void symbex_gated_evaluate(
    const uint32_t* input_planes,
    bool* active_blocks,
    const struct SymbexGatedLayerData* layer,
    uint16_t k_active
);

/*
 * Fase 2: inferencia condicional. Solo procesa bloques marcados
 * activos -- los inactivos se saltean (avanza el puntero de pesos,
 * no hace el computo). Empaqueta la salida bit a bit (bipolar STE:
 * puntaje > 0 -> bit 1).
 */
void symbex_gated_forward(
    const uint32_t* input_planes,
    uint32_t* output_buffer,
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
);

/*
 * Fase 2 (alternativa): inferencia condicional con clasificacion
 * final directa (argmax), para usar como ultima capa.
 */
int symbex_gated_argmax(
    const uint32_t* input_planes,
    const bool* active_blocks,
    const struct SymbexGatedLayerData* layer
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_GATED_H
