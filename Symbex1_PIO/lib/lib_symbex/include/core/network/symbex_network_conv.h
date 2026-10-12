/*
 * symbex_network_conv.h -- encadena BLOQUES residuales (conv1+conv2,
 * cada uno con su propia suma de identity), no capas planas.
 *
 * REDISEÑO respecto a la version anterior: esa version asumia una
 * cadena simple de capas independientes -- no existe tal cosa, cada
 * bloque tiene DOS mitades con DOS sumas residuales internas (ver
 * el forward real de BasicBlockBiReal, confirmado hoy).
 *
 * ACTUALIZADO: ahora SI soporta bloques con downsample (RSign +
 * pool binario 2x2 + conv 3x3 + BN afin, sin RPReLU) -- validado en
 * Python contra el modulo DownsampleBinario real (2.46% de
 * diferencias por encima de una tolerancia estricta, consistente con
 * redondeo normal de punto fijo, no un termino faltante).
 */

#ifndef SYMBEX_NETWORK_CONV_H
#define SYMBEX_NETWORK_CONV_H

#include <stdint.h>
#include <stdbool.h>
#include "layers/spatial/symbex_conv2d.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SYMBEX_NETWORK_CONV_MAX_BLOQUES 16

struct SymbexResidualBlock {
    struct SymbexConv2DLayer conv1;
    struct SymbexConv2DLayer conv2;
    const int32_t* beta_rsign1;   // Q_ACT, uno por canal de entrada del bloque
    const int32_t* beta_rsign2;   // Q_ACT, uno por canal de salida de conv1
    bool tiene_downsample;

    // Campos del downsample -- solo validos si tiene_downsample==true.
    // Cadena: binarizar(beta_rsign_downsample) -> pool binario 2x2 ->
    // conv 3x3 (downsample_pesos) -> BN afin (downsample_dsp), sin
    // RPReLU -- el resultado ES el identity para el bloque principal.
    const uint32_t* downsample_pesos;
    const int32_t* beta_rsign_downsample;
    SymbexDspParamsSimple downsample_dsp;
    uint16_t downsample_kernel;   // normalmente 3
    uint16_t downsample_padding;  // normalmente 1 (kernel/2)
};

struct SymbexNetworkConv {
    const struct SymbexResidualBlock* bloques[SYMBEX_NETWORK_CONV_MAX_BLOQUES];
    int bloque_count;
};

void symbex_network_conv_init(struct SymbexNetworkConv* net);

bool symbex_network_conv_add_bloque(struct SymbexNetworkConv* net, const struct SymbexResidualBlock* bloque);

/*
 * Corre la imagen (ya en formato bits, salida del stem binarizado)
 * a traves de todos los bloques. x_inicial_q_act: valores Q_ACT de
 * entrada al primer bloque (salida del stem, convertida a Q_ACT).
 * out_final_q_act: buffer de salida, valores Q_ACT del ULTIMO bloque
 * (listo para convertir a float y pasar al clasificador).
 *
 * Devuelve true si corrio bien. false solo si bloque_count==0.
 *
 * buffer_a, buffer_b, buffer_identity: buffers de trabajo Q_ACT,
 * cada uno lo bastante grande para el mayor mapa de activacion de la
 * red. line_buffer, patch_buffer: iguales que antes (bits).
 */
/*
 * Procesa UN bloque residual completo (con downsample si aplica),
 * escribiendo el resultado en destino_q_act.
 *
 * debug_identity1 / debug_conv1_out: OPCIONALES (pasar NULL si no
 * hacen falta) -- si no son NULL, se copia ahi el identity1 (el
 * "atajo" real usado para conv1: downsample(x) o x directo) y la
 * salida de conv1 (=identity2), respectivamente. Tamaño esperado:
 * conv1.out_ch * h1 * w1 (h1/w1 = resolucion de SALIDA de conv1).
 * Reusa el computo YA VALIDADO de la funcion, sin reproducirlo a
 * mano en otro lado (evita bugs de transcripcion).
 */
void symbex_network_conv_procesar_bloque(
    const struct SymbexResidualBlock* bloque,
    const int32_t* x_actual_q_act,
    int32_t* destino_q_act,
    int32_t* buffer_temporal,
    uint32_t* bits_buffer,
    int32_t* buffer_identity_downsample,
    uint32_t* bits_buffer_full_downsample,
    uint32_t* bits_buffer_pooled_downsample,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* debug_identity1, int32_t* debug_conv1_out
);

bool symbex_network_conv_run(
    const struct SymbexNetworkConv* net,
    const int32_t* x_inicial_q_act,
    int32_t* buffer_a, int32_t* buffer_b,
    uint32_t* bits_buffer,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* out_final_q_act
);

/*
 * Version extendida -- necesaria si la red tiene bloques con
 * downsample. Los 3 buffers extra son SOLO para el camino del
 * downsample (RSign+pool+conv3x3+BN):
 *   buffer_identity_downsample: Q_ACT, tamaño del mayor identity de
 *     downsample de la red (out_ch de conv1 x su h x su w)
 *   bits_buffer_full_downsample: bits, resolucion COMPLETA de
 *     entrada al downsample (antes de pool)
 *   bits_buffer_pooled_downsample: bits, resolucion REDUCIDA (mitad)
 *
 * symbex_network_conv_run (la version simple) llama a esta pasando
 * NULL en los 3 -- solo funciona si NINGUN bloque tiene downsample.
 */
bool symbex_network_conv_run_ext(
    const struct SymbexNetworkConv* net,
    const int32_t* x_inicial_q_act,
    int32_t* buffer_a, int32_t* buffer_b,
    uint32_t* bits_buffer,
    int32_t* buffer_identity_downsample,
    uint32_t* bits_buffer_full_downsample,
    uint32_t* bits_buffer_pooled_downsample,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* out_final_q_act
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_NETWORK_CONV_H
