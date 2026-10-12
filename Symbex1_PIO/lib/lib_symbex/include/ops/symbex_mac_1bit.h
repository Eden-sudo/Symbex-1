/*
 * symbex_mac_1bit.h -- MAC binarizado con aritmetica de punto fijo
 * (DSP), validado end-to-end en Python contra PyTorch real
 * (84.80% vs 86.20% de precision, 500 imagenes de CIFAR-10).
 *
 * REEMPLAZA la version anterior (SYMBEX_MODO_SIMPLE/TRIPLE) -- esa
 * version NO incluia la suma residual (identity) que el forward real
 * del modelo si hace, lo que causaba ~27-38% de error por capa
 * cuando se probo rigurosamente (100+ imagenes, no 1).
 *
 * Formula, la misma para CUALQUIER conv (1 o 2) de cualquier bloque:
 *
 *   y = (raw_popcount * escala) >> (Q_MULT - Q_ACT)
 *   y += shift_bn
 *   y += identity                     <- la suma real, en runtime
 *   diff = y - gamma_rprelu
 *   si diff <= 0: salida = shift(diff, log2_beta, signo_beta) + zeta
 *   si no:        salida = diff + zeta
 *
 * "salida" queda en formato Q_ACT -- se usa DIRECTO como el
 * "identity" de la proxima capa. El bit para alimentar la SIGUIENTE
 * convolucion se obtiene aparte, comparando salida contra el beta
 * del RSign siguiente (ver symbex_dsp_binarizar).
 *
 * CAMBIO (version int32): escala pasa de int64 a int32, en
 * SymbexDspParams y en SymbexDspParamsSimple. Con Q_MULT=20 cabe
 * cualquier |escala| < 2048 (el exportador lo verifica). Los
 * productos siguen siendo de 64 bits, pero 32x32->64. Resultado
 * identico bit a bit a la version int64.
 */

#ifndef SYMBEX_MAC_1BIT_H
#define SYMBEX_MAC_1BIT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Parametros DSP de UNA capa (conv1 o conv2 de un bloque), un valor
 * por canal de salida en cada array -- salvo q_act/q_mult, que son
 * globales a toda la red (mismo formato en todas las capas).
 */
typedef struct {
    const int32_t* escala;          // Q_MULT -- alpha*scale_bn combinados (antes int64)
    const int32_t* shift_bn;        // Q_ACT
    const int32_t* gamma_rprelu;    // Q_ACT -- umbral de la rama de RPReLU (bias1)
    const int32_t* zeta;            // Q_ACT (bias2)
    bool usa_shift;                 // true: RPReLU_Shift (potencia de 2, con shift).
                                     // false: RPReLU normal (multiplicacion, con slope_q)
    const int8_t* log2_beta;        // solo si usa_shift==true -- exponente,
                                     // NEGATIVO si beta<1 (el caso comun)
    const int8_t* signo_beta;       // solo si usa_shift==true -- +1 o -1
    const int32_t* slope_q;         // solo si usa_shift==false -- pendiente en Q_ACT
    uint8_t q_act;                  // bits fraccionarios de activaciones
    uint8_t q_mult;                 // bits fraccionarios del multiplicador
} SymbexDspParams;

/*
 * Calcula la salida de UNA capa, para el canal 'canal', dado el
 * popcount crudo de esa posicion y el identity (Q_ACT) que corresponde
 * sumarle. Devuelve la salida en formato Q_ACT -- USARLA como
 * identity de la proxima capa, o pasarla a symbex_dsp_binarizar
 * para obtener el bit que alimenta la siguiente convolucion.
 */
int32_t symbex_dsp_aplicar_capa(int32_t raw_popcount, int32_t identity_q_act, uint16_t canal, const SymbexDspParams* params);

/*
 * Convierte un valor Q_ACT (salida de symbex_dsp_aplicar_capa, o la
 * entrada de un bloque) en un bit -- comparando contra el beta
 * (Q_ACT) del RSign correspondiente. sign(valor - beta).
 */
bool symbex_dsp_binarizar(int32_t valor_q_act, int32_t beta_rsign_q_act);

/*
 * Version SIMPLIFICADA, para el downsample -- solo BN afin, SIN
 * RPReLU (el downsample no tiene esa rama). y = (raw*escala)>>shift
 * + shift_bn -- eso es directo el "identity" a sumar en el bloque
 * principal, no hace falta ningun paso mas.
 */
typedef struct {
    const int32_t* escala;          // antes int64
    const int32_t* shift_bn;
    uint8_t q_act;
    uint8_t q_mult;
} SymbexDspParamsSimple;

int32_t symbex_dsp_aplicar_downsample(int32_t raw_popcount, uint16_t canal, const SymbexDspParamsSimple* params);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_MAC_1BIT_H
