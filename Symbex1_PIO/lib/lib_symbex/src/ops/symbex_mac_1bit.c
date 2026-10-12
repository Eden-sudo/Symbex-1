/*
 * symbex_mac_1bit.c -- implementacion. Misma aritmetica, mismo orden
 * de operaciones, que el simulador Python V4 (validado en el test
 * set COMPLETO de CIFAR-10: 74.63% simulador vs 74.68% PyTorch real,
 * 10,000 imagenes -- 0.05 puntos de diferencia).
 *
 * Incluye redondeo "al mas cercano" en cada shift (en vez de
 * truncar), y soporta tanto RPReLU_Shift (potencia de 2, con shift)
 * como RPReLU normal (multiplicacion general, con slope_q) -- el
 * stem del modelo completamente binario usa RPReLU normal.
 *
 * VERSION int32 + RPReLU SIN SALTO -- mismos numeros, bit a bit, que
 * la version int64 (validado contra ella). Dos cambios de rendimiento:
 *   1. escala int32: el producto raw*escala es 32x32->64, no 64x64.
 *      En x86-64 da casi igual; en el S3 (32 bits) es la diferencia
 *      entre un par de instrucciones y una rutina de 64x64.
 *   2. Sin el "if (diff <= 0)": se calculan las dos ramas y se elige
 *      con una seleccion (cmov en x86). El signo de diff es casi al
 *      azar, y un salto mal predicho cuesta ~15 ciclos en el i5.
 * Los valores Q_ACT (y, diff, salida) caben en int32 mientras el
 * valor real sea menor a 32768, igual que antes al devolver int32.
 */

#include "ops/symbex_mac_1bit.h"

int32_t symbex_dsp_aplicar_capa(int32_t raw_popcount, int32_t identity_q_act, uint16_t canal, const SymbexDspParams* params) {
    uint8_t shift_reescalado = params->q_mult - params->q_act;
    int64_t redondeo_mult = (int64_t)1 << (shift_reescalado - 1);

    int32_t y = (int32_t)((((int64_t)raw_popcount * params->escala[canal]) + redondeo_mult) >> shift_reescalado);
    y += params->shift_bn[canal];
    y += identity_q_act;

    int32_t diff = y - params->gamma_rprelu[canal];
    int64_t rama_negativa;          // 64 bits como antes: con log2_beta >= 0 el shift puede pasar de int32

    if (params->usa_shift) {
        // Este salto SI se predice bien: usa_shift es fijo por capa.
        // La magnitud se calcula siempre sobre un valor >= 0 (0 si
        // diff > 0) -- desplazar a la izquierda un negativo es
        // comportamiento indefinido en C.
        int64_t negado = (diff > 0) ? 0 : -(int64_t)diff;
        int8_t log2b = params->log2_beta[canal];
        int64_t magnitud = (log2b >= 0) ? (negado << log2b) : (negado >> (-log2b));
        rama_negativa = (params->signo_beta[canal] >= 0) ? -magnitud : magnitud;
    } else {
        // RPReLU normal: diff * slope_q, redondeado al mas cercano.
        // SIEMPRE se suma la mitad (no depende del signo) -- el shift
        // aritmetico ya hace floor, sumar la mitad antes lo convierte
        // en "redondeo al mas cercano" para CUALQUIER signo.
        int64_t redondeo_act = (int64_t)1 << (params->q_act - 1);
        rama_negativa = (((int64_t)diff * params->slope_q[canal]) + redondeo_act) >> params->q_act;
    }

    int64_t salida = ((diff > 0) ? (int64_t)diff : rama_negativa) + params->zeta[canal];
    return (int32_t)salida;
}

bool symbex_dsp_binarizar(int32_t valor_q_act, int32_t beta_rsign_q_act) {
    return valor_q_act >= beta_rsign_q_act;
}

int32_t symbex_dsp_aplicar_downsample(int32_t raw_popcount, uint16_t canal, const SymbexDspParamsSimple* params) {
    uint8_t shift_reescalado = params->q_mult - params->q_act;
    int64_t redondeo = (int64_t)1 << (shift_reescalado - 1);
    int32_t y = (int32_t)((((int64_t)raw_popcount * params->escala[canal]) + redondeo) >> shift_reescalado);
    return y + params->shift_bn[canal];
}
