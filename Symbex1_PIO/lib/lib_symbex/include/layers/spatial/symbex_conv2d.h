/*
 * symbex_conv2d.h -- capa de convolucion 2D binarizada, con la
 * formula DSP corregida (suma residual real) Y correccion exacta
 * de padding de bordes (bit=0 uniforme fila+columna, corregido por
 * canal via mascara_padding -- el "padding=1 por defecto" anterior
 * no coincidia con el zero-padding real de PyTorch).
 *
 * CAMBIO IMPORTANTE respecto a la version anterior: ya NO escribe
 * bits directo -- escribe VALORES Q_ACT (punto fijo). Convertir a
 * bits es un paso aparte (symbex_dsp_binarizar_mapa).
 */

#ifndef SYMBEX_CONV2D_H
#define SYMBEX_CONV2D_H

#include <stdint.h>
#include "ops/symbex_mac_1bit.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SymbexConv2DLayer {
    SymbexDspParams dsp;
    const uint32_t* weights;
    uint16_t in_ch, in_h, in_w;
    uint16_t out_ch, kernel, stride, padding;
};

/*
 * Procesa UNA posicion (oh, ow), TODOS los canales de salida.
 * mascara_padding: mismo tamaño que patch_buffer, 1 en cada bit que
 * sea padding (fila o columna fuera de imagen), 0 en los reales --
 * todo-cero para posiciones interiores. Usada para corregir el
 * popcount exacto por canal.
 */
void symbex_conv2d_procesar_posicion(
    const uint32_t* patch_buffer,
    const uint32_t* mascara_padding,
    const int32_t* identity_q_act,
    int32_t* out_valores_q_act,
    const struct SymbexConv2DLayer* layer
);

void symbex_dsp_binarizar_mapa(
    const int32_t* valores_q_act, uint32_t* out_bits,
    const int32_t* beta_rsign_q_act,
    uint16_t ch, uint16_t h, uint16_t w
);

void symbex_binarizar_imagen(const float* input_map, uint32_t* out_bits,
                              uint16_t ch, uint16_t h, uint16_t w);

void symbex_conv2d_preparar_fila(const uint32_t* img_bits, uint32_t* line_buffer,
                                  uint16_t in_ch, uint16_t in_h, uint16_t in_w,
                                  uint16_t kernel, uint16_t stride, uint16_t padding, uint16_t oh);

/*
 * mascara_padding: buffer de SALIDA, mismo tamaño que patch_buffer --
 * esta funcion lo llena marcando que posiciones son padding.
 * in_h y oh son NUEVOS -- necesarios para detectar padding de FILA
 * aca tambien (antes solo se detectaba padding de columna).
 */
void symbex_conv2d_extraer_ventana(const uint32_t* line_buffer, uint32_t* patch_buffer, uint32_t* mascara_padding,
                                    uint16_t in_ch, uint16_t in_h, uint16_t in_w, uint16_t kernel,
                                    uint16_t stride, uint16_t padding, uint16_t oh, uint16_t ow);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_CONV2D_H
