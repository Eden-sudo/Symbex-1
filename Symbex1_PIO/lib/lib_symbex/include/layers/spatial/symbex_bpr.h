/*
 * symbex_bpr.h -- codificacion "Binary Pixel Representation" (BPR):
 * descompone una imagen RGB en 24 planos de bits (8 por canal, MSB
 * primero), el primer paso de toda la red -- equivalente en C de
 * RGB_a_24Bits (Python), validado extensamente hoy.
 *
 * Responde a la objecion original de XNOR-Net sobre "muy pocos
 * canales para que valga la pena binarizar la primera capa" -- 3
 * canales RGB se convierten en 24 planos, suficiente ancho para que
 * XNOR+popcount tenga sentido real.
 */

#ifndef SYMBEX_BPR_H
#define SYMBEX_BPR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * imagen_rgb: entrelazado, [h][w][3] -- 3 bytes por pixel (R,G,B),
 *   el formato tipico de una camara, NO el [C,H,W] de PyTorch.
 * out_bits: 24 planos empaquetados (orden: R bit7..bit0, G bit7..bit0,
 *   B bit7..bit0), listos para alimentar el stem de la red.
 */
void symbex_rgb_a_24bits(const uint8_t* imagen_rgb, uint32_t* out_bits, uint16_t h, uint16_t w);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_BPR_H
