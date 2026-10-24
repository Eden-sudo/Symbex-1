/*
 * symbex_hal.h -- INTERFAZ de hardware que cualquier placa debe implementar.
 *
 * Implementaciones:
 *   hal/esp32s3/symbex_hal_impl.c  -- solo ESP32-S3
 *   hal/generico/symbex_hal_impl.c -- todo lo demas (elige en compilacion:
 *                                     popcount nativo, lut16 en ESP32, SWAR, libgcc en AVR)
 */

#ifndef SYMBEX_HAL_H
#define SYMBEX_HAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Layout for the ESP32-S3 PIE kernel (-DSYMBEX_USE_PIE): ee.vld.128 ignores the low 4 address bits,
 * so the kernel only runs when BOTH operands are 16-byte aligned (else that call takes the default
 * path, same result). Declare weight arrays and activation buffers with SYMBEX_ALIGN16; for every
 * neuron row of a dense layer to stay aligned, the row stride ceil(in_features / 32) words must be a
 * multiple of 4 (in_features a multiple of 128, or rows padded to it).
 */
#if defined(__GNUC__)
#define SYMBEX_ALIGN16 __attribute__((aligned(16)))
#else
#define SYMBEX_ALIGN16
#endif

/* XNOR + popcount sobre bloques de bits: devuelve 2*coincidencias - n_bits_reales. */
int32_t symbex_xnor_popcount(const uint32_t* a, const uint32_t* b, int n_palabras, int n_bits_reales);

/* Popcount de UNA palabra, con el mismo metodo que el HAL usa por dentro.
 * Usar esta en vez de __builtin_popcount (que en ESP32 es una rutina lenta de libgcc). */
int32_t symbex_hal_popcount32(uint32_t x);

/* Texto con la ruta de conteo activa (y si lut16 consiguio RAM). Para logs y para el paper. */
const char* symbex_hal_nombre(void);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_HAL_H
