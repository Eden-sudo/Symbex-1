/*
 * symbex_bitops.h -- operaciones de bits de bajo nivel, genericas,
 * usadas por CUALQUIER capa espacial (conv1d, conv2d, pool...).
 *
 * Migrado de symbex_conv.cpp (las funciones "El Mesero" y "El
 * Ayudante") -- rescatado porque maneja el caso GENERAL de anchos
 * de imagen arbitrarios (a diferencia de symbex_extraer_ventana en
 * conv2d, que asume que una fila entra en una palabra de 32 bits).
 *
 * NOTA DE DISENO: a diferencia del original, load_line_buffer ya NO
 * recibe una struct de capa completa -- recibe parametros sueltos,
 * para que ops/ no dependa de ningun tipo de capa especifico.
 */

#ifndef SYMBEX_BITOPS_H
#define SYMBEX_BITOPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extrae un rango de N bits (hasta 32) empezando en bit_start,
 * incluso si ese rango CRUZA el limite de una palabra de 32 bits.
 *
 * plane: bloque de bits de origen.
 * bit_start: posicion inicial, en bits, dentro de plane.
 * n_bits: cuantos bits extraer (1 a 32).
 * Devuelve: los n_bits extraidos, alineados a la derecha.
 */
uint32_t symbex_extract_bits(const uint32_t* plane, uint32_t bit_start, uint8_t n_bits);

/*
 * Inserta un valor de N bits en plane, empezando en bit_start_local
 * -- LIMPIA primero esos bits (AND con mascara invertida), despues
 * los pone (OR) -- funciona sin importar el valor inicial de plane
 * (antes asumia que arrancaba en cero; con patch_buffer arrancando
 * en 0xFFFFFFFF esto era un bug real, corregido).
 * Tambien maneja el caso de cruzar el limite de una palabra.
 *
 * plane: bloque de bits de destino (se modifica).
 * bit_start_local: posicion inicial, en bits, dentro de plane.
 * n_bits: cuantos bits del value son validos (1 a 32).
 * value: los bits a insertar, alineados a la derecha.
 */
void symbex_insert_bits(uint32_t* plane, uint32_t bit_start_local, uint8_t n_bits, uint32_t value);

/*
 * Cuenta cuantos bits estan en 1 tanto en a como en b (popcount de
 * a AND b), sobre n_palabras completas -- usada para corregir el
 * popcount de padding: cuenta cuantos pesos estan en 1 justo en las
 * posiciones marcadas como padding por una mascara.
 */
int32_t symbex_popcount_and(const uint32_t* a, const uint32_t* b, uint16_t n_palabras);

/*
 * Carga un buffer de trabajo (SRAM) con las 'kernel' filas
 * necesarias para procesar la fila de salida 'oh', copiando desde
 * una fuente mas grande (ej. la imagen completa binarizada). Rellena
 * con ceros las filas que caen fuera de los limites (padding de
 * convolucion).
 *
 * source_input: bloque de bits de origen (la imagen completa).
 * sram_buffer: buffer de destino, debe tener espacio para
 *   in_ch * kernel * ((in_w+31)/32) palabras.
 * in_ch, in_h, in_w: dimensiones de la imagen de entrada.
 * kernel, stride, padding: parametros de la convolucion.
 * oh: fila de SALIDA que se va a procesar (no de entrada).
 */
void symbex_load_line_buffer(
    const uint32_t* source_input,
    uint32_t* sram_buffer,
    uint16_t in_ch, uint16_t in_h, uint16_t in_w,
    uint16_t kernel, uint16_t stride, uint16_t padding,
    int16_t oh
);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_BITOPS_H
