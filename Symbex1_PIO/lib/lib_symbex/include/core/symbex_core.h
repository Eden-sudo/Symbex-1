/*
 * symbex_core.h -- utilidades generales, portables a cualquier placa.
 *
 * NOTA DE MIGRACION: SYMBEX_BIT_MAC y SYMBEX_XNOR del archivo
 * original fueron RETIRADOS -- estan completamente superados por
 * symbex_xnor_popcount (hal/symbex_hal.h), que SI tiene aceleracion
 * real en ESP32-S3 (el viejo symbex_core.h tenia un placeholder sin
 * llenar para eso, caia a __builtin_popcount sin PIE).
 *
 * PENDIENTE REAL, no resuelto todavia: symbex_xnor_popcount no lee
 * memoria via pgm_read_dword en AVR -- si los pesos viven en flash
 * de un AVR real, hace falta agregar ese soporte a hal/generico
 * antes de que esto funcione correctamente ahi. Se mantiene
 * SYMBEX_READ_W32 aca mientras tanto, para las partes del codigo que
 * todavia leen palabra por palabra.
 */

#ifndef SYMBEX_CORE_H
#define SYMBEX_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AVR__)
    #include <avr/pgmspace.h>
    #define SYMBEX_READ_W32(addr) pgm_read_dword(addr)
#else
    #define SYMBEX_READ_W32(addr) (*(const uint32_t*)(addr))
#endif

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_CORE_H
