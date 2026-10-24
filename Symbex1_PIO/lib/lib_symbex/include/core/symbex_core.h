/*
 * symbex_core.h -- general utilities, portable to any board.
 *
 * MIGRATION NOTE: SYMBEX_BIT_MAC and SYMBEX_XNOR from the original
 * file have been REMOVED -- they are completely superseded by
 * symbex_xnor_popcount (hal/symbex_hal.h), which DOES have real
 * acceleration on the ESP32-S3 (the old symbex_core.h had an unfilled
 * placeholder for that, falling back to __builtin_popcount without PIE).
 *
 * REAL PENDING TASK, not resolved yet: symbex_xnor_popcount does not read
 * memory via pgm_read_dword on AVR -- if weights live in the flash
 * of a real AVR, that support needs to be added to hal/generic
 * before this works correctly there. SYMBEX_READ_W32 is kept here in the
 * meantime, for parts of the code that still read word by word.
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
