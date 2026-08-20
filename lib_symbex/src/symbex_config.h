#ifndef SYMBEX_CONFIG_H
#define SYMBEX_CONFIG_H

#include <stdint.h>

// =====================================================================
// 1. ESP32 (Sistemas de 32 bits con RAM externa)
// =====================================================================
#if defined(ESP32)
    // Fuerza los tensores masivos a la memoria Flash
    #define SYMBEX_MEM_MODEL __attribute__((section(".ext_ram.rodata")))
    
    // El ESP32 puede leer la Flash como si fuera RAM normal (XIP)
    #define SYMBEX_READ_BYTE(addr) (*(addr))
    #define SYMBEX_READ_INT16(addr) (*(addr))

// =====================================================================
// 2. 8-BIT AVR (La legendaria "papa": Arduino Uno, Mega, Nano)
// =====================================================================
#elif defined(__AVR__)
    #include <avr/pgmspace.h>
    
    // Etiqueta obligatoria para guardar en los 32KB de Flash del Arduino
    #define SYMBEX_MEM_MODEL PROGMEM
    
    // En AVR, DEBEMOS extraer los bytes con instrucciones de ensamblador especificas
    #define SYMBEX_READ_BYTE(addr) pgm_read_byte(addr)
    #define SYMBEX_READ_INT16(addr) pgm_read_word(addr)

// =====================================================================
// 3. ARM CORTEX (STM32, Arduino Nano 33 BLE, Teensy)
// =====================================================================
#elif defined(__arm__) || defined(__thumb__)
    #define SYMBEX_MEM_MODEL const
    #define SYMBEX_READ_BYTE(addr) (*(addr))
    #define SYMBEX_READ_INT16(addr) (*(addr))

// =====================================================================
// 4. PC CONVENCIONAL (Linux x86_64, Windows, Mac) / DEFAULT
// =====================================================================
#else
    // En PC no hay restricciones, lo dejamos vacio para que cargue en la RAM L1/L2
    #define SYMBEX_MEM_MODEL
    #define SYMBEX_READ_BYTE(addr) (*(addr))
    #define SYMBEX_READ_INT16(addr) (*(addr))

#endif

// LUT en PROGMEM (usando tu macro dinámica) para un Popcount de tiempo constante O(1)
const uint8_t POPCOUNT_LUT[16] SYMBEX_MEM_MODEL = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

inline uint8_t symbex_popcount(uint8_t x) {
    // Lee dos nibbles (4 bits) directamente de la memoria Flash sin bucles
    return SYMBEX_READ_BYTE(&POPCOUNT_LUT[x & 0x0F]) + SYMBEX_READ_BYTE(&POPCOUNT_LUT[x >> 4]);
}

#endif // SYMBEX_CONFIG_H
