#ifndef SYMBEX_CONFIG_H
#define SYMBEX_CONFIG_H

#include <stdint.h>

// --------------------------------------------------------
// 1. GESTIÓN DE MEMORIA (STACK) POR ARQUITECTURA
// --------------------------------------------------------
#ifdef __AVR__
    // En un Arduino Uno (2KB RAM), limitamos los bloques para no asfixiar el Stack
    #define SYMBEX_MAX_BLOCKS 16  
#else
    // En ESP32 / PC / STM32 (RAM abundante), damos holgura para redes masivas
    #define SYMBEX_MAX_BLOCKS 64  
#endif

// --------------------------------------------------------
// 2. HAL: OPERACIONES MATEMÁTICAS A NIVEL DE SILICIO
// --------------------------------------------------------
// Envolvemos el popcount para que la librería no dependa de GCC ciegamente.
#if defined(ESP32) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)
    // Hardware nativo de 32 bits (1 ciclo de reloj)
    #define SYMBEX_POPCOUNT32(x) __builtin_popcount(x)
#else
    // Fallback genérico para 8-bits (El compilador inyectará su subrutina segura)
    // Opcionalmente, aquí puedes escribir tu propio popcount de software para 32 bits después.
    #define SYMBEX_POPCOUNT32(x) __builtin_popcount(x) 
#endif

// (Aquí asumo que ya tienes tus macros SYMBEX_READ_BYTE, PROGMEM, etc.)

#endif
