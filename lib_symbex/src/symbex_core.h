#ifndef SYMBEX_CORE_H
#define SYMBEX_CORE_H

#include <stdint.h>

/**
 * @brief Unified logical XNOR macro.
 */
#define SYMBEX_XNOR(a, b) (~((a) ^ (b)))

#if defined(__AVR__)
     
    /**
     * @brief AVR optimization: Emulated bit counting.
     */
    #define SYMBEX_POPCOUNT32(x) __builtin_popcountl(x)
    #define SYMBEX_BIT_MAC(in, w) ((2 * SYMBEX_POPCOUNT32(SYMBEX_XNOR(in, w))) - 32)

#elif defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
     
    /**
     * @brief Accelerated multiplication function for Xtensa (ESP32).
     * @param in 32-bit input word.
     * @param w 32-bit weight word.
     * @return Mathematical similarity (votes).
     */
    static inline int32_t symbex_xtensa_int8_mac(uint32_t in, uint32_t w) {
        // [!] Replace the inside of this function with your exact int8 routine
        // that achieved 7236 cycles in the benchmark.
        // If you operated on casted memory, you can unpack here.
        // Meanwhile, this safely bridges to native popcount.
        return (2 * __builtin_popcount(SYMBEX_XNOR(in, w))) - 32;
    }

    #define SYMBEX_POPCOUNT32(x) __builtin_popcount(x)
    #define SYMBEX_BIT_MAC(in, w) symbex_xtensa_int8_mac(in, w)

#else
     
    /** @brief Generic fallback for other architectures. */
    #define SYMBEX_POPCOUNT32(x) __builtin_popcount(x)
    #define SYMBEX_BIT_MAC(in, w) ((2 * SYMBEX_POPCOUNT32(SYMBEX_XNOR(in, w))) - 32)

#endif

#endif
