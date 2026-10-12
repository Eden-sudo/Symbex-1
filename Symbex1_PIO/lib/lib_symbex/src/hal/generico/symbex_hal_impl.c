/*
 * symbex_hal_impl.c (generic) -- chooses AT COMPILE TIME how to count bits:
 *   NATIVE : x86 with POPCNT, AArch64, RISC-V with Zbb (64-bit words)
 *   LUT16  : ESP32 (ESP_PLATFORM): 64 KB table in RAM; SWAR if there is no RAM for it
 *   LIBGCC : AVR (Arduino Uno), as always
 *   SWAR   : everything else
 * Flags: -DSYMBEX_POPCOUNT_FORZAR_SWAR, -DSYMBEX_POPCOUNT_FORZAR_LUT16,
 *        -DSYMBEX_POPCOUNT_SIN_LUT16 (ESP32 without the table)
 * On the ESP32-S3, hal/esp32s3/ is used and this file compiles to nothing.
 *
 * SYMBEX v3: symbex_xnor_popcount counts ONLY the first n_bits_reales bits. Padding bits
 * (the rest of the last word and any extra words) never count, whatever their value.
 * Before, padding counted as a match whenever it was equal in a and b (both 0): +1 per
 * padding bit. Where the padding already differed (the convs), the result is identical.
 */

#include <string.h>
#include <stdlib.h>
#include "hal/symbex_hal.h"

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif

#if !defined(CONFIG_IDF_TARGET_ESP32S3)

#if defined(ESP_PLATFORM)
#include "esp_heap_caps.h"
#endif

#if defined(SYMBEX_POPCOUNT_FORZAR_SWAR)
  #define RUTA_SWAR 1
#elif defined(SYMBEX_POPCOUNT_FORZAR_LUT16)
  #define RUTA_LUT16 1
#elif defined(__POPCNT__) || defined(__aarch64__) || defined(__riscv_zbb)
  #define RUTA_NATIVO 1
#elif defined(__AVR__)
  #define RUTA_LIBGCC 1
#elif defined(ESP_PLATFORM) && !defined(SYMBEX_POPCOUNT_SIN_LUT16)
  #define RUTA_LUT16 1
#else
  #define RUTA_SWAR 1
#endif

static inline int32_t popcount_swar(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    return (int32_t)((x * 0x01010101U) >> 24);
}

#if defined(RUTA_LUT16)

/* ---- lut16: 65536-entry table (64 KB) in RAM, allocated on first use ---- */
static uint8_t* tabla16 = 0;
static int estado16 = 0;                     /* 0: not tried | 1: ready | -1: no RAM */

static void preparar_tabla16(void) {
#if defined(ESP_PLATFORM)
    tabla16 = (uint8_t*)heap_caps_malloc(65536, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
    tabla16 = (uint8_t*)malloc(65536);
#endif
    if (!tabla16) { estado16 = -1; return; }
    for (uint32_t i = 0; i < 65536; i++) {
        uint32_t x = i;
        x = x - ((x >> 1) & 0x5555U);
        x = (x & 0x3333U) + ((x >> 2) & 0x3333U);
        x = (x + (x >> 4)) & 0x0F0FU;
        tabla16[i] = (uint8_t)((x & 0xFFU) + (x >> 8));
    }
    estado16 = 1;
}

static inline int lut16_lista(void) {
    if (estado16 == 0) preparar_tabla16();
    return estado16 == 1;
}

#endif

int32_t symbex_hal_popcount32(uint32_t x) {
#if defined(RUTA_LUT16)
    if (lut16_lista()) return tabla16[x & 0xFFFFU] + tabla16[x >> 16];
    return popcount_swar(x);
#elif defined(RUTA_NATIVO) || defined(RUTA_LIBGCC)
    return __builtin_popcount(x);
#else
    return popcount_swar(x);
#endif
}

/* 2*matches - n_bits_reales, counting ONLY the first n_bits_reales bits.
 * It counts the DIFFERENCES (XOR): matches = n - differences, so the result is
 * n_bits_reales - 2*differences. */
int32_t symbex_xnor_popcount(const uint32_t* a, const uint32_t* b, int n_palabras, int n_bits_reales) {
    if (n_bits_reales > n_palabras * 32) n_bits_reales = n_palabras * 32;   /* never read past the buffer */
    if (n_bits_reales <= 0) return 0;
    const int completas = n_bits_reales / 32;
    const int resto = n_bits_reales % 32;
    const uint32_t mascara = resto ? ((1UL << resto) - 1UL) : 0;
    int32_t dif = 0;
    int w = 0;
#if defined(RUTA_LUT16)
    if (lut16_lista()) {
        const uint8_t* t = tabla16;
        for (; w < completas; w++) {
            uint32_t x = a[w] ^ b[w];
            dif += t[x & 0xFFFFU] + t[x >> 16];
        }
        if (resto) {
            uint32_t x = (a[w] ^ b[w]) & mascara;
            dif += t[x & 0xFFFFU] + t[x >> 16];
        }
        return n_bits_reales - 2 * dif;
    }
#endif
#if defined(RUTA_NATIVO) && (UINTPTR_MAX > 0xFFFFFFFFu)
    for (; w + 1 < completas; w += 2) {
        uint64_t x, y;
        memcpy(&x, a + w, sizeof x);
        memcpy(&y, b + w, sizeof y);
        dif += __builtin_popcountll(x ^ y);
    }
#endif
    for (; w < completas; w++) {
#if defined(RUTA_NATIVO) || defined(RUTA_LIBGCC)
        dif += __builtin_popcount(a[w] ^ b[w]);
#else
        dif += popcount_swar(a[w] ^ b[w]);
#endif
    }
    if (resto) dif += symbex_hal_popcount32((a[w] ^ b[w]) & mascara);
    return n_bits_reales - 2 * dif;
}

const char* symbex_hal_nombre(void) {
#if defined(RUTA_LUT16)
    return lut16_lista() ? "generico: lut16 (tabla de 64 KB en RAM)" : "generico: SWAR (NO hubo RAM para lut16)";
#elif defined(RUTA_NATIVO)
    return "generico: popcount NATIVO, palabras de 64 bits";
#elif defined(RUTA_LIBGCC)
    return "generico: __builtin_popcount de libgcc (AVR)";
#else
    return "generico: SWAR en software";
#endif
}

#endif // !CONFIG_IDF_TARGET_ESP32S3
