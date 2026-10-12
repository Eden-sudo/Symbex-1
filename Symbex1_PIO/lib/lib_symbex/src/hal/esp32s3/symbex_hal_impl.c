/*
 * symbex_hal_esp32s3 -- implementation for the ESP32-S3.
 * Counting with lut16 (measured on the S3: 1.44-1.58x faster than SWAR). If there are not
 * 64 KB of RAM for the table, it falls back to the previous path: PIE ee.xorq + SWAR.
 * -DSYMBEX_POPCOUNT_SIN_LUT16 forces the previous path.
 *
 * SYMBEX v3: symbex_xnor_popcount counts ONLY the first n_bits_reales bits (same as
 * hal/generico): padding bits never count. Where the padding already differed (the convs),
 * the result is identical to before.
 * -DSYMBEX_HAL_PRUEBA_SIN_PIE replaces the PIE instruction with C (only for testing on a PC).
 */

#include <stdbool.h>
#include <stdlib.h>
#include "hal/symbex_hal.h"

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif

#if defined(CONFIG_IDF_TARGET_ESP32S3)

#if !defined(SYMBEX_HAL_PRUEBA_SIN_PIE)
#include "esp_heap_caps.h"
#endif

static inline int32_t popcount_swar(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    return (int32_t)((x * 0x01010101U) >> 24);
}

#if !defined(SYMBEX_POPCOUNT_SIN_LUT16)
#define RUTA_LUT16 1

/* ---- lut16: 65536-entry table (64 KB) in RAM, allocated on first use ---- */
static uint8_t* tabla16 = 0;
static int estado16 = 0;                     /* 0: not tried | 1: ready | -1: no RAM */

static void preparar_tabla16(void) {
#if !defined(SYMBEX_HAL_PRUEBA_SIN_PIE)
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

/* XOR of 4 words (128 bits) with PIE: ee.xorq */
static inline void xor128(const uint32_t** pa, const uint32_t** pb, uint32_t* salida) {
#if !defined(SYMBEX_HAL_PRUEBA_SIN_PIE)
    uint32_t* ps = salida;
    asm volatile (
        "ee.vld.128.ip q0, %0, 16 \n\t"
        "ee.vld.128.ip q1, %1, 16 \n\t"
        "ee.xorq q2, q0, q1 \n\t"
        "ee.vst.128.ip q2, %2, 16 \n\t"
        : "+r"(*pa), "+r"(*pb), "+r"(ps)
        :
        : "memory"
    );
#else
    for (int i = 0; i < 4; i++) salida[i] = (*pa)[i] ^ (*pb)[i];
    *pa += 4;
    *pb += 4;
#endif
}

/* Previous path: PIE (ee.xorq, 128 bits) + SWAR. Counts differences only in the real bits. */
static int32_t xnor_pie_swar(const uint32_t* a, const uint32_t* b, int n_bits_reales) {
    const int completas = n_bits_reales / 32;
    const int resto = n_bits_reales % 32;
    int32_t dif = 0;
    int w = 0;
    bool alineado = (((uintptr_t)a % 16) == 0) && (((uintptr_t)b % 16) == 0);
    if (alineado) {
        static uint32_t resultado_xor[4] __attribute__((aligned(16)));
        const uint32_t* ptr_a = a;
        const uint32_t* ptr_b = b;
        for (int g = 0; g < completas / 4; g++) {
            xor128(&ptr_a, &ptr_b, resultado_xor);
            for (int i = 0; i < 4; i++) dif += popcount_swar(resultado_xor[i]);
        }
        w = (completas / 4) * 4;
    }
    for (; w < completas; w++) dif += popcount_swar(a[w] ^ b[w]);
    if (resto) dif += popcount_swar((a[w] ^ b[w]) & ((1UL << resto) - 1UL));
    return n_bits_reales - 2 * dif;
}

int32_t symbex_hal_popcount32(uint32_t x) {
#if defined(RUTA_LUT16)
    if (lut16_lista()) return tabla16[x & 0xFFFFU] + tabla16[x >> 16];
#endif
    return popcount_swar(x);
}

/* 2*matches - n_bits_reales, counting ONLY the first n_bits_reales bits. */
int32_t symbex_xnor_popcount(const uint32_t* a, const uint32_t* b, int n_palabras, int n_bits_reales) {
    if (n_bits_reales > n_palabras * 32) n_bits_reales = n_palabras * 32;   /* never read past the buffer */
    if (n_bits_reales <= 0) return 0;
#if defined(RUTA_LUT16)
    if (lut16_lista()) {
        const uint8_t* t = tabla16;
        const int completas = n_bits_reales / 32;
        const int resto = n_bits_reales % 32;
        int32_t dif = 0;
        int w = 0;
        for (; w < completas; w++) {
            uint32_t x = a[w] ^ b[w];
            dif += t[x & 0xFFFFU] + t[x >> 16];
        }
        if (resto) {
            uint32_t x = (a[w] ^ b[w]) & ((1UL << resto) - 1UL);
            dif += t[x & 0xFFFFU] + t[x >> 16];
        }
        return n_bits_reales - 2 * dif;
    }
#endif
    return xnor_pie_swar(a, b, n_bits_reales);
}

const char* symbex_hal_nombre(void) {
#if defined(RUTA_LUT16)
    if (lut16_lista()) return "esp32s3: lut16 (tabla de 64 KB en RAM)";
    return "esp32s3: PIE ee.xorq + SWAR (NO hubo RAM para lut16)";
#else
    return "esp32s3: PIE ee.xorq + SWAR";
#endif
}

#endif // CONFIG_IDF_TARGET_ESP32S3
