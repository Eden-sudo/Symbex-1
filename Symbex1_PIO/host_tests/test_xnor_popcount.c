/*
 * test_xnor_popcount.c -- PC test of symbex_xnor_popcount (hal/), bit-exact against a naive
 * bit-by-bit reference. The same file is compiled against each HAL variant by run_host_tests.sh:
 *   generic HAL (portable path: native popcount / SWAR)
 *   esp32s3 HAL with -DSYMBEX_HAL_PRUEBA_SIN_PIE (every PIE instruction emulated in C), default
 *   path (lut16), PIE xor + SWAR, and the PIE multiplier kernel (-DSYMBEX_USE_PIE).
 * Covers random lengths (odd, not multiples of 32 / 128), the 15-block batch boundary, long vectors,
 * worst-case per-byte accumulation (all bits different), garbage in padding bits, unaligned pointers
 * (fallback) and n_bits_reales larger than the buffer (clamp).
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hal/symbex_hal.h"

#if defined(SYMBEX_USE_PIE) && defined(SYMBEX_HAL_PRUEBA_SIN_PIE)
extern unsigned long symbex_pie_prueba_lotes;
#define LOTES_PIE() symbex_pie_prueba_lotes
#else
#define LOTES_PIE() 0UL
#endif

#define MAX_PALABRAS 512                      /* 16384 bits */

static uint32_t buf_a[MAX_PALABRAS + 8] SYMBEX_ALIGN16;
static uint32_t buf_b[MAX_PALABRAS + 8] SYMBEX_ALIGN16;

static uint64_t estado_rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rng32(void) {                 /* xorshift64*, fixed seed: reproducible */
    estado_rng ^= estado_rng >> 12;
    estado_rng ^= estado_rng << 25;
    estado_rng ^= estado_rng >> 27;
    return (uint32_t)((estado_rng * 0x2545F4914F6CDD1DULL) >> 32);
}

static int32_t referencia(const uint32_t* a, const uint32_t* b, int n_palabras, int n_bits) {
    if (n_bits > n_palabras * 32) n_bits = n_palabras * 32;
    if (n_bits <= 0) return 0;
    int32_t coincidencias = 0;
    for (int i = 0; i < n_bits; i++) {
        uint32_t ba = (a[i / 32] >> (i % 32)) & 1u;
        uint32_t bb = (b[i / 32] >> (i % 32)) & 1u;
        coincidencias += (ba == bb);
    }
    return 2 * coincidencias - n_bits;
}

static long casos = 0, fallos = 0;

static void comprobar(const char* nombre, const uint32_t* a, const uint32_t* b, int n_palabras, int n_bits) {
    int32_t esperado = referencia(a, b, n_palabras, n_bits);
    int32_t obtenido = symbex_xnor_popcount(a, b, n_palabras, n_bits);
    casos++;
    if (esperado != obtenido) {
        if (fallos < 20)
            printf("FALLO %s: n_bits=%d n_palabras=%d esperado=%d obtenido=%d\n",
                   nombre, n_bits, n_palabras, (int)esperado, (int)obtenido);
        fallos++;
    }
}

static void llenar(uint32_t* p, int n, int modo) {
    for (int i = 0; i < n; i++) p[i] = modo == 0 ? rng32() : (modo == 1 ? 0u : 0xFFFFFFFFu);
}

int main(void) {
    printf("HAL: %s\n", symbex_hal_nombre());

    /* 1. random vectors, every length 1..1100 bits; padding words/bits are random garbage */
    for (int n = 1; n <= 1100; n++) {
        const int palabras = (n + 31) / 32;
        llenar(buf_a, palabras + 4, 0);
        llenar(buf_b, palabras + 4, 0);
        comprobar("aleatorio", buf_a, buf_b, palabras, n);
        comprobar("aleatorio_palabras_extra", buf_a, buf_b, palabras + 4, n);
    }

    /* 2. long random vectors around and past the 15-block batch (1920 bits) */
    const int largos[] = {127, 128, 129, 1919, 1920, 1921, 1952, 2047, 2048, 3840, 3841, 4096, 5000,
                          8192, 12345, 16384};
    for (unsigned k = 0; k < sizeof largos / sizeof largos[0]; k++) {
        const int n = largos[k];
        const int palabras = (n + 31) / 32;
        for (int rep = 0; rep < 20; rep++) {
            llenar(buf_a, palabras, 0);
            llenar(buf_b, palabras, 0);
            comprobar("largo", buf_a, buf_b, palabras, n);
        }
    }

    /* 3. worst case per byte lane (every bit differs: 8 per byte per block) and best case (equal) */
    for (unsigned k = 0; k < sizeof largos / sizeof largos[0]; k++) {
        const int n = largos[k];
        const int palabras = (n + 31) / 32;
        llenar(buf_a, palabras, 1);
        llenar(buf_b, palabras, 2);
        comprobar("todos_distintos", buf_a, buf_b, palabras, n);
        llenar(buf_b, palabras, 1);
        comprobar("todos_iguales", buf_a, buf_b, palabras, n);
    }

    /* 4. unaligned operands (4 and 8 bytes off): the PIE kernel must step aside, result unchanged */
    const unsigned long lotes_antes = LOTES_PIE();
    for (int n = 1; n <= 600; n += 7) {
        const int palabras = (n + 31) / 32;
        llenar(buf_a, palabras + 4, 0);
        llenar(buf_b, palabras + 4, 0);
        comprobar("desalineado_a", buf_a + 1, buf_b, palabras, n);
        comprobar("desalineado_b", buf_a, buf_b + 2, palabras, n);
    }
    const unsigned long lotes_desalineados = LOTES_PIE() - lotes_antes;

    /* 5. edge arguments: n_bits <= 0, n_bits larger than the buffer (clamped to n_palabras * 32) */
    llenar(buf_a, 8, 0);
    llenar(buf_b, 8, 0);
    comprobar("cero_bits", buf_a, buf_b, 4, 0);
    comprobar("bits_negativos", buf_a, buf_b, 4, -5);
    comprobar("recorte", buf_a, buf_b, 4, 1000);

    printf("casos=%ld fallos=%ld\n", casos, fallos);
#if defined(SYMBEX_USE_PIE) && defined(SYMBEX_HAL_PRUEBA_SIN_PIE)
    printf("lotes PIE emulados=%lu (con datos desalineados: %lu)\n", LOTES_PIE(), lotes_desalineados);
    if (LOTES_PIE() == 0) { printf("FALLO: el kernel PIE nunca se ejecuto\n"); return 1; }
    if (lotes_desalineados != 0) { printf("FALLO: el kernel PIE corrio con datos desalineados\n"); return 1; }
#else
    (void)lotes_desalineados;
#endif
    return fallos == 0 ? 0 : 1;
}
