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
 *
 * -DSYMBEX_USE_PIE (opt-in, default OFF) selects the PIE multiplier kernel for symbex_xnor_popcount:
 * 128-bit ee.xorq, then the bit count done by the SIMD unit (bit planes ee.andq/ee.vsr.32 accumulated
 * per byte with ee.vadds.s8, summed by ee.vmulas.u8.accx against a vector of ones). Needs a and b
 * 16-byte aligned (see SYMBEX_ALIGN16 in hal/symbex_hal.h); otherwise that call uses the default path
 * below. Bits past the last full 128-bit block are counted in C, so any n_bits_reales is exact.
 * With SYMBEX_HAL_PRUEBA_SIN_PIE every PIE instruction of that kernel is emulated in C, one to one.
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

#if defined(SYMBEX_USE_PIE)

/*
 * PIE multiplier kernel. One "lote" (batch) = up to PIE_BLOQUES_POR_LOTE blocks of 128 bits:
 *   q2 = a XOR b                                  ee.xorq
 *   8 times: q4 += q2 AND 0x01..01 ; q2 >>= 1     ee.andq, ee.vadds.s8, ee.vsr.32 (SAR = 1)
 *   ACCX = sum_i q4[i] * 1                        ee.vmulas.u8.accx against the ones vector
 * q4 holds, per byte, how many set bits that byte position had across the batch: at most 8 per block,
 * so 15 blocks (120) never reach the s8 saturation limit (127). The 0x01 vector is both the bit mask
 * and the "ones" operand of the multiplier. ee.vsr.32 may be arithmetic: the sign bits it shifts in
 * only reach bits 25..31 of each lane and the mask keeps bits 0, 8, 16, 24, so it does not matter.
 */
#define PIE_BLOQUES_POR_LOTE 15

static const uint8_t pie_unos[16] __attribute__((aligned(16))) = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

#if !defined(SYMBEX_HAL_PRUEBA_SIN_PIE)

#define PIE_PLANO_BIT                     \
        "ee.andq q3, q2, q5 \n\t"         \
        "ee.vadds.s8 q4, q4, q3 \n\t"     \
        "ee.vsr.32 q2, q2 \n\t"

/* Differences (popcount of a XOR b) over n_bloques (1..15) blocks of 128 bits; advances *pa and *pb. */
static inline int32_t pie_dif_lote(const uint32_t** pa, const uint32_t** pb, int n_bloques) {
    int32_t dif;
    uint32_t sar_previo;
    int n = n_bloques;
    /* GCC has no clobber name for SAR on Xtensa: it is saved and restored inside the block */
    asm volatile (
        "rsr.sar %[sar] \n\t"
        "ee.zero.accx \n\t"
        "ee.zero.q q4 \n\t"
        "ee.vld.128.ip q5, %[unos], 0 \n\t"
        "ssai 1 \n\t"
        "1: \n\t"
        "ee.vld.128.ip q0, %[a], 16 \n\t"
        "ee.vld.128.ip q1, %[b], 16 \n\t"
        "ee.xorq q2, q0, q1 \n\t"
        PIE_PLANO_BIT PIE_PLANO_BIT PIE_PLANO_BIT PIE_PLANO_BIT
        PIE_PLANO_BIT PIE_PLANO_BIT PIE_PLANO_BIT
        "ee.andq q3, q2, q5 \n\t"
        "ee.vadds.s8 q4, q4, q3 \n\t"
        "addi %[n], %[n], -1 \n\t"
        "bnez %[n], 1b \n\t"
        "ee.vmulas.u8.accx q4, q5 \n\t"
        "rur.accx_0 %[dif] \n\t"
        "wsr.sar %[sar] \n\t"
        : [a] "+r"(*pa), [b] "+r"(*pb), [n] "+r"(n), [dif] "=&r"(dif), [sar] "=&r"(sar_previo)
        : [unos] "r"(pie_unos)
        : "memory"
    );
    return dif;
}

#undef PIE_PLANO_BIT

#else  /* SYMBEX_HAL_PRUEBA_SIN_PIE: C emulation of each PIE instruction used above, same order */

typedef struct { uint8_t b[16]; } pie_q;

static void emu_vld128_ip(pie_q* q, const uint32_t** p) {          /* ee.vld.128.ip q, p, 16 */
    const uint8_t* bytes = (const uint8_t*)*p;
    for (int i = 0; i < 16; i++) q->b[i] = bytes[i];
    *p += 4;
}
static void emu_xorq(pie_q* d, const pie_q* x, const pie_q* y) {
    for (int i = 0; i < 16; i++) d->b[i] = (uint8_t)(x->b[i] ^ y->b[i]);
}
static void emu_andq(pie_q* d, const pie_q* x, const pie_q* y) {
    for (int i = 0; i < 16; i++) d->b[i] = (uint8_t)(x->b[i] & y->b[i]);
}
static void emu_vadds_s8(pie_q* d, const pie_q* x, const pie_q* y) {   /* signed, saturating */
    for (int i = 0; i < 16; i++) {
        int s = (int)(int8_t)x->b[i] + (int)(int8_t)y->b[i];
        d->b[i] = (uint8_t)(int8_t)(s > 127 ? 127 : (s < -128 ? -128 : s));
    }
}
static void emu_vsr32(pie_q* d, const pie_q* s, unsigned sar) {      /* 4 little-endian 32-bit lanes */
    for (int l = 0; l < 4; l++) {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) v |= (uint32_t)s->b[4 * l + k] << (8 * k);
#if defined(SYMBEX_PIE_EMU_VSR_LOGICO)
        v = v >> sar;
#else
        v = (uint32_t)((int32_t)v >> sar);                              /* arithmetic (worst case) */
#endif
        for (int k = 0; k < 4; k++) d->b[4 * l + k] = (uint8_t)(v >> (8 * k));
    }
}
static void emu_vmulas_u8_accx(uint64_t* accx, const pie_q* x, const pie_q* y) {
    uint64_t s = 0;
    for (int i = 0; i < 16; i++) s += (uint64_t)x->b[i] * y->b[i];
    *accx = (*accx + s) & 0xFFFFFFFFFFULL;                               /* ACCX is 40 bits */
}

/* Batches run by the emulated kernel: lets a PC test prove the PIE path was taken (no fallback). */
unsigned long symbex_pie_prueba_lotes = 0;

static int32_t pie_dif_lote(const uint32_t** pa, const uint32_t** pb, int n_bloques) {
    pie_q q0, q1, q2, q3, q4, q5;
    symbex_pie_prueba_lotes++;
    uint64_t accx = 0;                                                   /* ee.zero.accx */
    for (int i = 0; i < 16; i++) q4.b[i] = 0;                            /* ee.zero.q q4 */
    for (int i = 0; i < 16; i++) q5.b[i] = pie_unos[i];                  /* ee.vld.128.ip q5, unos, 0 */
    const unsigned sar = 1;                                              /* ssai 1 */
    int n = n_bloques;
    do {
        emu_vld128_ip(&q0, pa);
        emu_vld128_ip(&q1, pb);
        emu_xorq(&q2, &q0, &q1);
        for (int k = 0; k < 7; k++) {
            emu_andq(&q3, &q2, &q5);
            emu_vadds_s8(&q4, &q4, &q3);
            emu_vsr32(&q2, &q2, sar);
        }
        emu_andq(&q3, &q2, &q5);
        emu_vadds_s8(&q4, &q4, &q3);
    } while (--n != 0);                                                  /* addi / bnez */
    emu_vmulas_u8_accx(&accx, &q4, &q5);
    return (int32_t)(uint32_t)accx;                                      /* rur.accx_0 */
}

#endif /* SYMBEX_HAL_PRUEBA_SIN_PIE */

static inline int pie_alineado(const uint32_t* a, const uint32_t* b) {
    return (((uintptr_t)a | (uintptr_t)b) & 15u) == 0;
}

/* Same contract as symbex_xnor_popcount; a and b must be 16-byte aligned. */
static int32_t xnor_pie_vmulas(const uint32_t* a, const uint32_t* b, int n_bits_reales) {
    int bloques = n_bits_reales / 128;
    int32_t dif = 0;
    const uint32_t* pa = a;
    const uint32_t* pb = b;
    while (bloques > 0) {
        const int lote = bloques < PIE_BLOQUES_POR_LOTE ? bloques : PIE_BLOQUES_POR_LOTE;
        dif += pie_dif_lote(&pa, &pb, lote);
        bloques -= lote;
    }
    const int bits_cola = n_bits_reales % 128;                           /* tail: < 4 words, in C */
    const int completas = bits_cola / 32;
    const int resto = bits_cola % 32;
    int w = 0;
    for (; w < completas; w++) dif += popcount_swar(pa[w] ^ pb[w]);
    if (resto) dif += popcount_swar((pa[w] ^ pb[w]) & ((1UL << resto) - 1UL));
    return n_bits_reales - 2 * dif;
}

#endif /* SYMBEX_USE_PIE */

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
#if defined(SYMBEX_USE_PIE)
    if (pie_alineado(a, b)) return xnor_pie_vmulas(a, b, n_bits_reales);
#endif
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
#if defined(SYMBEX_USE_PIE) && defined(RUTA_LUT16)
    if (lut16_lista()) return "esp32s3: PIE ee.xorq + ee.vmulas (datos alineados a 16), si no lut16";
    return "esp32s3: PIE ee.xorq + ee.vmulas (datos alineados a 16), si no PIE ee.xorq + SWAR";
#elif defined(SYMBEX_USE_PIE)
    return "esp32s3: PIE ee.xorq + ee.vmulas (datos alineados a 16), si no PIE ee.xorq + SWAR";
#elif defined(RUTA_LUT16)
    if (lut16_lista()) return "esp32s3: lut16 (tabla de 64 KB en RAM)";
    return "esp32s3: PIE ee.xorq + SWAR (NO hubo RAM para lut16)";
#else
    return "esp32s3: PIE ee.xorq + SWAR";
#endif
}

#endif // CONFIG_IDF_TARGET_ESP32S3
