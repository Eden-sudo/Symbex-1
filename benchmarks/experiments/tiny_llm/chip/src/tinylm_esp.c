/* tinylm.c for the ESP32-S3: per-token buffers and the lut8 table in internal RAM, the KV cache in PSRAM. */
#include "esp_heap_caps.h"
#define TINYLM_MALLOC_FAST(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define TINYLM_MALLOC_BIG(n) heap_caps_malloc((n), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
/* cycles per part of a token (main.cpp prints them): see TINYLM_PROF in tinylm.c */
#include "xtensa/hal.h"
uint64_t tinylm_prof[12];
uint32_t tinylm_prof_last;
#define TINYLM_PROF(k) do { uint32_t c_ = xthal_get_ccount(); tinylm_prof[k] += c_ - tinylm_prof_last; \
                            tinylm_prof_last = c_; } while (0)
void tinylm_prof_start(void) { tinylm_prof_last = xthal_get_ccount(); }

/* the output head on lib_symbex's PIE int8 row products (16 per instruction); x copied to a 16-byte aligned,
   zero-padded buffer as symbex_s8_dot_rows asks */
#include <string.h>
#include "hal/symbex_hal.h"
#include "../../c/tinylm.h"   /* emb4, emb2 */
static void tinylm_dot_rows(const int8_t *x, int n, const int8_t *w, int stride, int n_rows, int32_t *out) {
    static int8_t xa[1024] __attribute__((aligned(16)));
    const int padded = (n + 15) & ~15;
    if (padded > (int)sizeof(xa)) {                   /* wider than the buffer: plain C */
        for (int r = 0; r < n_rows; r++, w += stride) {
            int32_t d = 0;
            for (int i = 0; i < n; i++) d += x[i] * w[i];
            out[r] = d;
        }
        return;
    }
    memcpy(xa, x, n);
    memset(xa + n, 0, padded - n);
    symbex_s8_dot_rows(xa, n, w, stride, n_rows, out);
}
#define TINYLM_DOT_ROWS(x, n, w, stride, n_rows, out) tinylm_dot_rows((x), (n), (w), (stride), (n_rows), (out))

/* the attention values: unsigned bytes x unsigned rows, symbex_s8_dot_rows's loop with ee.vmulas.u8.accx (the
   rows of the transposed V cache at any alignment, x 16-byte aligned and zero padded, n <= 1024) */
static void tinylm_dot_rows_u8(const uint8_t *x, int n, const uint8_t *w, int stride, int n_rows, int32_t *out) {
    static uint8_t xa[1024] __attribute__((aligned(16)));
    const int padded = (n + 15) & ~15;
    if (padded > (int)sizeof(xa)) {
        for (int r = 0; r < n_rows; r++, w += stride) {
            int32_t d = 0;
            for (int i = 0; i < n; i++) d += x[i] * w[i];
            out[r] = d;
        }
        return;
    }
    memcpy(xa, x, n);
    memset(xa + n, 0, padded - n);
    const int nv = padded >> 4;
    for (int r = 0; r < n_rows; r++, w += stride) {
        const uint8_t *pw = w, *px = xa;
        const int nl = (int)((((uintptr_t)w & 15U) + (unsigned)n - 1U) >> 4);
        __asm__ __volatile__ ("ee.zero.accx \n\t ee.ld.128.usar.ip q0, %0, 16 \n\t" : "+r"(pw) : : "memory");
        for (int v = 0; v < nl; v++)
            __asm__ __volatile__ ("ee.ld.128.usar.ip q1, %0, 16 \n\t ee.vld.128.ip q2, %1, 16 \n\t"
                                  "ee.src.q.qup q3, q0, q1 \n\t ee.vmulas.u8.accx q2, q3 \n\t"
                                  : "+r"(pw), "+r"(px) : : "memory");
        if (nl < nv)
            __asm__ __volatile__ ("ee.vld.128.ip q2, %0, 16 \n\t ee.src.q q3, q0, q0 \n\t"
                                  "ee.vmulas.u8.accx q2, q3 \n\t" : "+r"(px) : : "memory");
        uint32_t lo;
        __asm__ __volatile__ ("rur.accx_0 %0" : "=r"(lo));   /* < 2^31: the low word of ACCX */
        out[r] = (int32_t)lo;
    }
}
#define TINYLM_DOT_ROWS_U8(x, n, w, stride, n_rows, out) tinylm_dot_rows_u8((x), (n), (w), (stride), (n_rows), (out))

/* the int4 output head (model.bin version 3): per 16 packed bytes (32 weights), high nibbles = bytes & 0xF0 and low
   nibbles = (bytes << 4) & 0xF0 (ee.vsl.32 by 4: a byte's low bits get the byte below's high nibble, masked away), both
   16 x the int4 value as int8, two ee.vmulas.s8.accx with x[32g ..] and x[32g + 16 ..], the sum / 16. Rows of n / 2
   bytes, 16-byte aligned (model.bin keeps them so); otherwise, or n > 1024, plain C (emb4, tinylm.c). */
static void tinylm_dot_rows4(const int8_t *x, int n, const uint8_t *w, int n_rows, int32_t *out) {
    static int8_t xa[1024] __attribute__((aligned(16)));
    static const uint8_t mask[16] __attribute__((aligned(16))) = {
        0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};
    if (n > (int)sizeof(xa) || n % 32 || ((uintptr_t)w & 15U)) {
        for (int r = 0; r < n_rows; r++, w += n / 2) {
            int32_t d = 0;
            for (int i = 0; i < n; i++) d += x[i] * emb4(w, i);
            out[r] = d;
        }
        return;
    }
    memcpy(xa, x, n);
    const int ng = n / 32;
    const int sar = 4;
    for (int r = 0; r < n_rows; r++) {
        const int8_t *px = xa;
        const uint8_t *pm = mask;
        __asm__ __volatile__ ("ee.zero.accx \n\t ee.vld.128.ip q7, %0, 0 \n\t" : "+r"(pm) : : "memory");
        for (int g = 0; g < ng; g++)
            __asm__ __volatile__ ("ee.vld.128.ip q1, %0, 16 \n\t ee.vld.128.ip q2, %1, 16 \n\t"
                                  "ee.vld.128.ip q3, %1, 16 \n\t ee.andq q4, q1, q7 \n\t wsr.sar %2 \n\t"
                                  "ee.vsl.32 q5, q1 \n\t ee.andq q5, q5, q7 \n\t"
                                  "ee.vmulas.s8.accx q2, q5 \n\t ee.vmulas.s8.accx q3, q4 \n\t"
                                  : "+r"(w), "+r"(px) : "r"(sar) : "memory");
        uint32_t lo;
        __asm__ __volatile__ ("rur.accx_0 %0" : "=r"(lo));   /* |16 x sum| < 2^31: the low word of ACCX */
        out[r] = (int32_t)lo >> 4;
    }
}
#define TINYLM_DOT_ROWS4(x, n, w, n_rows, out) tinylm_dot_rows4((x), (n), (w), (n_rows), (out))

/* the int2 output head (model.bin version 4): per 16 packed bytes (64 weights), crumb k moved to the top bits
   ((bytes << (6 - 2k)) & 0xC0 = 64 c as int8, ee.vsl.32: lower bits from the byte below are masked away), four
   ee.vmulas.s8.accx with x[64g + 16k ..], the sum / 64 = sum x c; the dot is 2 sum x c + sum x (values 2c + 1). Rows
   of n / 4 bytes, 16-byte aligned; otherwise, or n > 1024, plain C (emb2, tinylm.h). */
static void tinylm_dot_rows2(const int8_t *x, int n, const uint8_t *w, int n_rows, int32_t *out) {
    static int8_t xa[1024] __attribute__((aligned(16)));
    static const uint8_t mask[16] __attribute__((aligned(16))) = {
        0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0};
    if (n > (int)sizeof(xa) || n % 64 || ((uintptr_t)w & 15U)) {
        for (int r = 0; r < n_rows; r++, w += n / 4) {
            int32_t d = 0;
            for (int i = 0; i < n; i++) d += x[i] * emb2(w, i);
            out[r] = d;
        }
        return;
    }
    memcpy(xa, x, n);
    int32_t sx = 0;
    for (int i = 0; i < n; i++) sx += x[i];
    const int ng = n / 64;
    const int s6 = 6, s4 = 4, s2 = 2;
    for (int r = 0; r < n_rows; r++) {
        const int8_t *px = xa;
        const uint8_t *pm = mask;
        __asm__ __volatile__ ("ee.zero.accx \n\t ee.vld.128.ip q7, %0, 0 \n\t" : "+r"(pm) : : "memory");
        for (int g = 0; g < ng; g++)
            __asm__ __volatile__ ("ee.vld.128.ip q1, %0, 16 \n\t ee.vld.128.ip q2, %1, 16 \n\t"
                                  "ee.vld.128.ip q3, %1, 16 \n\t ee.vld.128.ip q4, %1, 16 \n\t"
                                  "ee.vld.128.ip q5, %1, 16 \n\t"
                                  "wsr.sar %2 \n\t ee.vsl.32 q6, q1 \n\t ee.andq q6, q6, q7 \n\t"
                                  "ee.vmulas.s8.accx q2, q6 \n\t"
                                  "wsr.sar %3 \n\t ee.vsl.32 q6, q1 \n\t ee.andq q6, q6, q7 \n\t"
                                  "ee.vmulas.s8.accx q3, q6 \n\t"
                                  "wsr.sar %4 \n\t ee.vsl.32 q6, q1 \n\t ee.andq q6, q6, q7 \n\t"
                                  "ee.vmulas.s8.accx q4, q6 \n\t"
                                  "ee.andq q6, q1, q7 \n\t ee.vmulas.s8.accx q5, q6 \n\t"
                                  : "+r"(w), "+r"(px) : "r"(s6), "r"(s4), "r"(s2) : "memory");
        uint32_t lo;
        __asm__ __volatile__ ("rur.accx_0 %0" : "=r"(lo));   /* |64 x sum x c| < 2^31: the low word of ACCX */
        out[r] = 2 * ((int32_t)lo >> 6) + sx;
    }
}
#define TINYLM_DOT_ROWS2(x, n, w, n_rows, out) tinylm_dot_rows2((x), (n), (w), (n_rows), (out))
#ifdef TINYLM_SYMBEX_LINEAR
#include "../../c/tinylm_symbex.h"   /* linear layers on lib_symbex's 8-bit x 1-bit PIE kernel */
#endif
#include "../../c/tinylm.c"
