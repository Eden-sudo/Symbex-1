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
#ifdef TINYLM_SYMBEX_LINEAR
#include "../../c/tinylm_symbex.h"   /* linear layers on lib_symbex's 8-bit x 1-bit PIE kernel */
#endif
#include "../../c/tinylm.c"
