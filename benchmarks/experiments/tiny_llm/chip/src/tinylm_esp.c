/* tinylm.c for the ESP32-S3: per-token buffers and the lut8 table in internal RAM, the KV cache in PSRAM. */
#include "esp_heap_caps.h"
#define TINYLM_MALLOC_FAST(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define TINYLM_MALLOC_BIG(n) heap_caps_malloc((n), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
/* cycles per part of a token (main.cpp prints them): see TINYLM_PROF in tinylm.c */
#include "xtensa/hal.h"
uint64_t tinylm_prof[8];
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
#include "../../c/tinylm.c"
