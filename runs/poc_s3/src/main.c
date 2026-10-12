#define KRONECKER 1

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_nn.h"

#define N_OUT 256
#define ROW 512
#define WORDS (ROW / 32)
#define REP 50

static inline uint32_t swar(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

static int8_t *w8, *x8, *o8;
static int32_t *b32;
static uint32_t *wb, *xb;
static uint8_t* t16;
static int32_t sal[N_OUT];
static uint32_t x_xor[4] __attribute__((aligned(16)));
static const uint8_t unos[16] __attribute__((aligned(16))) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

static int32_t int8_c(void) {
    int32_t s = 0;
    for (int j = 0; j < N_OUT; j++) {
        int32_t acc = 0;
        const int8_t* w = &w8[j * ROW];
        for (int i = 0; i < ROW; i++) acc += x8[i] * w[i];
        sal[j] = acc; s += acc;
    }
    return s;
}
static int32_t int8_espnn(void) {
    esp_nn_fully_connected_s8(x8, 0, ROW, w8, 0, b32, o8, N_OUT, 0, 0, 1 << 30, -128, 127);
    return o8[0];
}
static int32_t bin_swar(void) {
    int32_t s = 0;
    for (int j = 0; j < N_OUT; j++) {
        const uint32_t* w = &wb[j * WORDS]; uint32_t d = 0;
        for (int p = 0; p < WORDS; p++) d += swar(xb[p] ^ w[p]);
        s += ROW - 2 * (int32_t)d;
    }
    return s;
}
static int32_t bin_lut16(void) {
    int32_t s = 0;
    for (int j = 0; j < N_OUT; j++) {
        const uint32_t* w = &wb[j * WORDS]; uint32_t d = 0;
        for (int p = 0; p < WORDS; p++) { uint32_t v = xb[p] ^ w[p]; d += t16[v & 0xFFFF] + t16[v >> 16]; }
        s += ROW - 2 * (int32_t)d;
    }
    return s;
}
static int32_t bin_pie_swar(void) {
    int32_t s = 0;
    for (int j = 0; j < N_OUT; j++) {
        const uint32_t* pa = xb; const uint32_t* pb = &wb[j * WORDS]; uint32_t d = 0;
        for (int g = 0; g < WORDS / 4; g++) {
            uint32_t* ps = x_xor;
            asm volatile ("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.xorq q2, q0, q1 \n\t"
                          "ee.vst.128.ip q2, %2, 16 \n\t" : "+r"(pa), "+r"(pb), "+r"(ps) : : "memory");
            d += swar(x_xor[0]) + swar(x_xor[1]) + swar(x_xor[2]) + swar(x_xor[3]);
        }
        s += ROW - 2 * (int32_t)d;
    }
    return s;
}
#ifdef KRONECKER
/* EXPERIMENTAL: the SIMD multiplier counts the bits. For k = 0..7: q3 = (x >> k) AND 0x01 per byte,
   ACCX += sum(q3[i] * 1). ACCX ends holding popcount(x) summed over the whole row. */
#define PASO(k) "ssai " #k " \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q7 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
static int32_t bin_pie_mac(void) {
    int32_t s = 0;
    const uint8_t* pu = unos;
    asm volatile ("ee.vld.128.ip q7, %0, 0 \n\t" : "+r"(pu) : : "memory");
    for (int j = 0; j < N_OUT; j++) {
        const uint32_t* pa = xb; const uint32_t* pb = &wb[j * WORDS]; uint32_t d;
        asm volatile ("ee.zero.accx \n\t" : : : );
        for (int g = 0; g < WORDS / 4; g++) {
            asm volatile ("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.xorq q2, q0, q1 \n\t"
                          "ee.andq q3, q2, q7 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
                          PASO(1) PASO(2) PASO(3) PASO(4) PASO(5) PASO(6) PASO(7)
                          : "+r"(pa), "+r"(pb) : : "memory");
        }
        asm volatile ("rur.accx_0 %0 \n\t" : "=r"(d) : : );
        s += ROW - 2 * (int32_t)d;
    }
    return s;
}
#endif

typedef int32_t (*kernel_t)(void);
static void medir(const char* nombre, kernel_t k) {
    int64_t t[REP]; int32_t c = 0;
    for (int r = 0; r < REP; r++) { int64_t a = esp_timer_get_time(); c = k(); t[r] = esp_timer_get_time() - a; }
    for (int i = 1; i < REP; i++) for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { int64_t z = t[j]; t[j] = t[j - 1]; t[j - 1] = z; }
    printf("K:%s,%lld,%ld\n", nombre, (long long)t[REP / 2], (long)c);
    fflush(stdout);
}

void app_main(void) {
    esp_log_level_set("*", ESP_LOG_WARN);
    vTaskDelay(pdMS_TO_TICKS(500));
    const int caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    w8 = heap_caps_aligned_alloc(16, N_OUT * ROW, caps); x8 = heap_caps_aligned_alloc(16, ROW, caps);
    o8 = heap_caps_aligned_alloc(16, N_OUT, caps); b32 = heap_caps_aligned_alloc(16, 4 * N_OUT, caps);
    wb = heap_caps_aligned_alloc(16, N_OUT * ROW / 8, caps); xb = heap_caps_aligned_alloc(16, ROW / 8, caps);
    t16 = heap_caps_malloc(65536, caps);
    if (!w8 || !x8 || !o8 || !b32 || !wb || !xb || !t16) { printf("ERROR:no RAM\n"); return; }
    uint32_t r = 12345;
    for (int i = 0; i < N_OUT * ROW; i++) { r = r * 1103515245u + 12345u; w8[i] = (int8_t)(r >> 24); }
    for (int i = 0; i < ROW; i++) { r = r * 1103515245u + 12345u; x8[i] = (int8_t)(r >> 24); }
    for (int i = 0; i < N_OUT * ROW / 32; i++) { r = r * 1103515245u + 12345u; wb[i] = r; }
    for (int i = 0; i < ROW / 32; i++) { r = r * 1103515245u + 12345u; xb[i] = r; }
    memset(b32, 0, 4 * N_OUT);
    for (int i = 0; i < 65536; i++) t16[i] = (uint8_t)__builtin_popcount(i);
    printf("\nPOC listo\n");
    medir("int8 C", int8_c);
    medir("int8 ESP-NN", int8_espnn);
    medir("bin SWAR", bin_swar);
    medir("bin lut16", bin_lut16);
    medir("bin PIE+SWAR", bin_pie_swar);
#ifdef KRONECKER
    medir("bin PIE+MAC", bin_pie_mac);
#endif
    printf("FIN\n");
    fflush(stdout);
}
