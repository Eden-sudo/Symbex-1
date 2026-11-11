/*
 * symbex_hal_esp32s3 -- implementation for the ESP32-S3.
 * Counting with lut16 (measured on the S3: 1.44-1.58x faster than SWAR). If there are not
 * 64 KB of RAM for the table, it falls back to the previous path: PIE ee.xorq + SWAR.
 * -DSYMBEX_POPCOUNT_NO_LUT16 forces the previous path.
 *
 * SYMBEX v3: symbex_xnor_popcount counts ONLY the first n_real_bits bits (same as
 * hal/generic): padding bits never count. Where the padding already differed (the convs),
 * the result is identical to before.
 * -DSYMBEX_HAL_TEST_NO_PIE replaces the PIE instruction with C (only for testing on a PC).
 *
 * Whole layers (*_rows): the PIE registers (constant vectors, accumulator) are set up ONCE per layer and the
 * table is looked up once per call; before, every neuron paid that setup (~50-65 cycles per neuron, measured),
 * which was more than half of the time on rows of 1-4 words. The C fallback of the first layer (natural layout)
 * uses the SWAR method of ops/symbex_swar.h.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "hal/symbex_hal.h"
#include "ops/symbex_swar.h"

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif

#if defined(CONFIG_IDF_TARGET_ESP32S3)

#if !defined(SYMBEX_HAL_TEST_NO_PIE)
#include "esp_heap_caps.h"
#endif

static inline int32_t popcount_swar(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    return (int32_t)((x * 0x01010101U) >> 24);
}

#if !defined(SYMBEX_POPCOUNT_NO_LUT16)
#define PATH_LUT16 1

/* ---- lut16: 65536-entry table (64 KB) in RAM, allocated on first use ---- */
static uint8_t* table16 = 0;
static int state16 = 0;                     /* 0: not tried | 1: ready | -1: no RAM */

static void prepare_table16(void) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    table16 = (uint8_t*)heap_caps_malloc(65536, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
    table16 = (uint8_t*)malloc(65536);
#endif
    if (!table16) { state16 = -1; return; }
    for (uint32_t i = 0; i < 65536; i++) {
        uint32_t x = i;
        x = x - ((x >> 1) & 0x5555U);
        x = (x & 0x3333U) + ((x >> 2) & 0x3333U);
        x = (x + (x >> 4)) & 0x0F0FU;
        table16[i] = (uint8_t)((x & 0xFFU) + (x >> 8));
    }
    state16 = 1;
}

static inline int lut16_ready(void) {
    if (state16 == 0) prepare_table16();
    return state16 == 1;
}

#endif

/* XOR of 4 words (128 bits) with PIE: ee.xorq */
static inline void xor128(const uint32_t** pa, const uint32_t** pb, uint32_t* output) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    uint32_t* ps = output;
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
    for (int i = 0; i < 4; i++) output[i] = (*pa)[i] ^ (*pb)[i];
    *pa += 4;
    *pb += 4;
#endif
}

/* Previous path: PIE (ee.xorq, 128 bits) + SWAR. Counts differences only in the real bits. */
static int32_t xnor_pie_swar(const uint32_t* a, const uint32_t* b, int n_real_bits) {
    const int full_words = n_real_bits / 32;
    const int remainder = n_real_bits % 32;
    int32_t diff = 0;
    int w = 0;
    bool is_aligned = (((uintptr_t)a % 16) == 0) && (((uintptr_t)b % 16) == 0);
    
    if (is_aligned) {
        static uint32_t xor_result[4] __attribute__((aligned(16)));
        const uint32_t* ptr_a = a;
        const uint32_t* ptr_b = b;
        for (int g = 0; g < full_words / 4; g++) {
            xor128(&ptr_a, &ptr_b, xor_result);
            for (int i = 0; i < 4; i++) diff += popcount_swar(xor_result[i]);
        }
        w = (full_words / 4) * 4;
    }
    
    for (; w < full_words; w++) diff += popcount_swar(a[w] ^ b[w]);
    if (remainder) diff += popcount_swar((a[w] ^ b[w]) & ((1UL << remainder) - 1UL));
    return n_real_bits - 2 * diff;
}

/* the lut16 table if it is available, else NULL: looked up ONCE per call */
static inline const uint8_t* count_table(void) {
#if defined(PATH_LUT16)
    return lut16_ready() ? table16 : 0;
#else
    return 0;
#endif
}

int32_t symbex_hal_popcount32(uint32_t x) {
    const uint8_t* t = count_table();
    return t ? t[x & 0xFFFFU] + t[x >> 16] : popcount_swar(x);
}

/* lut16 path of the XNOR count: 2*matches - n_real_bits over the first n_real_bits bits (t != NULL) */
static inline int32_t xnor_lut16(const uint32_t* a, const uint32_t* b, int n_real_bits, const uint8_t* t) {
    const int full_words = n_real_bits / 32;
    const int remainder = n_real_bits % 32;
    int32_t diff = 0;
    int w = 0;
    for (; w < full_words; w++) {
        uint32_t x = a[w] ^ b[w];
        diff += t[x & 0xFFFFU] + t[x >> 16];
    }
    if (remainder) {
        uint32_t x = (a[w] ^ b[w]) & ((1UL << remainder) - 1UL);
        diff += t[x & 0xFFFFU] + t[x >> 16];
    }
    return n_real_bits - 2 * diff;
}

/* 2*matches - n_real_bits, counting ONLY the first n_real_bits bits. */
int32_t symbex_xnor_popcount(const uint32_t* a, const uint32_t* b, int n_words, int n_real_bits) {
    if (n_real_bits > n_words * 32) n_real_bits = n_words * 32;   /* never read past the buffer */
    if (n_real_bits <= 0) return 0;
    const uint8_t* t = count_table();
    return t ? xnor_lut16(a, b, n_real_bits, t) : xnor_pie_swar(a, b, n_real_bits);
}

void symbex_xnor_popcount_rows(const uint32_t* a, const uint32_t* w, int n_words, int n_real_bits,
                               int n_rows, int32_t* out) {
    if (n_real_bits > n_words * 32) n_real_bits = n_words * 32;
    if (n_real_bits <= 0) {
        for (int r = 0; r < n_rows; r++) out[r] = 0;
        return;
    }
    const uint8_t* t = count_table();
    if (t) {
        for (int r = 0; r < n_rows; r++, w += n_words) out[r] = xnor_lut16(a, w, n_real_bits, t);
    } else {
        for (int r = 0; r < n_rows; r++, w += n_words) out[r] = xnor_pie_swar(a, w, n_real_bits);
    }
}

const char* symbex_hal_name(void) {
#if defined(PATH_LUT16)
    if (lut16_ready()) return "esp32s3: PIE pie_swar (AND / XOR layers) + lut16 (64 KB table: single words, XNOR layers)";
    return "esp32s3: PIE pie_swar (AND / XOR layers) + SWAR (single words, XNOR layers; no RAM for lut16)";
#else
    return "esp32s3: PIE pie_swar (AND / XOR layers) + SWAR (single words, XNOR layers)";
#endif
}

/* ------------------------------------------------------------------------------------------------ */
/* AND arithmetic primitives (used by ops/symbex_and.c): the PIE multiplier does the counting       */
/* ------------------------------------------------------------------------------------------------ */
/* The S3 has no popcount but has SIMD multiply-accumulate (ee.vmulas.u8.accx). These primitives use it
   when the buffers are 16-byte aligned and n_words is a multiple of 4; otherwise portable C, same results. */
static inline int is_aligned16(const void* p) { return ((uintptr_t)p & 15U) == 0; }

#if !defined(SYMBEX_HAL_TEST_NO_PIE)
static const uint8_t and_ones[16] __attribute__((aligned(16))) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static const uint8_t and_selector[16] __attribute__((aligned(16))) = {1,1,2,2,4,4,8,8,16,16,32,32,64,64,128,128};

static inline void load_q6_q7(const uint8_t* q6, const uint8_t* q7) {
    __asm__ __volatile__ ("ee.vld.128.ip q6, %0, 0 \n\t ee.vld.128.ip q7, %1, 0 \n\t" : "+r"(q6), "+r"(q7) : : "memory");
}
static inline uint32_t read_accx(void) { uint32_t v; __asm__ __volatile__ ("rur.accx_0 %0" : "=r"(v)); return v; }

/* Counting with the multiplier, "pie_swar" (benchmarks/experiments/chip_kernels/pie_swar_bench.py: 1.24-1.39x faster than one
   multiply per bit position, and faster than the 64 KB table at every window size). Per 128 bits of (a OP b): count
   inside every byte with vector SWAR -- the bits 4 at a time with the mask 0x11 (every nibble collects its count, at
   most 4), then the two nibbles (at most 8): a byte never passes 68, so ee.vadds.s8, which SATURATES at 127, is exact
   -- then ONE ee.vmulas.u8.accx with a vector of ones adds the 16 byte counts. q5 = 0x11, q6 = 0x0F, q7 = ones. */
static const uint8_t swar_m11[16] __attribute__((aligned(16))) = {0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11};
static const uint8_t swar_m0f[16] __attribute__((aligned(16))) = {0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F};

static inline void load_swar_masks(void) {
    const uint8_t* a = swar_m11; const uint8_t* b = swar_m0f; const uint8_t* c = and_ones;
    __asm__ __volatile__ ("ee.vld.128.ip q5, %0, 0 \n\t ee.vld.128.ip q6, %1, 0 \n\t ee.vld.128.ip q7, %2, 0 \n\t"
                          : "+r"(a), "+r"(b), "+r"(c) : : "memory");
}

#define SWAR_COUNT_Q2                                                                                      \
    "ee.andq q4, q2, q5 \n\t"                                                                              \
    "ssai 1 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"                 \
    "ssai 2 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"                 \
    "ssai 3 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"                 \
    "ssai 4 \n\t ee.vsr.32 q3, q4 \n\t ee.andq q3, q3, q6 \n\t ee.andq q4, q4, q6 \n\t ee.vadds.s8 q4, q4, q3 \n\t" \
    "ee.vmulas.u8.accx q4, q7 \n\t"

/* count(a AND b) with the multiplier; the masks must already be loaded (load_swar_masks) */
static inline int32_t and_pie(const uint32_t* a, const uint32_t* b, int n_words) {
    __asm__ __volatile__ ("ee.zero.accx");
    for (int g = 0; g < n_words / 4; g++) {
        __asm__ __volatile__ ("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.andq q2, q0, q1 \n\t"
                              SWAR_COUNT_Q2
                              : "+r"(a), "+r"(b) : : "memory");
    }
    return (int32_t)read_accx();
}

/* count(a XOR b) with the multiplier: the differences directly; the masks must already be loaded */
static inline int32_t xor_pie(const uint32_t* a, const uint32_t* b, int n_words) {
    __asm__ __volatile__ ("ee.zero.accx");
    for (int g = 0; g < n_words / 4; g++) {
        __asm__ __volatile__ ("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.xorq q2, q0, q1 \n\t"
                              SWAR_COUNT_Q2
                              : "+r"(a), "+r"(b) : : "memory");
    }
    return (int32_t)read_accx();
}
#endif

static inline int32_t and_c(const uint32_t* a, const uint32_t* b, int n_words) {
    int32_t c = 0;
    for (int p = 0; p < n_words; p++) c += popcount_swar(a[p] & b[p]);
    return c;
}

/* count(a AND b): per 128 bits, AND, then bit k of every byte (k = 0..7) times a vector of ones,
   accumulated by the multiplier */
int32_t symbex_and_popcount(const uint32_t* a, const uint32_t* b, int n_words) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    if ((n_words & 3) == 0 && is_aligned16(a) && is_aligned16(b)) {
        load_swar_masks();
        return and_pie(a, b, n_words);
    }
#endif
    return and_c(a, b, n_words);
}

SYMBEX_FAST_CODE void symbex_and_popcount_rows(const uint32_t* a, const uint32_t* w, int stride, int n_rows, int32_t* out) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    if ((stride & 3) == 0 && is_aligned16(a) && is_aligned16(w)) {     /* every row is aligned too */
        load_swar_masks();                                               /* once per layer */
        for (int r = 0; r < n_rows; r++, w += stride) out[r] = and_pie(a, w, stride);
        return;
    }
#endif
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = and_c(a, w, stride);
}

void symbex_xor_popcount_rows(const uint32_t* a, const uint32_t* w, int stride, int n_rows, int32_t* out) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    if ((stride & 3) == 0 && is_aligned16(a) && is_aligned16(w)) {
        load_swar_masks();                                               /* once per layer */
        for (int r = 0; r < n_rows; r++, w += stride) out[r] = xor_pie(a, w, stride);
        return;
    }
#endif
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += stride) {
        int32_t c = 0;
        for (int p = 0; p < stride; p++) {
            const uint32_t x = a[p] ^ w[p];
            c += t ? t[x & 0xFFFFU] + t[x >> 16] : popcount_swar(x);
        }
        out[r] = c;
    }
}

/* the first n_words words of every row; with PIE, n_words rounded up to 4 (the words between are 0 on both sides) */
void symbex_and_popcount_rows_n(const uint32_t* a, const uint32_t* w, int n_words, int stride, int n_rows, int32_t* out) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    const int n4 = (n_words + 3) & ~3;
    if ((stride & 3) == 0 && n4 <= stride && is_aligned16(a) && is_aligned16(w)) {
        load_swar_masks();
        for (int r = 0; r < n_rows; r++, w += stride) out[r] = and_pie(a, w, n4);
        return;
    }
#endif
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = and_c(a, w, n_words);
}

void symbex_xor_popcount_rows_n(const uint32_t* a, const uint32_t* w, int n_words, int stride, int n_rows, int32_t* out) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    const int n4 = (n_words + 3) & ~3;
    if ((stride & 3) == 0 && n4 <= stride && is_aligned16(a) && is_aligned16(w)) {
        load_swar_masks();
        for (int r = 0; r < n_rows; r++, w += stride) out[r] = xor_pie(a, w, n4);
        return;
    }
#endif
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += stride) {
        int32_t c = 0;
        for (int p = 0; p < n_words; p++) {
            const uint32_t x = a[p] ^ w[p];
            c += t ? t[x & 0xFFFFU] + t[x >> 16] : popcount_swar(x);
        }
        out[r] = c;
    }
}

/* XOR with PIE (pie_swar) is the S3's fastest count at every window size (pie_swar_bench.py, count_bench.py) */
int symbex_hal_prefers_xor(void) { return 1; }

void symbex_and_planes_rows(const uint32_t* planes, int k, int low, const uint32_t* w, int n_words,
                            int n_rows, int32_t* out) {
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += n_words) {
        int32_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int p = 0; p < n_words; p++) {
            const uint32_t wv = w[p];                        /* each weight word read once */
            if (t) {
                for (int q = 0; q < k; q++) {
                    const uint32_t x = planes[q * n_words + p] & wv;
                    s[q] += t[x & 0xFFFFU] + t[x >> 16];
                }
            } else {
                for (int q = 0; q < k; q++) s[q] += popcount_swar(planes[q * n_words + p] & wv);
            }
        }
        int32_t total = 0;
        for (int q = 0; q < k; q++) total += s[q] << (low + q);
        out[r] = total;
    }
}

/* The layout symbex_u8_prepare expects: with PIE (aligned, n_words multiple of 4), every 16 inputs
   are interleaved (lane 2m = input m, lane 2m+1 = input 8+m) to match the broadcast of 16 weight bits;
   otherwise natural order. Zero padding up to n_words * 32 bytes. */
static inline int u8_is_interleaved(const uint8_t* workspace, int n_words) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    return (n_words & 3) == 0 && is_aligned16(workspace);
#else
    (void)workspace; (void)n_words;
    return 0;
#endif
}

void symbex_u8_prepare(const uint8_t* x, int n_inputs, uint8_t* workspace, int n_words) {
    const int bytes = n_words * 32;
    for (int i = 0; i < bytes; i++) workspace[i] = 0;
    if (u8_is_interleaved(workspace, n_words)) {
        for (int i = 0; i < n_inputs; i++) {
            const int g = i >> 4, m = i & 15;
            workspace[g * 16 + (m < 8 ? 2 * m : 2 * (m - 8) + 1)] = x[i];
        }
    } else {
        for (int i = 0; i < n_inputs; i++) workspace[i] = x[i];
    }
}

#if !defined(SYMBEX_HAL_TEST_NO_PIE)
/* PIE sum of one row (interleaved workspace, aligned row): q6 = selector, q7 = ones must already be loaded */
static inline int32_t u8_pie(const uint8_t* workspace, const uint32_t* w, int n_words) {
    const uint16_t* pw16 = (const uint16_t*)w;
    const uint8_t* px = workspace;
    __asm__ __volatile__ ("ee.zero.accx");
    for (int g = 0; g < n_words * 2; g += 4) {
        __asm__ __volatile__ ("ee.vldbc.16.ip q0, %0, 2 \n\t ee.vld.128.ip q1, %1, 16 \n\t"
                              "ee.vldbc.16.ip q2, %0, 2 \n\t ee.vld.128.ip q3, %1, 16 \n\t"
                              "ee.andq q0, q0, q6 \n\t ee.andq q2, q2, q6 \n\t"
                              "ee.vcmp.eq.s8 q0, q0, q6 \n\t ee.vcmp.eq.s8 q2, q2, q6 \n\t"
                              "ee.vmulas.u8.accx q1, q0 \n\t ee.vmulas.u8.accx q3, q2 \n\t"
                              "ee.vldbc.16.ip q0, %0, 2 \n\t ee.vld.128.ip q1, %1, 16 \n\t"
                              "ee.vldbc.16.ip q2, %0, 2 \n\t ee.vld.128.ip q3, %1, 16 \n\t"
                              "ee.andq q0, q0, q6 \n\t ee.andq q2, q2, q6 \n\t"
                              "ee.vcmp.eq.s8 q0, q0, q6 \n\t ee.vcmp.eq.s8 q2, q2, q6 \n\t"
                              "ee.vmulas.u8.accx q1, q0 \n\t ee.vmulas.u8.accx q3, q2 \n\t"
                              : "+r"(pw16), "+r"(px) : : "memory");
    }
    return (int32_t)(read_accx() / 255U);            /* exact: ACCX = 255 * sum */
}
#endif

/* C sum of one row: interleaved layout (one input at a time) or natural layout (SWAR, ops/symbex_swar.h) */
static int32_t u8_c(const uint8_t* workspace, const uint32_t* w, int n_words, int interleaved) {
    if (interleaved) {
        int32_t s = 0;
        for (int pos = 0; pos < n_words * 32; pos++) {
            const int g = pos >> 4, l = pos & 15;
            const int i = g * 16 + ((l & 1) ? 8 + (l >> 1) : (l >> 1));
            s += workspace[pos] & -(int32_t)((w[i >> 5] >> (i & 31)) & 1U);
        }
        return s;
    }
    return symbex_swar_u8_masked_sum(workspace, w, (uint16_t)(n_words * 32));
}

/* sum of the inputs whose weight bit is 1. PIE: per 16 inputs, broadcast the 16 weight bits (ee.vldbc.16),
   AND + compare with the selector -> 0xFF / 0x00 mask per lane, multiply the inputs by it: ACCX = 255 * sum.
   4 groups of 16 per block. */
int32_t symbex_u8_sum_bits(const uint8_t* workspace, const uint32_t* w, int n_words) {
    const int interleaved = u8_is_interleaved(workspace, n_words);
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    if (interleaved && is_aligned16(w)) {
        load_q6_q7(and_selector, and_ones);
        return u8_pie(workspace, w, n_words);
    }
#endif
    return u8_c(workspace, w, n_words, interleaved);
}

void symbex_u8_sum_bits_rows(const uint8_t* workspace, int n_inputs, const uint32_t* w, int stride, int n_rows,
                             int32_t* out) {
    const int interleaved = u8_is_interleaved(workspace, stride);
    if (n_inputs > stride * 32) n_inputs = stride * 32;
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    if (interleaved && is_aligned16(w)) {                                /* stride % 4 == 0: every row aligned */
        load_q6_q7(and_selector, and_ones);                              /* once per layer */
        for (int r = 0; r < n_rows; r++, w += stride) out[r] = u8_pie(workspace, w, stride);
        return;
    }
#endif
    if (!interleaved) {                         /* natural layout: only the real inputs (SWAR) */
        for (int r = 0; r < n_rows; r++, w += stride) out[r] = symbex_swar_u8_masked_sum(workspace, w, (uint16_t)n_inputs);
        return;
    }
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = u8_c(workspace, w, stride, interleaved);
}

/* int8 window x int8 rows (a first layer with int8 weights): out[r] = sum_{i < n} x[i] * w[r * stride + i].
   PIE: 16 products per ee.vmulas.s8.accx. The rows keep any alignment (ee.ld.128.usar.ip + ee.src.q.qup realign them),
   so the weights stay where the exporter put them, in flash or in RAM. Each row is read in whole aligned 16-byte
   blocks, only the blocks that hold bytes of the row (never past the weight array); the lanes of the last vector past
   the row's end hold other bytes, and they meet the zero padding of x. */
void symbex_s8_dot_rows(const int8_t* x, int n, const int8_t* w, int stride, int n_rows, int32_t* out) {
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
    const int nv = (n + 15) >> 4;
    for (int r = 0; r < n_rows; r++, w += stride) {
        const int8_t* pw = w;
        const int8_t* px = x;
        /* vectors that load the next block: the row touches 1 + nl aligned blocks, nl = nv or nv - 1 */
        const int nl = (int)((((uintptr_t)w & 15U) + (unsigned)n - 1U) >> 4);
        __asm__ __volatile__ ("ee.zero.accx \n\t ee.ld.128.usar.ip q0, %0, 16 \n\t" : "+r"(pw) : : "memory");
        for (int v = 0; v < nl; v++)
            __asm__ __volatile__ ("ee.ld.128.usar.ip q1, %0, 16 \n\t ee.vld.128.ip q2, %1, 16 \n\t"
                                  "ee.src.q.qup q3, q0, q1 \n\t ee.vmulas.s8.accx q2, q3 \n\t"
                                  : "+r"(pw), "+r"(px) : : "memory");
        if (nl < nv)                                 /* the last vector: its bytes past q0 are past the row (x is 0) */
            __asm__ __volatile__ ("ee.vld.128.ip q2, %0, 16 \n\t ee.src.q q3, q0, q0 \n\t"
                                  "ee.vmulas.s8.accx q2, q3 \n\t" : "+r"(px) : : "memory");
        out[r] = (int32_t)read_accx();               /* |sum| < 2^31: the low 32 bits of the 40-bit ACCX */
    }
#else
    for (int r = 0; r < n_rows; r++, w += stride) {
        int32_t s = 0;
        for (int i = 0; i < n; i++) s += (int32_t)x[i] * w[i];
        out[r] = s;
    }
#endif
}

/* ---- the 3 x 3 int8 stem, vectorized over the output channels ------------------------------------------------
   Per group of 16 channels and per output pixel: QACC (16 lanes, 20-bit) starts at -thr, then 27 ee.vsmulas.s8.qacc,
   each the 16 channels' weights of one tap times that tap's pixel byte (broadcast from a register holding the window
   row), then ee.srcmb.s8.qacc saturates the lanes to int8 and keeps their sign: bit = lane >= 0 = dot >= thr.
   The image rows go through a ring of 3 padded rows (int8 pixel - 128, a zero pixel left and right, rows outside the
   image all zero), so a window row is one unaligned 9-byte load and the borders need no special case.
   (benchmarks/experiments/chip_kernels/stem_i8_pie_bench.py: ResNet-8's stem 4.19 -> 0.73 ms, exact.) */
#define STEM3_QLIM 524287
#define STEM3_RING_ROW(w) ((((w) + 2) * 3 + 15) & ~15)        /* one padded row: whole 16-byte blocks */
/* channel j of a group runs in lane STEM3_LANE(j): the 4 words of the 16 saturated lanes then pack into 16 bits with
   shifts and masks only (stem3_pack16) */
#define STEM3_LANE(j) (4 * ((j) & 3) + ((j) >> 2))

/* QACC as ee.ld.qacc_* read it (ESP-DL's bias layout, checked on the chip): bytes 0-19 lanes 0-7, bytes 32-51 lanes
   8-15, 20 bits each, little-endian */
static void stem3_qacc_put(uint8_t* q, int lane, int32_t v) {
    uint8_t* h = q + (lane >> 3) * 32;
    const int bit = (lane & 7) * 20;
    const uint32_t u = (uint32_t)v & 0xFFFFFu;
    for (int b = 0; b < 20; b++) {
        const int p = bit + b;
        if ((u >> b) & 1u) h[p >> 3] |= (uint8_t)(1u << (p & 7));
    }
}

/* a_i = lanes 4i..4i+3 (one int8 byte each): bit 4j + i of the result = lane 4i + j >= 0 = channel 4j + i */
static inline uint32_t stem3_pack16(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    const uint32_t M = 0x01010101u;
    uint32_t m = ((a0 >> 7) & M) | ((a1 >> 6) & (M << 1)) | ((a2 >> 5) & (M << 2)) | ((a3 >> 4) & (M << 3));
    m ^= 0x0F0F0F0Fu;
    return (m & 0xFu) | ((m >> 4) & 0xF0u) | ((m >> 8) & 0xF00u) | ((m >> 12) & 0xF000u);
}

SYMBEX_FAST_CODE static void stem3_ring_fill(int8_t* dst, const uint8_t* img, int y, int h, int w) {
    memset(dst, 0, STEM3_RING_ROW(w));
    if (y < 0 || y >= h) return;
    const uint8_t* src = img + (size_t)y * w * 3;
    for (int i = 0; i < w * 3; i++) dst[3 + i] = (int8_t)(src[i] ^ 0x80u);
}

#if !defined(SYMBEX_HAL_TEST_NO_PIE)
/* a window row: realigned from the 2 aligned blocks of its first and last byte (never past them) */
#define STEM3_ROW(p, q) "addi %[a0], " p ", 8 \n\t ee.ld.128.usar.ip q0, " p ", 0 \n\t ee.vld.128.ip q1, %[a0], 0 \n\t" \
                        "ee.src.q " q ", q0, q1 \n\t"
/* tap i: its 16 weights (in q[i % 3], loaded 2 taps before) times byte sel of the row; the weights of tap i + 2 load
   meanwhile */
#define STEM3_T(wc, wn, q, sel) "ee.vsmulas.s8.qacc.ld.incp " #wn ", %[w], " #wc ", " #q ", " #sel " \n\t"
#define STEM3_TL(wc, q, sel) "ee.vsmulas.s8.qacc " #wc ", " #q ", " #sel " \n\t"
#define STEM3_PIXEL                                                                                                 \
    STEM3_ROW("%[p0]", "q4") STEM3_ROW("%[p1]", "q5") STEM3_ROW("%[p2]", "q6")                                      \
    "ee.ld.qacc_l.l.128.ip %[q], 16 \n\t ee.ld.qacc_l.h.32.ip %[q], 16 \n\t"                                       \
    "ee.ld.qacc_h.l.128.ip %[q], 16 \n\t ee.ld.qacc_h.h.32.ip %[q], 16 \n\t"                                       \
    "ee.vld.128.ip q0, %[w], 16 \n\t ee.vld.128.ip q1, %[w], 16 \n\t"                                             \
    STEM3_T(q0, q2, q4, 0) STEM3_T(q1, q0, q4, 1) STEM3_T(q2, q1, q4, 2) STEM3_T(q0, q2, q4, 3)                    \
    STEM3_T(q1, q0, q4, 4) STEM3_T(q2, q1, q4, 5) STEM3_T(q0, q2, q4, 6) STEM3_T(q1, q0, q4, 7)                    \
    STEM3_T(q2, q1, q4, 8) STEM3_T(q0, q2, q5, 0) STEM3_T(q1, q0, q5, 1) STEM3_T(q2, q1, q5, 2)                    \
    STEM3_T(q0, q2, q5, 3) STEM3_T(q1, q0, q5, 4) STEM3_T(q2, q1, q5, 5) STEM3_T(q0, q2, q5, 6)                    \
    STEM3_T(q1, q0, q5, 7) STEM3_T(q2, q1, q5, 8) STEM3_T(q0, q2, q6, 0) STEM3_T(q1, q0, q6, 1)                    \
    STEM3_T(q2, q1, q6, 2) STEM3_T(q0, q2, q6, 3) STEM3_T(q1, q0, q6, 4) STEM3_T(q2, q1, q6, 5)                    \
    STEM3_T(q0, q2, q6, 6) STEM3_TL(q1, q6, 7) STEM3_TL(q2, q6, 8)                                                  \
    "movi %[a0], 0 \n\t ee.srcmb.s8.qacc q2, %[a0], 0 \n\t"                                                        \
    "ee.movi.32.a q2, %[a0], 0 \n\t ee.movi.32.a q2, %[a1], 1 \n\t"                                                \
    "ee.movi.32.a q2, %[a2], 2 \n\t ee.movi.32.a q2, %[a3], 3 \n\t"
#endif

SYMBEX_FAST_CODE int symbex_s8_stem3_bits(const uint8_t* img, int in_h, int in_w, int stride, int out_h, int out_w,
                                          const int8_t* weights, int row_bytes, const int32_t* thresholds, int out_ch,
                                          uint32_t* out) {
    const int pout = (out_ch + 31) / 32;
    if (in_w > SYMBEX_S8_STEM3_MAX_W || in_w < 1 || (stride != 1 && stride != 2)) return -1;
    for (int o = 0; o < out_ch; o++) {                /* the checks first: refuse before writing anything */
        int32_t b = 0;
        for (int i = 0; i < 27; i++) { const int v = weights[o * row_bytes + i]; b += 128 * (v < 0 ? -v : v); }
        const int32_t t = thresholds[o];
        if (t > b || t <= -b) continue;               /* a constant bit */
        if ((t < 0 ? -t : t) + b > STEM3_QLIM) return -1;
    }
    const int rb = STEM3_RING_ROW(in_w);
    int8_t ring[3 * rb] __attribute__((aligned(16)));
    int8_t wt[27 * 16 + 32] __attribute__((aligned(16)));   /* + 2 vectors: the loads 2 taps ahead read past 27 */
    uint8_t qi[64] __attribute__((aligned(16)));
    for (int g = 0; g * 16 < out_ch; g++) {
        const int m = out_ch - g * 16 < 16 ? out_ch - g * 16 : 16;
        uint32_t live = 0, ones = 0;                  /* live: computed channels; ones: channels always 1 */
        memset(wt, 0, sizeof wt);
        memset(qi, 0, sizeof qi);
        for (int j = 0; j < m; j++) {
            const int8_t* r = weights + (g * 16 + j) * row_bytes;
            int32_t b = 0;
            for (int i = 0; i < 27; i++) { wt[i * 16 + STEM3_LANE(j)] = r[i]; b += 128 * (r[i] < 0 ? -r[i] : r[i]); }
            const int32_t t = thresholds[g * 16 + j];
            if (t > b) continue;                      /* always 0 */
            if (t <= -b) { ones |= 1u << j; continue; }
            live |= 1u << j;
            stem3_qacc_put(qi, STEM3_LANE(j), -t);
        }
        const int sh = (g & 1) * 16, wd = g >> 1;
        int have = -2;                                /* the last image row in the ring (row y in slot (y + 3) % 3) */
        for (int oy = 0; oy < out_h; oy++) {
            const int iy0 = oy * stride - 1;
            for (int y = have + 1 > iy0 ? have + 1 : iy0; y <= iy0 + 2; y++)
                stem3_ring_fill(ring + ((y + 3) % 3) * rb, img, y, in_h, in_w);
            have = iy0 + 2;
            const int8_t* r0 = ring + ((iy0 + 3) % 3) * rb;
            const int8_t* r1 = ring + ((iy0 + 4) % 3) * rb;
            const int8_t* r2 = ring + ((iy0 + 5) % 3) * rb;
            uint32_t* o = out + (size_t)oy * out_w * pout + wd;
            for (int ox = 0; ox < out_w; ox++, o += pout) {
                const int8_t *p0 = r0 + ox * stride * 3, *p1 = r1 + ox * stride * 3, *p2 = r2 + ox * stride * 3;
                uint32_t a0, a1, a2, a3;
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
                const int8_t* wp = wt;
                const uint8_t* qp = qi;
                __asm__ __volatile__ (STEM3_PIXEL
                                      : [p0] "+r"(p0), [p1] "+r"(p1), [p2] "+r"(p2), [w] "+r"(wp), [q] "+r"(qp),
                                        [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3)
                                      : : "memory");
#else               /* the tests: the same lanes in C, each saturated to int8 as ee.srcmb.s8.qacc does */
                const int8_t* rows[3] = {p0, p1, p2};
                uint32_t a[4] = {0, 0, 0, 0};
                for (int lane = 0; lane < 16; lane++) {
                    int32_t v = 0;                    /* the 20-bit start value back from qi */
                    const uint8_t* h = qi + (lane >> 3) * 32;
                    for (int b = 0; b < 20; b++) {
                        const int p = (lane & 7) * 20 + b;
                        v |= (int32_t)((h[p >> 3] >> (p & 7)) & 1u) << b;
                    }
                    if (v & 0x80000) v -= 0x100000;
                    for (int i = 0; i < 27; i++) v += (int32_t)wt[i * 16 + lane] * rows[i / 9][i % 9];
                    const int32_t sat = v > 127 ? 127 : v < -128 ? -128 : v;
                    a[lane >> 2] |= (uint32_t)(uint8_t)(int8_t)sat << ((lane & 3) * 8);
                }
                a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
#endif
                const uint32_t bits = ((stem3_pack16(a0, a1, a2, a3) & live) | ones) << sh;
                if (sh) *o |= bits; else *o = bits;   /* the first group of a word writes it whole */
            }
        }
    }
    return 0;
}

/* ---- the same stem with the 16 lanes on 16 pixels of one row: the bits land straight in the vertical engine's planes -
   Stride 1, width a multiple of 16. Per output channel and block of 16 pixels: QACC starts at -thr in every lane, then
   27 ee.vsmulas.s8.qacc, each one tap's 16 pixels (one aligned vector) times that tap's weight (broadcast from the
   channel's 2 weight registers), ee.srcmb.s8.qacc, and the 16 signs are 16 bits of the channel's plane: no HWC map and
   no transpose. Ring of 3 image rows, each holding per block of 16 pixels its 9 vectors [kw][color] already shifted
   (pixel - 128, zero outside the image), so a tap is one aligned load. */
#define STEM3P_BLOCK 144                                     /* bytes of one block's 9 vectors in a ring row */
#define STEM3P_DROW (SYMBEX_S8_STEM3_MAX_W + 32)              /* one color of a row: 16 zero bytes, the pixels, 16 */
typedef uint16_t __attribute__((may_alias)) stem3p_u16;
_Static_assert(STEM3P_DROW == 160, "stem3p_ring_fill's step from one color to the next (128 = 160 - 32)");

/* lane x (a_k = lanes 4k..4k+3) >= 0 -> bit x, as STEM3P_SIGNS does it on the chip: a word's 4 signs (bit 7 of its
   bytes, at 0, 8, 16, 24 after the shift) gathered into bits 0..3 by two shifted adds (no two bits meet) */
static inline uint32_t stem3p_pack16(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    const uint32_t a[4] = {a0, a1, a2, a3};
    uint32_t s = 0;
    for (int k = 0; k < 4; k++) {
        uint32_t m = (a[k] >> 7) & 0x01010101u;
        m += m >> 7;
        m += m >> 14;
        s |= (m & 0xFu) << (4 * k);
    }
    return 0xFFFFu - s;
}

/* image row y into a ring row; d: the 3 colors of the row as int8, with zero bytes at x = -1 and x = w */
SYMBEX_FAST_CODE static void stem3p_ring_fill(int8_t* dst, int8_t* d, const uint8_t* img, int y, int h, int w) {
    const int nb = w / 16;
    if (y < 0 || y >= h) { memset(dst, 0, (size_t)nb * STEM3P_BLOCK); return; }
    const uint8_t* src = img + (size_t)y * w * 3;
    for (int x = 0; x < w; x++, src += 3) {
        d[16 + x] = (int8_t)(src[0] ^ 0x80u);
        d[STEM3P_DROW + 16 + x] = (int8_t)(src[1] ^ 0x80u);
        d[2 * STEM3P_DROW + 16 + x] = (int8_t)(src[2] ^ 0x80u);
    }
    for (int b = 0; b < nb; b++) {
        /* per color the aligned blocks X, Y, Z around pixels 16 b .. + 15 (Y): kw 0 = bytes 15.. of X:Y, kw 1 = Y,
           kw 2 = bytes 1.. of Y:Z */
        const int8_t* s = d + b * 16;
        int8_t* o = dst + b * STEM3P_BLOCK;
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
        const int s15 = 15, s1 = 1;
        __asm__ __volatile__ (
#define STEM3P_FILL_COLOR                                                                                           \
            "ee.vld.128.ip q0, %[s], 16 \n\t ee.vld.128.ip q1, %[s], 16 \n\t ee.vld.128.ip q2, %[s], 128 \n\t"        \
            "wur.sar_byte %[s15] \n\t ee.src.q q3, q0, q1 \n\t wur.sar_byte %[s1] \n\t ee.src.q q4, q1, q2 \n\t"       \
            "ee.vst.128.ip q3, %[o], 48 \n\t ee.vst.128.ip q1, %[o], 48 \n\t ee.vst.128.ip q4, %[o], -80 \n\t"
            STEM3P_FILL_COLOR STEM3P_FILL_COLOR STEM3P_FILL_COLOR
#undef STEM3P_FILL_COLOR
            : [s] "+r"(s), [o] "+r"(o) : [s15] "r"(s15), [s1] "r"(s1) : "memory");
#else
        for (int c = 0; c < 3; c++)
            for (int kw = 0; kw < 3; kw++) memcpy(o + (kw * 3 + c) * 16, s + c * STEM3P_DROW + 15 + kw, 16);
#endif
    }
}

#if !defined(SYMBEX_HAL_TEST_NO_PIE)
/* tap i: its 16 pixels (in q2 + i % 3, loaded 2 taps before) times weight sel of w; the pixels of tap i + 2 load
   meanwhile from row p */
#define STEM3P_T(n, p, c, w, sel) "ee.vsmulas.s8.qacc.ld.incp " #n ", %[" #p "], " #c ", " #w ", " #sel " \n\t"
#define STEM3P_TL(c, w, sel) "ee.vsmulas.s8.qacc " #c ", " #w ", " #sel " \n\t"
/* lane word k of q5 -> its 4 signs (1 = negative) in bits 0..3 of a1 (4 bytes' bit 7 at 0, 8, 16, 24, gathered by
   two shifted adds: no two bits meet) */
#define STEM3P_SIGNS(k) "ee.movi.32.a q5, %[a1], " #k " \n\t srli %[a1], %[a1], 7 \n\t and %[a1], %[a1], %[m] \n\t"   \
                        "srli %[a2], %[a1], 7 \n\t add %[a1], %[a1], %[a2] \n\t srli %[a2], %[a1], 14 \n\t"           \
                        "add %[a1], %[a1], %[a2] \n\t extui %[a1], %[a1], 0, 4 \n\t"
/* one block of 16 pixels of one channel, its 16 bits stored at o (o += 2); QACC's and the weights' pointers end where
   they started, the rows' at the next block */
#define STEM3P_BLOCK16                                                                                              \
    "ee.ld.qacc_l.l.128.ip %[q], 16 \n\t ee.ld.qacc_l.h.32.ip %[q], 16 \n\t"                                       \
    "ee.ld.qacc_h.l.128.ip %[q], 16 \n\t ee.ld.qacc_h.h.32.ip %[q], -48 \n\t"                                      \
    "ee.vld.128.ip q0, %[w], 16 \n\t ee.vld.128.ip q1, %[w], -16 \n\t"                                            \
    "ee.vld.128.ip q2, %[p0], 16 \n\t ee.vld.128.ip q3, %[p0], 16 \n\t"                                           \
    STEM3P_T(q4, p0, q2, q0, 0) STEM3P_T(q2, p0, q3, q0, 1) STEM3P_T(q3, p0, q4, q0, 2) STEM3P_T(q4, p0, q2, q0, 3) \
    STEM3P_T(q2, p0, q3, q0, 4) STEM3P_T(q3, p0, q4, q0, 5) STEM3P_T(q4, p0, q2, q0, 6) STEM3P_T(q2, p1, q3, q0, 7) \
    STEM3P_T(q3, p1, q4, q0, 8) STEM3P_T(q4, p1, q2, q0, 9) STEM3P_T(q2, p1, q3, q0, 10)                            \
    STEM3P_T(q3, p1, q4, q0, 11) STEM3P_T(q4, p1, q2, q0, 12) STEM3P_T(q2, p1, q3, q0, 13)                          \
    STEM3P_T(q3, p1, q4, q0, 14) STEM3P_T(q4, p1, q2, q0, 15) STEM3P_T(q2, p2, q3, q1, 0)                           \
    STEM3P_T(q3, p2, q4, q1, 1) STEM3P_T(q4, p2, q2, q1, 2) STEM3P_T(q2, p2, q3, q1, 3) STEM3P_T(q3, p2, q4, q1, 4) \
    STEM3P_T(q4, p2, q2, q1, 5) STEM3P_T(q2, p2, q3, q1, 6) STEM3P_T(q3, p2, q4, q1, 7) STEM3P_T(q4, p2, q2, q1, 8) \
    STEM3P_TL(q3, q1, 9) STEM3P_TL(q4, q1, 10)                                                                      \
    "movi %[a0], 0 \n\t ee.srcmb.s8.qacc q5, %[a0], 0 \n\t"                                                        \
    STEM3P_SIGNS(0) "mov %[a0], %[a1] \n\t"                                                                         \
    STEM3P_SIGNS(1) "slli %[a1], %[a1], 4 \n\t or %[a0], %[a0], %[a1] \n\t"                                         \
    STEM3P_SIGNS(2) "slli %[a1], %[a1], 8 \n\t or %[a0], %[a0], %[a1] \n\t"                                         \
    STEM3P_SIGNS(3) "slli %[a1], %[a1], 12 \n\t or %[a0], %[a0], %[a1] \n\t"                                        \
    "movi %[a1], -1 \n\t extui %[a1], %[a1], 0, 16 \n\t sub %[a0], %[a1], %[a0] \n\t"                              \
    "s16i %[a0], %[o], 0 \n\t addi %[o], %[o], 2 \n\t"
#endif

SYMBEX_FAST_CODE int symbex_s8_stem3_planes(const uint8_t* img, int in_h, int in_w, const int8_t* weights,
                                            int row_bytes, const int32_t* thresholds, int out_ch, uint32_t* planes,
                                            int plane_stride) {
    if (in_w > SYMBEX_S8_STEM3_MAX_W || in_w < 16 || in_w % 16) return -1;
    for (int o = 0; o < out_ch; o++) {                /* the checks first: refuse before writing anything */
        int32_t b = 0;
        for (int i = 0; i < 27; i++) { const int v = weights[o * row_bytes + i]; b += 128 * (v < 0 ? -v : v); }
        const int32_t t = thresholds[o];
        if (t > b || t <= -b) continue;               /* a constant bit */
        if ((t < 0 ? -t : t) + b > STEM3_QLIM) return -1;
    }
    const int nb = in_w / 16, rb = nb * STEM3P_BLOCK;
    int8_t ring[3 * rb] __attribute__((aligned(16)));
    int8_t d[3 * STEM3P_DROW] __attribute__((aligned(16)));
    int8_t wt[16 * 32] __attribute__((aligned(16)));     /* per channel its 27 weights, zero padded to 2 vectors */
    uint8_t qi[16 * 64] __attribute__((aligned(16)));    /* per channel QACC's start: -thr in all 16 lanes */
    uint8_t kind[16];                                    /* 0: always 0, 1: always 1, 2: computed */
    memset(d, 0, sizeof d);
    for (int g = 0; g * 16 < out_ch; g++) {
        const int m = out_ch - g * 16 < 16 ? out_ch - g * 16 : 16;
        memset(wt, 0, sizeof wt);
        memset(qi, 0, sizeof qi);
        for (int j = 0; j < m; j++) {
            const int8_t* r = weights + (g * 16 + j) * row_bytes;
            int32_t b = 0;
            for (int i = 0; i < 27; i++) { wt[j * 32 + i] = r[i]; b += 128 * (r[i] < 0 ? -r[i] : r[i]); }
            const int32_t t = thresholds[g * 16 + j];
            kind[j] = t > b ? 0 : t <= -b ? 1 : 2;
            if (kind[j] == 2) {                       /* lanes 0-7 bytewise, then lanes 8-15 the same 20 bytes */
                uint8_t* q = qi + j * 64;
                const uint32_t u = (uint32_t)-t & 0xFFFFFu;
                for (int lane = 0; lane < 8; lane++) {
                    const int bit = lane * 20;
                    const uint32_t v = u << (bit & 7);
                    for (int k = 0; k < 4 && (bit >> 3) + k < 20; k++) q[(bit >> 3) + k] |= (uint8_t)(v >> (8 * k));
                }
                memcpy(q + 32, q, 20);
            }
        }
        int have = -2;                                /* the last image row in the ring (row y in slot (y + 3) % 3) */
        for (int oy = 0; oy < in_h; oy++) {
            const int iy0 = oy - 1;
            for (int y = have + 1 > iy0 ? have + 1 : iy0; y <= iy0 + 2; y++)
                stem3p_ring_fill(ring + ((y + 3) % 3) * rb, d, img, y, in_h, in_w);
            have = iy0 + 2;
            const int8_t* r0 = ring + ((iy0 + 3) % 3) * rb;
            const int8_t* r1 = ring + ((iy0 + 4) % 3) * rb;
            const int8_t* r2 = ring + ((iy0 + 5) % 3) * rb;
            for (int j = 0; j < m; j++) {
                /* the 16 bits of pixels oy * w + 16 b .. + 15: half a word of the plane (stem3p_u16: may alias the
                   words; a memcpy here was a call per block) */
                stem3p_u16* o = (stem3p_u16*)(void*)(planes + (size_t)(g * 16 + j) * plane_stride) + (size_t)oy * nb;
                if (kind[j] != 2) {
                    for (int b = 0; b < nb; b++) o[b] = kind[j] ? 0xFFFFu : 0;
                    continue;
                }
                const int8_t *p0 = r0, *p1 = r1, *p2 = r2;   /* each block leaves them at the next one */
#if !defined(SYMBEX_HAL_TEST_NO_PIE)
                const int8_t* wp = wt + j * 32;
                const uint8_t* qp = qi + j * 64;
                const uint32_t mk = 0x01010101u;
                int n = nb;
                uint32_t a0, a1, a2;
                __asm__ __volatile__ ("1: \n\t" STEM3P_BLOCK16 "addi %[n], %[n], -1 \n\t bnez %[n], 1b \n\t"
                                      : [p0] "+r"(p0), [p1] "+r"(p1), [p2] "+r"(p2), [w] "+r"(wp), [q] "+r"(qp),
                                        [o] "+r"(o), [n] "+r"(n), [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2)
                                      : [m] "r"(mk) : "memory");
#else
                for (int b = 0; b < nb; b++) {   /* the tests: the same lanes in C, saturated as ee.srcmb.s8.qacc does */
                    const int8_t* rows[3] = {p0, p1, p2};
                    uint32_t a[4] = {0, 0, 0, 0};
                    for (int lane = 0; lane < 16; lane++) {
                        int32_t v = 0;                /* the 20-bit start value back from qi */
                        const uint8_t* h = qi + j * 64 + (lane >> 3) * 32;
                        for (int q = 0; q < 20; q++) {
                            const int p = (lane & 7) * 20 + q;
                            v |= (int32_t)((h[p >> 3] >> (p & 7)) & 1u) << q;
                        }
                        if (v & 0x80000) v -= 0x100000;
                        for (int i = 0; i < 27; i++) v += (int32_t)wt[j * 32 + i] * rows[i / 9][(i % 9) * 16 + lane];
                        const int32_t sat = v > 127 ? 127 : v < -128 ? -128 : v;
                        a[lane >> 2] |= (uint32_t)(uint8_t)(int8_t)sat << ((lane & 3) * 8);
                    }
                    p0 += STEM3P_BLOCK; p1 += STEM3P_BLOCK; p2 += STEM3P_BLOCK;
                    o[b] = (uint16_t)stem3p_pack16(a[0], a[1], a[2], a[3]);
                }
#endif
            }
        }
    }
    return 0;
}

#endif // CONFIG_IDF_TARGET_ESP32S3
