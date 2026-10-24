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

/* Counting with the multiplier, "pie_swar" (benchmarks/experiments/pie_swar_bench.py: 1.24-1.39x faster than one
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

void symbex_and_popcount_rows(const uint32_t* a, const uint32_t* w, int stride, int n_rows, int32_t* out) {
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

#endif // CONFIG_IDF_TARGET_ESP32S3
