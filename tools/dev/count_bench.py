"""
count_bench.py -- which way of COUNTING BITS is fastest on this machine: the raw methods, not the library's kernels.

    python count_bench.py                                  # x86 (this PC): every method the CPU supports
    python count_bench.py --reps 200                       # more repetitions (steadier numbers)
    python count_bench.py --board esp32 --port /dev/ttyUSB0
    python count_bench.py --board s3 --port /dev/ttyUSB0
    python count_bench.py --board pico2_arm --port /dev/ttyACM1      # Raspberry Pi Pico 2: Cortex-M33 core
    python count_bench.py --board pico2_riscv --port /dev/ttyACM1    # Raspberry Pi Pico 2: Hazard3 RISC-V core

A binary layer is "logical operation, then count the 1 bits": the count is what costs, and every chip has a different
best way to do it. This measures them all on the same pattern as a convolution (P positions x O output channels, each
one a window of W 32-bit words against a weight row), with the window sizes of the binary ResNet-8 and wider ones:
    builtin     __builtin_popcount on 32-bit words (POPCNT when the CPU has it: -march=native)
    builtin64   the same on 64-bit words (two words per instruction)
    swar        bit tricks in software, no table (12 operations per word; a barrier keeps the compiler from turning
                it into the popcount instruction, as GCC does on x86)
    lut8        a 256-entry table, 4 lookups per word
    lut16       a 65,536-entry table (64 KB), 2 lookups per word
    hs          Harley-Seal: carry-save adders over 16 words, then few counts (32-bit words)
    avx2        x86: vectorial, 256 bits at a time, nibble table inside the registers (pshufb, Mula's method)
    pie         ESP32-S3: 128 bits per logical op, counted with the SIMD multiplier (ee.vmulas.u8.accx: bit k of each
                of the 16 bytes times 1, summed), 8 steps per 128 bits -- the S3 has no popcount instruction
    dsp         Arm cores with the DSP extension (Cortex-M33 of the Pico 2): SWAR down to byte counts, then usada8 adds
                the 4 bytes into the running count in one instruction (replaces the multiply, shift and add)
each in two forms:
    xnor        differences = count(x XOR w)
    and         differences = count(x) + count(w) - 2 * count(x AND w)   (count(x) once per position; count(w) stored)
And, on the same pattern, the INTEGER arithmetic a quantized network uses instead (one multiply-accumulate per element),
to compare against: how many binary operations per cycle against how many 8-bit multiply-accumulates per cycle:
    int8 / int16 / int32 / float32   dot products in C (whatever the compiler makes of them on this chip)
    pie_int8                         ESP32-S3: 16 int8 multiply-accumulates per instruction (ee.vmulas.s8.accx, as ESP-NN)
    avx2_int8 / vnni_int8            x86: 8-bit SIMD (maddubs + madd / AVX-VNNI dpbusd)
    dsp_int8                         Arm DSP (Cortex-M33): sxtb16 + smlad, two 8-bit multiply-accumulates per
                                     instruction, as CMSIS-NN does
Every method must give the same raw values (checksums), otherwise its time is not trusted. On the PC the unit is the
nanosecond; on the boards, CPU cycles (ESP32s: esp_cpu_get_cycle_count, 240 MHz; Pico 2: the core's own counter, DWT
CYCCNT on the M33 and mcycle on Hazard3, 150 MHz, checked against micros() at start). Results: one block per method,
and tools/chip_profiles/count_<x86|esp32|s3|pico2_arm|pico2_riscv>.json (the fastest per window size, for the HAL).
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTHS = [5, 9, 18, 36, 72, 144]                      # words per window (5..36: the binary ResNet-8)
POSITIONS, OUTPUTS = 64, 64

C_SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(ESP_PLATFORM)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "sdkconfig.h"
#elif defined(ARDUINO_ARCH_RP2040)
#define ON_PICO 1                         /* Arduino-Pico core (RP2040 / RP2350): output through main.cpp's Serial */
int bench_printf(const char* fmt, ...);
#define printf bench_printf
#define fflush(f) ((void)0)
#else
#include <time.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#if defined(__ARM_FEATURE_DSP)
#include <arm_acle.h>
#endif

#define P POSITIONS
#define O OUTPUTS
static uint32_t *X, *Wt;                 /* windows and weight rows, each `st` words (padded to 8 with zeros) */
static int32_t PCW[O];
static uint8_t LUT8[256], *LUT16;
static int nw, st;

static inline int pc_builtin(uint32_t v) { return __builtin_popcount(v); }
static inline int pc_swar(uint32_t x) {
    __asm__ volatile("" : "+r"(x));          /* keeps the compiler from replacing this by its popcount idiom */
    x = x - ((x >> 1) & 0x55555555u); x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu; return (int)((x * 0x01010101u) >> 24);
}
static inline int pc_lut8(uint32_t v) { return LUT8[v & 255] + LUT8[(v >> 8) & 255] + LUT8[(v >> 16) & 255] + LUT8[v >> 24]; }
static inline int pc_lut16(uint32_t v) { return LUT16[v & 0xFFFF] + LUT16[v >> 16]; }

/* one count over n words of (a OP b), OP = 0 xor / 1 and; per method */
#define DEF_WORDS(NAME, PC) \
static inline int NAME(const uint32_t* a, const uint32_t* b, int n, int op) { \
    int c = 0; \
    if (op) { for (int i = 0; i < n; i++) c += PC(a[i] & b[i]); } else { for (int i = 0; i < n; i++) c += PC(a[i] ^ b[i]); } \
    return c; }
DEF_WORDS(cnt_builtin, pc_builtin)
DEF_WORDS(cnt_swar, pc_swar)
DEF_WORDS(cnt_lut8, pc_lut8)
DEF_WORDS(cnt_lut16, pc_lut16)

static inline int cnt_builtin64(const uint32_t* a, const uint32_t* b, int n, int op) {
    int c = 0, i = 0;
    for (; i + 1 < n; i += 2) {
        uint64_t x, y; memcpy(&x, a + i, 8); memcpy(&y, b + i, 8);
        c += __builtin_popcountll(op ? (x & y) : (x ^ y));
    }
    for (; i < n; i++) c += __builtin_popcount(op ? (a[i] & b[i]) : (a[i] ^ b[i]));
    return c;
}

/* Harley-Seal: carry-save adders over 16 words, popcount of the sums at the end */
#define CSA(h, l, a, b, c) { uint32_t u = a ^ b; h = (a & b) | (u & c); l = u ^ c; }
static inline int cnt_hs(const uint32_t* a, const uint32_t* b, int n, int op) {
    uint32_t ones = 0, twos = 0, fours = 0, eights = 0, sixteens, twosA, twosB, foursA, foursB, eightsA, eightsB;
    int total = 0, i = 0;
#define V(k) (op ? (a[i + k] & b[i + k]) : (a[i + k] ^ b[i + k]))
    for (; i + 16 <= n; i += 16) {
        CSA(twosA, ones, ones, V(0), V(1)); CSA(twosB, ones, ones, V(2), V(3)); CSA(foursA, twos, twos, twosA, twosB);
        CSA(twosA, ones, ones, V(4), V(5)); CSA(twosB, ones, ones, V(6), V(7)); CSA(foursB, twos, twos, twosA, twosB);
        CSA(eightsA, fours, fours, foursA, foursB);
        CSA(twosA, ones, ones, V(8), V(9)); CSA(twosB, ones, ones, V(10), V(11)); CSA(foursA, twos, twos, twosA, twosB);
        CSA(twosA, ones, ones, V(12), V(13)); CSA(twosB, ones, ones, V(14), V(15)); CSA(foursB, twos, twos, twosA, twosB);
        CSA(eightsB, fours, fours, foursA, foursB);
        CSA(sixteens, eights, eights, eightsA, eightsB);
        total += __builtin_popcount(sixteens);
    }
    total = 16 * total + 8 * __builtin_popcount(eights) + 4 * __builtin_popcount(fours) + 2 * __builtin_popcount(twos)
            + __builtin_popcount(ones);
    for (; i < n; i++) total += __builtin_popcount(V(0));
#undef V
    return total;
}

#if defined(__ARM_FEATURE_DSP)
/* SWAR down to 4 byte counts (0..8 each), then usada8: |x - 0| summed over the 4 bytes, added to the count */
static inline int cnt_dsp(const uint32_t* a, const uint32_t* b, int n, int op) {
    uint32_t c = 0;
#define DSP_WORD(EXPR) { uint32_t x = (EXPR); __asm__ volatile("" : "+r"(x)); \
        x = x - ((x >> 1) & 0x55555555u); x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u); \
        x = (x + (x >> 4)) & 0x0F0F0F0Fu; c = __usada8(x, 0, c); }
    if (op) { for (int i = 0; i < n; i++) DSP_WORD(a[i] & b[i]) } else { for (int i = 0; i < n; i++) DSP_WORD(a[i] ^ b[i]) }
#undef DSP_WORD
    return (int)c;
}
#endif

#ifdef __AVX2__
/* Mula: popcount of 256 bits with a 16-entry nibble table held in a register (pshufb), summed with sad_epu8 */
static inline int cnt_avx2(const uint32_t* a, const uint32_t* b, int st_words, int op) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low = _mm256_set1_epi8(0x0F);
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < st_words; i += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(a + i)), y = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i v = op ? _mm256_and_si256(x, y) : _mm256_xor_si256(x, y);
        __m256i c = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v, low)),
                                    _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), low)));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(c, _mm256_setzero_si256()));
    }
    return (int)(_mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 1) + _mm256_extract_epi64(acc, 2)
                 + _mm256_extract_epi64(acc, 3));
}
#endif

#if defined(CONFIG_IDF_TARGET_ESP32S3)
/* the S3's SIMD multiplier as a bit counter: (a OP b) per 128 bits, then bit k of every byte times 1, summed in ACCX.
   q6 = q7 = 16 ones (loaded once per layer by pie_prepare); st must be a multiple of 4 and the rows 16-byte aligned */
static const uint8_t PIE_ONES[16] __attribute__((aligned(16))) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
#define PIE_STEP(k) "ssai " #k " \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q6 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
static inline void pie_prepare(void) {
    const uint8_t* a = PIE_ONES; const uint8_t* b = PIE_ONES;
    __asm__ __volatile__("ee.vld.128.ip q6, %0, 0 \n\t ee.vld.128.ip q7, %1, 0 \n\t" : "+r"(a), "+r"(b) : : "memory");
}
static inline int cnt_pie(const uint32_t* a, const uint32_t* b, int st_words, int op) {
    __asm__ __volatile__("ee.zero.accx");
    if (op) {
        for (int g = 0; g < st_words / 4; g++)
            __asm__ __volatile__("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.andq q2, q0, q1 \n\t"
                                 "ee.andq q3, q2, q6 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
                                 PIE_STEP(1) PIE_STEP(2) PIE_STEP(3) PIE_STEP(4) PIE_STEP(5) PIE_STEP(6) PIE_STEP(7)
                                 : "+r"(a), "+r"(b) : : "memory");
    } else {
        for (int g = 0; g < st_words / 4; g++)
            __asm__ __volatile__("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.xorq q2, q0, q1 \n\t"
                                 "ee.andq q3, q2, q6 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
                                 PIE_STEP(1) PIE_STEP(2) PIE_STEP(3) PIE_STEP(4) PIE_STEP(5) PIE_STEP(6) PIE_STEP(7)
                                 : "+r"(a), "+r"(b) : : "memory");
    }
    uint32_t v;
    __asm__ __volatile__("rur.accx_0 %0" : "=r"(v));
    return (int)v;
}
#endif

#if defined(ESP_PLATFORM)
/* CPU cycles. The counter is 32 bits and wraps every ~17.9 s at 240 MHz: the elapsed time is an UNSIGNED 32-bit
   difference, right across a wrap (each timed run is far shorter than 17.9 s) */
typedef uint32_t stamp_t;
static inline stamp_t now_t(void) { return (stamp_t)esp_cpu_get_cycle_count(); }
static inline double elapsed(stamp_t t0) { return (double)(uint32_t)(now_t() - t0); }
#define UNIT "cycles"
#elif defined(ON_PICO)
/* the core's own 32-bit cycle counter (Arduino-Pico's getCycleCount is a 24-bit SysTick on Arm and a PIO counter on
   RISC-V): DWT CYCCNT on the Cortex-M33, mcycle on Hazard3 (enabled by pico_counter_start). Unsigned differences, as
   on the ESP32s (a wrap every ~28.6 s at 150 MHz) */
typedef uint32_t stamp_t;
#if defined(__riscv)
static inline stamp_t now_t(void) { uint32_t c; __asm__ volatile("csrr %0, mcycle" : "=r"(c)); return c; }
void pico_counter_start(void) { __asm__ volatile("csrw mcountinhibit, zero"); }
#else
#define DEMCR_REG (*(volatile uint32_t*)0xE000EDFCu)
#define DWT_CTRL_REG (*(volatile uint32_t*)0xE0001000u)
#define DWT_CYCCNT_REG (*(volatile uint32_t*)0xE0001004u)
static inline stamp_t now_t(void) { return DWT_CYCCNT_REG; }
void pico_counter_start(void) { DEMCR_REG |= 1u << 24; DWT_CYCCNT_REG = 0; DWT_CTRL_REG |= 1u; }
#endif
uint32_t pico_cycles(void) { return now_t(); }
static inline double elapsed(stamp_t t0) { return (double)(uint32_t)(now_t() - t0); }
#define UNIT "cycles"
#else
typedef double stamp_t;
static inline stamp_t now_t(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static inline double elapsed(stamp_t t0) { return now_t() - t0; }
#define UNIT "ns"
#endif
static volatile uint32_t sink;

/* the layer: raw = bits - 2 * differences for every (position, output); returns a checksum */
#define LAYER(FN, LEN) LAYER_P(FN, LEN, )
#define LAYER_P(FN, LEN, PREP) \
static uint32_t layer_##FN(int op) { \
    uint32_t ck = 0; const int32_t nbits = nw * 32; PREP \
    for (int p = 0; p < P; p++) { \
        const uint32_t* x = &X[p * st]; int32_t pcx = 0; \
        if (op) for (int i = 0; i < nw; i++) pcx += pc_builtin(x[i]); \
        for (int o = 0; o < O; o++) { \
            int c = FN(x, &Wt[o * st], LEN, op); \
            int32_t d = op ? pcx + PCW[o] - 2 * c : c; \
            ck = ck * 31u + (uint32_t)(nbits - 2 * d); } } \
    return ck; }
LAYER(cnt_builtin, nw)
LAYER(cnt_builtin64, nw)
LAYER(cnt_swar, nw)
LAYER(cnt_lut8, nw)
LAYER(cnt_lut16, nw)
LAYER(cnt_hs, nw)
#if defined(__ARM_FEATURE_DSP)
LAYER(cnt_dsp, nw)
#endif
#ifdef __AVX2__
LAYER(cnt_avx2, st)
#endif
#if defined(CONFIG_IDF_TARGET_ESP32S3)
LAYER_P(cnt_pie, st, pie_prepare();)
#endif

typedef uint32_t (*layer_t)(int);
static void measure(const char* name, layer_t f, int reps) {
    for (int op = 0; op < 2; op++) {
        uint32_t ck = f(op);                                       /* warm-up */
        double best = 1e30;
        for (int r = 0; r < 7; r++) {                              /* the best of 7 runs of `reps` layers */
            stamp_t t0 = now_t();
            for (int k = 0; k < reps; k++) sink = f(op);
            double t = elapsed(t0) / reps;
            if (t < best) best = t;
        }
        printf("R:%s|%s|%d|%.3f|%u\n", name, op ? "and" : "xnor", nw, best / (P * O), (unsigned)ck);
        fflush(stdout);
    }
}

/* ================= integer multiply-accumulate: the same layer pattern, one MAC per element ================= */
#define IP 8                                         /* positions and outputs of the integer layers: small enough for the */
#define IO 8                                         /* ESP32's split RAM (the cost per MAC does not depend on how many) */
static const int INT_LENS[] = {160, 288, 576, 1152}; /* elements per window: 5, 9, 18, 36 words of bits */
#define DEF_DOT(NAME, T, ACC) \
static ACC NAME(const T* a, const T* b, int n) { ACC s = 0; for (int i = 0; i < n; i++) s += (ACC)a[i] * (ACC)b[i]; return s; }
DEF_DOT(dot_int8, int8_t, int32_t)
DEF_DOT(dot_int16, int16_t, int32_t)
DEF_DOT(dot_int32, int32_t, int32_t)
DEF_DOT(dot_float32, float, float)

#if defined(CONFIG_IDF_TARGET_ESP32S3)
static int32_t dot_pie_int8(const int8_t* a, const int8_t* b, int n) {    /* n multiple of 16, rows 16-byte aligned */
    __asm__ __volatile__("ee.zero.accx");
    for (int g = 0; g < n / 16; g++)
        __asm__ __volatile__("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.vmulas.s8.accx q0, q1 \n\t"
                             : "+r"(a), "+r"(b) : : "memory");
    uint32_t v;
    __asm__ __volatile__("rur.accx_0 %0" : "=r"(v));
    return (int32_t)v;
}
#endif
#if defined(__ARM_FEATURE_DSP)
/* CMSIS-NN's inner step: sxtb16 widens bytes 0 and 2 (and, rotated, 1 and 3) to two int16, smlad multiplies both pairs
   and adds them to the sum. n multiple of 4 (memcpy: the compiler makes it a single word load) */
static int32_t dot_dsp_int8(const int8_t* a, const int8_t* b, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; i += 4) {
        uint32_t x, y;
        memcpy(&x, a + i, 4); memcpy(&y, b + i, 4);
        s = __smlad(__sxtb16(x), __sxtb16(y), s);
        s = __smlad(__sxtb16(__ror(x, 8)), __sxtb16(__ror(y, 8)), s);
    }
    return s;
}
#endif
#ifdef __AVX2__
static int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
/* activations as unsigned bytes (0..63 here), weights signed: maddubs -> int16 pairs (no saturation at these values) */
static int32_t dot_avx2_int8(const int8_t* a, const int8_t* b, int n) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        __m256i p = _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(a + i)), _mm256_loadu_si256((const __m256i*)(b + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    return hsum_epi32(acc);
}
#endif
#ifdef __AVXVNNI__
static int32_t dot_vnni_int8(const int8_t* a, const int8_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32)
        acc = _mm256_dpbusd_avx_epi32(acc, _mm256_loadu_si256((const __m256i*)(a + i)), _mm256_loadu_si256((const __m256i*)(b + i)));
    return hsum_epi32(acc);
}
#endif

#define INT_LAYER(NAME, T, DOT) \
static uint32_t ilayer_##NAME(const T* X, const T* W, int n) { \
    uint32_t ck = 0; \
    for (int p = 0; p < IP; p++) for (int o = 0; o < IO; o++) ck = ck * 31u + (uint32_t)(int32_t)DOT(&X[p * n], &W[o * n], n); \
    return ck; }
INT_LAYER(int8, int8_t, dot_int8)
INT_LAYER(int16, int16_t, dot_int16)
INT_LAYER(int32, int32_t, dot_int32)
INT_LAYER(float32, float, dot_float32)
#if defined(CONFIG_IDF_TARGET_ESP32S3)
INT_LAYER(pie_int8, int8_t, dot_pie_int8)
#endif
#if defined(__ARM_FEATURE_DSP)
INT_LAYER(dsp_int8, int8_t, dot_dsp_int8)
#endif
#ifdef __AVX2__
INT_LAYER(avx2_int8, int8_t, dot_avx2_int8)
#endif
#ifdef __AVXVNNI__
INT_LAYER(vnni_int8, int8_t, dot_vnni_int8)
#endif

#if defined(ESP_PLATFORM)
static void* alloc32(size_t n) { return heap_caps_aligned_alloc(32, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); }
static void release(void* p) { heap_caps_free(p); }
#else
static void* alloc32(size_t n) { return aligned_alloc(32, (n + 31) / 32 * 32); }
static void release(void* p) { free(p); }
#endif

/* the integer layers: same data for every method (activations 0..63, weights -64..63: exact in every type) */
#define MEASURE_INT(NAME, T, XS, WS) { \
    uint32_t ck = ilayer_##NAME(XS, WS, n); double best = 1e30; \
    for (int r = 0; r < 7; r++) { stamp_t t0 = now_t(); for (int k = 0; k < scaled; k++) sink = ilayer_##NAME(XS, WS, n); \
        double t = elapsed(t0) / scaled; if (t < best) best = t; } \
    printf("I:%s|%d|%.4f|%u\n", #NAME, n, best / ((double)IP * IO * n), (unsigned)ck); fflush(stdout); }

static int run_int(int reps) {
    uint32_t r = 777;
    for (unsigned li = 0; li < sizeof INT_LENS / sizeof INT_LENS[0]; li++) {
        const int n = INT_LENS[li];
        const int scaled = reps * 4 * 1152 / n / 16 + 1;
        printf("WI:%d\n", n);
        fflush(stdout);
        int8_t* xa = (int8_t*)alloc32((size_t)IP * n); int8_t* wa = (int8_t*)alloc32((size_t)IO * n);
        if (!xa || !wa) { printf("ERROR: no RAM for the integer layers\n"); return 1; }
        for (int i = 0; i < IP * n; i++) xa[i] = (int8_t)((r = r * 1103515245u + 12345u) >> 26);            /* 0..63 */
        for (int i = 0; i < IO * n; i++) wa[i] = (int8_t)((int)((r = r * 1103515245u + 12345u) >> 25) - 64);  /* -64..63 */
        MEASURE_INT(int8, int8_t, xa, wa)
#if defined(CONFIG_IDF_TARGET_ESP32S3)
        MEASURE_INT(pie_int8, int8_t, xa, wa)
#endif
#if defined(__ARM_FEATURE_DSP)
        MEASURE_INT(dsp_int8, int8_t, xa, wa)
#endif
#ifdef __AVX2__
        MEASURE_INT(avx2_int8, int8_t, xa, wa)
#endif
#ifdef __AVXVNNI__
        MEASURE_INT(vnni_int8, int8_t, xa, wa)
#endif
        {   int16_t* x = (int16_t*)alloc32((size_t)IP * n * 2); int16_t* w = (int16_t*)alloc32((size_t)IO * n * 2);
            if (!x || !w) { printf("ERROR: no RAM for the integer layers\n"); return 1; }
            for (int i = 0; i < IP * n; i++) x[i] = xa[i];
            for (int i = 0; i < IO * n; i++) w[i] = wa[i];
            MEASURE_INT(int16, int16_t, x, w)
            release(x); release(w); }
        {   int32_t* x = (int32_t*)alloc32((size_t)IP * n * 4); int32_t* w = (int32_t*)alloc32((size_t)IO * n * 4);
            if (!x || !w) { printf("ERROR: no RAM for the integer layers\n"); return 1; }
            for (int i = 0; i < IP * n; i++) x[i] = xa[i];
            for (int i = 0; i < IO * n; i++) w[i] = wa[i];
            MEASURE_INT(int32, int32_t, x, w)
            release(x); release(w); }
        {   float* x = (float*)alloc32((size_t)IP * n * 4); float* w = (float*)alloc32((size_t)IO * n * 4);
            if (!x || !w) { printf("ERROR: no RAM for the integer layers\n"); return 1; }
            for (int i = 0; i < IP * n; i++) x[i] = xa[i];
            for (int i = 0; i < IO * n; i++) w[i] = wa[i];
            MEASURE_INT(float32, float, x, w)
            release(x); release(w); }
        release(xa); release(wa);
    }
    return 0;
}

static int run(int reps) {
    LUT16 = (uint8_t*)alloc32(65536);
    if (!LUT16) { printf("ERROR: no RAM for the 64 KB table\n"); return 1; }
    for (int i = 0; i < 256; i++) LUT8[i] = (uint8_t)__builtin_popcount(i);
    for (int i = 0; i < 65536; i++) LUT16[i] = (uint8_t)__builtin_popcount(i);
    static const int widths[] = {WIDTHS};
    uint32_t r = 12345;
    printf("UNIT:%s\n", UNIT);
    printf("FEATURES:%s%s%s\n",
#if defined(__POPCNT__)
           "popcnt ",
#elif defined(ESP_PLATFORM)
           "no popcount instruction (builtin = libgcc software) ",
#elif defined(__riscv_zbb)
           "RISC-V Zbb (cpop: popcount instruction) ",
#elif defined(__ARM_FEATURE_DSP)
           "Arm DSP extension (usada8, smlad), no popcount instruction ",
#else
           "no-popcnt ",
#endif
#ifdef __AVX2__
           "avx2",
#else
           "",
#endif
#if defined(CONFIG_IDF_TARGET_ESP32S3)
           "PIE SIMD (multiply-accumulate, no popcount)"
#else
           ""
#endif
    );
    for (unsigned wi = 0; wi < sizeof widths / sizeof widths[0]; wi++) {
        nw = widths[wi]; st = (nw + 7) / 8 * 8;
        X = (uint32_t*)alloc32((size_t)P * st * 4); Wt = (uint32_t*)alloc32((size_t)O * st * 4);
        if (!X || !Wt) { printf("ERROR: no RAM for %d-word windows\n", nw); return 1; }
        memset(X, 0, (size_t)P * st * 4); memset(Wt, 0, (size_t)O * st * 4);
        for (int i = 0; i < P; i++) for (int j = 0; j < nw; j++) X[i * st + j] = (r = r * 1103515245u + 12345u);
        for (int o = 0; o < O; o++) { PCW[o] = 0; for (int j = 0; j < nw; j++) { Wt[o * st + j] = (r = r * 1103515245u + 12345u); PCW[o] += pc_builtin(Wt[o * st + j]); } }
        int scaled = reps * 36 / nw + 1;                           /* about the same time per width */
        printf("W:%d\n", nw);
        fflush(stdout);
        measure("builtin", layer_cnt_builtin, scaled);
        measure("builtin64", layer_cnt_builtin64, scaled);
        measure("swar", layer_cnt_swar, scaled);
        measure("lut8", layer_cnt_lut8, scaled);
        measure("lut16", layer_cnt_lut16, scaled);
        measure("hs", layer_cnt_hs, scaled);
#if defined(__ARM_FEATURE_DSP)
        measure("dsp", layer_cnt_dsp, scaled);
#endif
#ifdef __AVX2__
        measure("avx2", layer_cnt_avx2, scaled);
#endif
#if defined(CONFIG_IDF_TARGET_ESP32S3)
        measure("pie", layer_cnt_pie, scaled);
#endif
        release(X); release(Wt);
    }
    release(LUT16);                                                /* the integer part does not need the 64 KB table */
    if (run_int(reps)) return 1;
    printf("DONE\n");
    fflush(stdout);
    return 0;
}

#if defined(ESP_PLATFORM)
void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(500));
    printf("\nCOUNT BENCH ready\n");
    fflush(stdout);
    run(REPS_BOARD);
    while (1) vTaskDelay(portMAX_DELAY);
}
#elif defined(ON_PICO)
int bench_run(int reps) { return run(reps); }
#else
int main(int argc, char** argv) { return run(argc > 1 ? atoi(argv[1]) : 100); }
#endif
"""


BOARDS = {"esp32": ("esp32dev", "ESP32 (Xtensa LX6, 240 MHz)"), "s3": ("esp32-s3-devkitc-1", "ESP32-S3 (Xtensa LX7 + PIE, 240 MHz)"),
          "pico2_arm": ("rpipico2", "Raspberry Pi Pico 2, RP2350 Arm Cortex-M33 (DSP, 150 MHz)"),
          "pico2_riscv": ("rpipico2", "Raspberry Pi Pico 2, RP2350 Hazard3 RISC-V (Zbb, 150 MHz)")}

# Pico 2: an Arduino-Pico sketch around the same C code. It waits for a 'g' from the PC (USB serial: no reset line),
# checks the cycle counter against micros(), then runs.
PICO_MAIN = r"""
#include <Arduino.h>
#include <stdarg.h>
extern "C" {
int bench_run(int reps);
void pico_counter_start(void);
uint32_t pico_cycles(void);
int bench_printf(const char* fmt, ...) {
    char buf[256];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    Serial.write(buf, n < (int)sizeof buf ? n : (int)sizeof buf - 1);
    return n;
}
}
void setup() {
    Serial.begin(115200);
    pico_counter_start();
}
void loop() {
    if (Serial.available() && Serial.read() == 'g') {
        Serial.printf("\nCOUNT BENCH ready\n");
        uint32_t c0 = pico_cycles(), t0 = micros();
        while (micros() - t0 < 200000) {}
        uint32_t c1 = pico_cycles(), t1 = micros();
        Serial.printf("CLOCK:%lu|%lu\n", (unsigned long)((c1 - c0) / (t1 - t0)), (unsigned long)(rp2040.f_cpu() / 1000000));
        bench_run(REPS_BOARD);
    }
}
"""


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def c_code(reps_board=1):
    return (C_SOURCE.replace("POSITIONS", str(POSITIONS)).replace("OUTPUTS", str(OUTPUTS))
            .replace("{WIDTHS}", "{" + ", ".join(map(str, WIDTHS)) + "}").replace("REPS_BOARD", str(reps_board)))


def run_pc(args):
    if shutil.which("gcc") is None:
        sys.exit("[X] gcc not found")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "count_bench.c")
        open(src, "w").write(c_code())
        exe = os.path.join(d, "count_bench")
        r = subprocess.run(["gcc", "-O2", "-march=native", "-o", exe, src], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("[X] did not compile:\n" + r.stderr[-2000:])
        cmd = ([shutil.which("taskset"), "-c", str(args.core)] if shutil.which("taskset") else []) + [exe, str(args.reps)]
        print(f"[*] measuring on {cpu_name()} (one core{', pinned to core ' + str(args.core) if shutil.which('taskset') else ''}) ...")
        return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines(), cpu_name()


def open_pico(port, timeout=30):
    """the Pico 2 re-enumerates its USB serial after flashing: wait for the port, then send the start byte"""
    import time
    import serial
    t0 = time.time()
    while True:
        try:
            s = serial.Serial(port, 115200, timeout=1)
            break
        except (serial.SerialException, OSError):
            if time.time() - t0 > timeout:
                sys.exit(f"[X] {port} did not come back after flashing")
            time.sleep(0.5)
    time.sleep(1.0)
    s.reset_input_buffer()
    s.write(b"g")
    return s


def run_board(args):
    """an ESP-IDF project (ESP32s) or an Arduino-Pico sketch (Pico 2) with the same C code, flashed to the board; the
    results come back over serial (115200)"""
    import time
    import serial
    board, title = BOARDS[args.board]
    pico = args.board.startswith("pico2")
    proj = os.path.join(HERE, "runs", f"count_bench_{args.board}")
    os.makedirs(os.path.join(proj, "src"), exist_ok=True)
    w = lambda name, text: open(os.path.join(proj, name), "w").write(text)
    if pico:
        w("platformio.ini", f"[env:{args.board}]\nplatform = raspberrypi\nboard = {board}\nframework = arduino\n"
                            f"board_build.core = earlephilhower\nupload_protocol = picotool\nupload_port = {args.port}\n"
                            + ("board_build.mcu = rp2350-riscv\n" if args.board == "pico2_riscv" else "")
                            + "build_flags = -O2 -Wl,-u,_printf_float\nbuild_unflags = -Os\n")   # floats in printf (newlib-nano)
        w("src/bench.c", c_code(args.board_reps))
        w("src/main.cpp", PICO_MAIN.replace("REPS_BOARD", str(args.board_reps)))
    else:
        w("platformio.ini", f"[env:{args.board}]\nplatform = espressif32\nboard = {board}\nframework = espidf\n"
                            f"monitor_speed = 115200\nupload_port = {args.port}\n")
        w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                                "CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\nCONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n")
        w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                            "project(count_bench)\n")
        w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" INCLUDE_DIRS ".")\n')
        w("src/main.c", c_code(args.board_reps))
    if not args.no_upload:
        print(f"[*] building and flashing {title} ...")
        if subprocess.run(["pio", "run", "-d", proj, "-t", "upload"]).returncode != 0:
            sys.exit(f"[X] build or upload failed (pio run -d {proj} -t upload)")
    if pico:
        s = open_pico(args.port)
    else:
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = args.port, 115200, 1
        s.dtr = s.rts = False
        s.open()
        s.reset_input_buffer()
        s.rts = True
        time.sleep(0.15)
        s.rts = False
    lines, ready, t0 = [], False, time.time()
    while True:
        line = s.readline().decode(errors="replace").strip()
        if "COUNT BENCH ready" in line:
            ready = True
        elif ready and line:
            if line.startswith("ERROR"):
                sys.exit(f"[X] {line}")
            if line.startswith("CLOCK:"):
                counted, nominal = map(int, line[6:].split("|"))
                print(f"      cycle counter: {counted} cycles per microsecond, clock {nominal} MHz")
                if abs(counted - nominal) > 1:
                    sys.exit("[X] the cycle counter does not run at the CPU clock: times not trusted")
            if line.startswith("W:"):
                print(f"      binary: windows of {line[2:]} words ...")
            if line.startswith("WI:"):
                print(f"      integer: windows of {line[3:]} elements ...")
            lines.append(line)
            if line == "DONE":
                return lines, title
        if not ready and time.time() - t0 > 10:
            if pico:
                sys.exit("[X] the Pico 2 did not answer the start byte")
            print("  >>> no answer to the software reset: press EN/RST once")
            t0 = time.time() + 1e9
        if ready and time.time() - t0 > 1800:
            sys.exit("[X] the board stopped answering")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=sorted(BOARDS), help="measure on this board (default: this PC)")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    ap.add_argument("--reps", type=int, default=100, help="PC: layers per timed run (default 100; more = steadier)")
    ap.add_argument("--board-reps", type=int, default=1, help="board: layers per timed run (default 1)")
    ap.add_argument("--core", type=int, default=2, help="PC: CPU core to pin the run to (taskset; default 2)")
    args = ap.parse_args()
    lines, name = run_board(args) if args.board else run_pc(args)
    feats, unit, rows, irows = "", "ns", [], []
    for line in lines:
        if line.startswith("FEATURES:"):
            feats = line[9:].strip()
        elif line.startswith("UNIT:"):
            unit = line[5:].strip()
        elif line.startswith("R:"):
            m, form, w, v, ck = line[2:].split("|")
            rows.append(dict(method=m, form=form, words=int(w), per_output=float(v), checksum=int(ck)))
        elif line.startswith("I:"):
            m, n, v, ck = line[2:].split("|")
            irows.append(dict(method=m, elements=int(n), per_mac=float(v), checksum=int(ck)))
    if not rows:
        sys.exit("[X] no results came back")
    ok = all(len({r["checksum"] for r in rows if r["words"] == w}) == 1 for w in WIDTHS)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    per = "cyc" if unit == "cycles" else "ns"
    print(f"\n    BIT COUNTING ON {name}  ({feats}; {POSITIONS} positions x {OUTPUTS} outputs per layer, best of 7 runs, "
          f"in {unit})")
    print(f"    checksums: {'every method gives the same raw values' if ok else 'DIFFER [X] (times not trusted)'}\n")
    base = {(r["form"], r["words"]): r["per_output"] for r in rows if r["method"] == "builtin"}
    for m in methods:
        print(f"    {m}")
        print(f"      {'words':>6}{'bits':>7}  {'xnor ' + per + '/out':>13}{'bit-ops/' + per:>12}{'vs builtin':>11}   "
              f"{'and ' + per + '/out':>12}{'bit-ops/' + per:>12}{'vs builtin':>11}")
        for w in WIDTHS:
            x = next((r for r in rows if r["method"] == m and r["words"] == w and r["form"] == "xnor"), None)
            a = next((r for r in rows if r["method"] == m and r["words"] == w and r["form"] == "and"), None)
            if not x or not a:
                continue
            print(f"      {w:>6}{w * 32:>7}  {x['per_output']:>13.2f}{w * 32 / x['per_output']:>12.2f}"
                  f"{base[('xnor', w)] / x['per_output']:>10.2f}x   {a['per_output']:>12.2f}"
                  f"{w * 32 / a['per_output']:>12.2f}{base[('and', w)] / a['per_output']:>10.2f}x")
        print()
    per_one = "cycle" if unit == "cycles" else "ns"
    iok = all(len({r["checksum"] for r in irows if r["elements"] == n}) == 1 for n in {r["elements"] for r in irows})
    if irows:
        print("    INTEGER MULTIPLY-ACCUMULATE (one per element; 8 positions x 8 outputs per layer)")
        print(f"    checksums: {'every method gives the same dot products' if iok else 'DIFFER [X] (times not trusted)'}\n")
        lens = sorted({r["elements"] for r in irows})
        for m in dict.fromkeys(r["method"] for r in irows):
            print(f"    {m}")
            print(f"      {'elements':>9}  {per + '/MAC':>10}{'MACs/' + per:>11}")
            for n in lens:
                r = next((r for r in irows if r["method"] == m and r["elements"] == n), None)
                if r:
                    print(f"      {n:>9}  {r['per_mac']:>10.3f}{1 / r['per_mac']:>11.2f}")
            print()
        print("    BINARY vs 8-BIT INTEGER  (the same window: bits of a binary layer, elements of an int8 layer)")
        print(f"      {'elements':>9}  {'best binary':<18}{'bit-ops/' + per_one:>14}   {'best int8':<12}{'MACs/' + per_one:>11}"
              f"{'binary / int8':>15}")
        for n in lens:
            bw = n // 32
            b = [r for r in rows if r["words"] == bw]
            i8 = [r for r in irows if r["elements"] == n and "int8" in r["method"]]
            if not b or not i8:
                continue
            rb = min(b, key=lambda r: r["per_output"])
            ri = min(i8, key=lambda r: r["per_mac"])
            bops, macs = n / rb["per_output"], 1 / ri["per_mac"]
            print(f"      {n:>9}  {rb['method'] + ' (' + rb['form'] + ')':<18}{bops:>14.2f}   {ri['method']:<12}{macs:>11.2f}"
                  f"{bops / macs:>14.2f}x")
        print("      binary / int8 = binary operations the chip does in the time of one 8-bit multiply-accumulate. With the same"
              "\n      architecture (x1) a binary layer does as many operations as its int8 version: it is faster where this is"
              "\n      above 1. Twice the channels (x2) is 4x the operations: faster only where this is above 4. (Raw arithmetic"
              "\n      only: windows, thresholds and memory traffic come on top, in both.)\n")
    print("    FASTEST PER WINDOW SIZE")
    best = {}
    for w in WIDTHS:
        cand = [r for r in rows if r["words"] == w]
        if not cand:
            continue
        r = min(cand, key=lambda r: r["per_output"])
        best[w] = dict(method=r["method"], form=r["form"], per_output=r["per_output"])
        print(f"      {w:>4} words ({w * 32:>5} bits): {r['method']} ({r['form']}), {r['per_output']:.2f} {unit} per output, "
              f"{w * 32 / r['per_output']:.2f} bit-ops per {'cycle' if unit == 'cycles' else 'ns'}")
    plat = args.board or "x86"
    path = os.path.join(HERE, "..", "chip_profiles", f"count_{plat}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(platform=plat, name=name, features=feats, unit=unit, positions=POSITIONS, outputs=OUTPUTS,
                   checksums_ok=ok, measurements=rows, fastest=best, integer_checksums_ok=iok, integer=irows),
              open(path, "w"), indent=1)
    print(f"\n    -> {path}")


if __name__ == "__main__":
    main()
