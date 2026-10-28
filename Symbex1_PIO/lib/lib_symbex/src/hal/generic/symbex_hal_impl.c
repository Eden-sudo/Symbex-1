/*
 * symbex_hal_impl.c (generic) -- chooses AT COMPILE TIME how to count bits:
 *   NATIVE : x86 with POPCNT, AArch64 (64-bit words), RISC-V with Zbb (cpop; the Hazard3 core of the RP2350)
 *   LUT16  : ESP32 (ESP_PLATFORM) and the Cortex-M33 core of the RP2350 (Raspberry Pi Pico 2): 64 KB table in RAM;
 *            SWAR if there is no RAM for it. Measured with tools/dev/count_bench.py on the Pico 2 (M33, 150 MHz):
 *            lut16 1.8x faster than SWAR and 2.8x faster than the compiler's popcount; SWAR with the DSP usada8
 *            (bytes summed in one instruction) only 1.1x faster than SWAR. On Hazard3, cpop is 1.25x faster than
 *            SWAR and 1.8x faster than lut16.
 *   LUT8   : AVR (Arduino Uno): 256-byte table in flash (PROGMEM), 4 lookups per 32-bit word; weights read
 *            from flash too. libgcc's __builtin_popcountl took ~300 cycles per word on the ATmega328P (measured);
 *            the table takes a few tens.
 *   SWAR   : everything else
 * Flags: -DSYMBEX_POPCOUNT_FORCE_SWAR, -DSYMBEX_POPCOUNT_FORCE_LUT16, -DSYMBEX_POPCOUNT_FORCE_LUT8 (the AVR path on
 *        any chip, to check it on a PC), -DSYMBEX_POPCOUNT_NO_LUT16 (ESP32 / RP2350 Arm
 *        without the table: 64 KB of RAM back)
 *        -DSYMBEX_POPCOUNT_FORCE_NATIVE (the compiler's __builtin_popcount, for a core whose compiler turns it
 *        into an instruction the automatic detection does not know)
 *        -DSYMBEX_U8_EXTERNAL (the first-layer kernels symbex_u8_* are left out: the application gives its own,
 *        e.g. benchmarks/conv1d/har_ultracompact/pulp/symbex_u8_pulp.c with the PULP dot-product instruction)
 * On the ESP32-S3, hal/esp32s3/ is used and this file compiles to nothing. On the RP2350 (SYMBEX_HAL_RP2350), the
 * first-layer kernels (symbex_u8_*) come from hal/rp2350/ and are left out here.
 *
 * SYMBEX v3: symbex_xnor_popcount counts ONLY the first n_real_bits bits. Padding bits
 * (the rest of the last word and any extra words) never count, whatever their value.
 * Before, padding counted as a match whenever it was equal in a and b (both 0): +1 per
 * padding bit. Where the padding already differed (the convs), the result is identical.
 *
 * Weights: the SECOND argument of every counting function (b / w) is the weight row, read with
 * SYMBEX_READ_W32 (core/symbex_core.h): a normal read on every chip, pgm_read_dword on AVR, where the
 * weights live in flash (PROGMEM). The first argument (the activations) is always in RAM.
 *
 * AVR fix: int is 16 bits there, so __builtin_popcount(uint32_t) counted only the low 16 bits; the
 * 32-bit counts now use __builtin_popcountl on AVR.
 *
 * Whole layers (*_rows): one call per layer, the count of every word inlined (pc_inline). Before, the AND path
 * called symbex_hal_popcount32 for every word (~50 cycles per word on the ESP32, measured) and every neuron
 * paid the call and its checks (~90 cycles per neuron, measured): on small rows that was most of the time.
 * First layer (symbex_u8_sum_bits): SWAR, 4 inputs per 32-bit operation (ops/symbex_swar.h).
 */

#include <string.h>
#include <stdlib.h>
#include "hal/symbex_hal.h"
#include "core/symbex_core.h"
#include "ops/symbex_swar.h"

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif

#if !defined(CONFIG_IDF_TARGET_ESP32S3)

#if defined(ESP_PLATFORM)
#include "esp_heap_caps.h"
#endif

#if defined(SYMBEX_POPCOUNT_FORCE_SWAR)
  #define PATH_SWAR 1
#elif defined(SYMBEX_POPCOUNT_FORCE_LUT16)
  #define PATH_LUT16 1
#elif defined(SYMBEX_POPCOUNT_FORCE_NATIVE)
  #define PATH_NATIVE 1          /* the compiler's popcount, trusted to be an instruction */
#elif defined(SYMBEX_POPCOUNT_FORCE_LUT8) || defined(__AVR__)
  #define PATH_LUT8 1
#elif defined(__POPCNT__) || defined(__aarch64__) || defined(__riscv_zbb)
  #define PATH_NATIVE 1
#elif (defined(ESP_PLATFORM) || (defined(PICO_RP2350) && !defined(__riscv))) && !defined(SYMBEX_POPCOUNT_NO_LUT16)
  #define PATH_LUT16 1
#else
  #define PATH_SWAR 1
#endif

/* 32-bit popcount with the compiler builtin: popcountl on AVR (int = 16 bits there) */
#if defined(__AVR__)
  #define BUILTIN_POPCOUNT32(x) __builtin_popcountl(x)
#else
  #define BUILTIN_POPCOUNT32(x) __builtin_popcount(x)
#endif

/* the weight word p of row w (flash on AVR) */
#define W(w, p) SYMBEX_READ_W32(&(w)[p])

/* constants with the U suffix (not UL): on a 64-bit PC, UL would make the product 64 bits wide and the >> 24
   would keep extra bits; on AVR the compiler already widens them to 32 bits because they do not fit in 16 */
static inline int32_t popcount_swar(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    return (int32_t)((uint32_t)(x * 0x01010101U) >> 24);
}

#if defined(PATH_LUT8)
/* ---- lut8: 256-entry table, in flash on AVR ---- */
#if defined(__AVR__)
  #define LUT8_ROM const PROGMEM
  #define LUT8(v) pgm_read_byte(&table8[(v)])
#else
  #define LUT8_ROM const
  #define LUT8(v) (table8[(v)])
#endif
#define B4(n) n, n + 1, n + 1, n + 2
#define B6(n) B4(n), B4(n + 1), B4(n + 1), B4(n + 2)
#define B8(n) B6(n), B6(n + 1), B6(n + 1), B6(n + 2)
static LUT8_ROM uint8_t table8[256] = { B8(0), B8(1), B8(1), B8(2) };
static inline uint8_t popcount_lut8(uint32_t x) {
    return (uint8_t)(LUT8((uint8_t)x) + LUT8((uint8_t)(x >> 8)) + LUT8((uint8_t)(x >> 16)) + LUT8((uint8_t)(x >> 24)));
}
#endif

#if defined(PATH_LUT16)

/* ---- lut16: 65536-entry table (64 KB) in RAM, allocated on first use ---- */
static uint8_t* table16 = 0;
static int state16 = 0;                     /* 0: not tried | 1: ready | -1: no RAM */

static void prepare_table16(void) {
#if defined(ESP_PLATFORM)
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

/* the lut16 table if this path has one and it got RAM, else NULL: looked up ONCE per call, then passed down */
static inline const uint8_t* count_table(void) {
#if defined(PATH_LUT16)
    return lut16_ready() ? table16 : 0;
#else
    return 0;
#endif
}

#define ALWAYS_INLINE static inline __attribute__((always_inline))

/* popcount of one word with the path chosen at compile time, without the table (lut16 is handled apart) */
ALWAYS_INLINE int32_t pc_notable(uint32_t x) {
#if defined(PATH_LUT8)
    return popcount_lut8(x);
#elif defined(PATH_NATIVE)
    return BUILTIN_POPCOUNT32(x);
#else
    return popcount_swar(x);
#endif
}

/* popcount of one word (t: the lut16 table or NULL) -- for single words; the loops below test t ONCE */
ALWAYS_INLINE int32_t pc_inline(uint32_t x, const uint8_t* t) {
#if defined(PATH_LUT16)
    if (t) return t[x & 0xFFFFU] + t[x >> 16];
    return popcount_swar(x);
#elif defined(PATH_LUT8)
    (void)t;
    return popcount_lut8(x);
#elif defined(PATH_NATIVE)
    (void)t;
    return BUILTIN_POPCOUNT32(x);
#else
    (void)t;
    return popcount_swar(x);
#endif
}

int32_t symbex_hal_popcount32(uint32_t x) {
    return pc_inline(x, count_table());
}

/* differences (XOR bits) over the first full_words words plus `mask` bits of the next one; b = the weights.
   The counting method is decided OUTSIDE the word loop (a test inside it cost ~25 cycles per word on the ESP32). */
ALWAYS_INLINE int32_t xnor_diff(const uint32_t* a, const uint32_t* b, int full_words, uint32_t mask, const uint8_t* t) {
    int32_t diff = 0;
    int w = 0;
#if defined(PATH_LUT16)
    if (t) {
        for (; w < full_words; w++) {
            const uint32_t x = a[w] ^ W(b, w);
            diff += t[x & 0xFFFFU] + t[x >> 16];
        }
        if (mask) {
            const uint32_t x = (a[w] ^ W(b, w)) & mask;
            diff += t[x & 0xFFFFU] + t[x >> 16];
        }
        return diff;
    }
#else
    (void)t;
#endif
#if defined(PATH_NATIVE) && (UINTPTR_MAX > 0xFFFFFFFFu)
    for (; w + 1 < full_words; w += 2) {
        uint64_t x, y;
        memcpy(&x, a + w, sizeof x);
        memcpy(&y, b + w, sizeof y);
        diff += __builtin_popcountll(x ^ y);
    }
#endif
    for (; w < full_words; w++) diff += pc_notable(a[w] ^ W(b, w));
    if (mask) diff += pc_notable((a[w] ^ W(b, w)) & mask);
    return diff;
}

/* 2*matches - n_real_bits, counting ONLY the first n_real_bits bits.
 * It counts the DIFFERENCES (XOR): matches = n - differences, so the result is
 * n_real_bits - 2*differences. b = the weights. */
int32_t symbex_xnor_popcount(const uint32_t* a, const uint32_t* b, int n_words, int n_real_bits) {
    if ((long)n_real_bits > (long)n_words * 32) n_real_bits = n_words * 32;   /* never read past the buffer */
    if (n_real_bits <= 0) return 0;
    const int remainder = n_real_bits % 32;
    const uint32_t mask = remainder ? ((1UL << remainder) - 1UL) : 0;
    return n_real_bits - 2 * xnor_diff(a, b, n_real_bits / 32, mask, count_table());
}

void symbex_xnor_popcount_rows(const uint32_t* a, const uint32_t* w, int n_words, int n_real_bits,
                               int n_rows, int32_t* out) {
    if ((long)n_real_bits > (long)n_words * 32) n_real_bits = n_words * 32;
    if (n_real_bits <= 0) {
        for (int r = 0; r < n_rows; r++) out[r] = 0;
        return;
    }
    const int full_words = n_real_bits / 32;
    const int remainder = n_real_bits % 32;
    const uint32_t mask = remainder ? ((1UL << remainder) - 1UL) : 0;
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += n_words) out[r] = n_real_bits - 2 * xnor_diff(a, w, full_words, mask, t);
}

#if defined(SYMBEX_HAL_RP2350)
#define NAME_TAIL ", rp2350 " SYMBEX_RP2350_U8_NAME
#else
#define NAME_TAIL ""
#endif

const char* symbex_hal_name(void) {
#if defined(PATH_LUT16)
    return lut16_ready() ? "generic: lut16 (64 KB table in RAM)" NAME_TAIL
                         : "generic: SWAR (no RAM available for lut16)" NAME_TAIL;
#elif defined(PATH_NATIVE) && (UINTPTR_MAX > 0xFFFFFFFFu)
    return "generic: NATIVE popcount, 64-bit words" NAME_TAIL;
#elif defined(PATH_NATIVE)
    return "generic: NATIVE popcount, 32-bit words" NAME_TAIL;
#elif defined(PATH_LUT8)
    return "generic: lut8 (256-byte table in flash)" NAME_TAIL;
#else
    return "generic: SWAR fallback in software" NAME_TAIL;
#endif
}

/* ------------------------------------------------------------------------------------------------ */
/* AND arithmetic primitives (used by ops/symbex_and.c), portable C                                 */
/* ------------------------------------------------------------------------------------------------ */
ALWAYS_INLINE int32_t and_count(const uint32_t* a, const uint32_t* b, int n_words, const uint8_t* t) {
    int32_t c = 0;
#if defined(PATH_LUT16)
    if (t) {
        for (int p = 0; p < n_words; p++) {
            const uint32_t x = a[p] & W(b, p);
            c += t[x & 0xFFFFU] + t[x >> 16];
        }
        return c;
    }
#else
    (void)t;
#endif
    for (int p = 0; p < n_words; p++) c += pc_notable(a[p] & W(b, p));
    return c;
}

int32_t symbex_and_popcount(const uint32_t* a, const uint32_t* b, int n_words) {
    return and_count(a, b, n_words, count_table());
}

void symbex_and_popcount_rows(const uint32_t* a, const uint32_t* w, int stride, int n_rows, int32_t* out) {
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = and_count(a, w, stride, t);
}

/* count(a XOR row r): the differences directly */
ALWAYS_INLINE int32_t xor_count(const uint32_t* a, const uint32_t* b, int n_words, const uint8_t* t) {
    int32_t c = 0;
#if defined(PATH_LUT16)
    if (t) {
        for (int p = 0; p < n_words; p++) {
            const uint32_t x = a[p] ^ W(b, p);
            c += t[x & 0xFFFFU] + t[x >> 16];
        }
        return c;
    }
#else
    (void)t;
#endif
    for (int p = 0; p < n_words; p++) c += pc_notable(a[p] ^ W(b, p));
    return c;
}

void symbex_xor_popcount_rows(const uint32_t* a, const uint32_t* w, int stride, int n_rows, int32_t* out) {
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = xor_count(a, w, stride, t);
}

void symbex_and_popcount_rows_n(const uint32_t* a, const uint32_t* w, int n_words, int stride, int n_rows, int32_t* out) {
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = and_count(a, w, n_words, t);
}

void symbex_xor_popcount_rows_n(const uint32_t* a, const uint32_t* w, int n_words, int stride, int n_rows, int32_t* out) {
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = xor_count(a, w, n_words, t);
}

/* measured (tools/count_bench.py): with the 64 KB table the AND form is 1.15-1.25x faster than XOR on the ESP32 (AND
   results reuse fewer table entries: fewer cache misses); with a popcount instruction, SWAR or the 8-bit table, XOR
   is as fast or faster and needs no stored counts. On the RP2350's M33 (no data cache in front of its SRAM) the table
   prefers XOR too: 1.05-1.14x faster than AND */
int symbex_hal_prefers_xor(void) {
#if defined(PATH_LUT16) && defined(ESP_PLATFORM)
    return count_table() ? 0 : 1;
#else
    return 1;
#endif
}

void symbex_and_planes_rows(const uint32_t* planes, int k, int low, const uint32_t* w, int n_words,
                            int n_rows, int32_t* out) {
#if defined(__AVR__) || defined(SYMBEX_PLANES_PLANE_OUTER)
    /* 8-bit chips: one plane at a time, its count in a 16-bit register, shifted ONCE per plane. The word-outer loop
       below keeps 8 int32 counts in memory: on AVR each update is 4 loads + 4 stores, and it made the first layer
       ~35% slower than this (measured on the ATmega328P). Re-reading each weight word k times from flash is cheap. */
    for (int r = 0; r < n_rows; r++, w += n_words) {
        int32_t total = 0;
        for (int q = 0; q < k; q++) {
            const uint32_t* pl = planes + q * n_words;
            uint16_t c = 0;
            for (int p = 0; p < n_words; p++) c += (uint16_t)pc_notable(pl[p] & W(w, p));
            total += (int32_t)c << (low + q);
        }
        out[r] = total;
    }
    return;
#endif
    const uint8_t* t = count_table();
    for (int r = 0; r < n_rows; r++, w += n_words) {
        int32_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int p = 0; p < n_words; p++) {
            const uint32_t wv = W(w, p);                     /* each weight word read once (it may be in flash) */
#if defined(PATH_LUT16)
            if (t) {
                for (int q = 0; q < k; q++) {
                    const uint32_t x = planes[q * n_words + p] & wv;
                    s[q] += t[x & 0xFFFFU] + t[x >> 16];
                }
                continue;
            }
#endif
            for (int q = 0; q < k; q++) s[q] += pc_notable(planes[q * n_words + p] & wv);
        }
        int32_t total = 0;
        for (int q = 0; q < k; q++) total += s[q] << (low + q);
        out[r] = total;
    }
#if !defined(PATH_LUT16)
    (void)t;
#endif
}

#if !defined(SYMBEX_HAL_RP2350) && !defined(SYMBEX_U8_EXTERNAL)   /* the RP2350 has its own: hal/rp2350/ */
void symbex_u8_prepare(const uint8_t* x, int n_inputs, uint8_t* workspace, int n_words) {
    const int bytes = n_words * 32;                      /* natural order, zero padding */
    for (int i = 0; i < bytes; i++) workspace[i] = (i < n_inputs) ? x[i] : 0;
}

/* natural layout: the SWAR method of ops/symbex_swar.h over the whole padded row (padding inputs are 0) */
int32_t symbex_u8_sum_bits(const uint8_t* workspace, const uint32_t* w, int n_words) {
    return symbex_swar_u8_masked_sum(workspace, w, (uint16_t)(n_words * 32));
}

void symbex_u8_sum_bits_rows(const uint8_t* workspace, int n_inputs, const uint32_t* w, int stride, int n_rows,
                             int32_t* out) {
    /* natural layout: only the real inputs (the padding is 0 and adds nothing; visiting it cost up to 3x on short rows) */
    if (n_inputs > stride * 32) n_inputs = stride * 32;
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = symbex_swar_u8_masked_sum(workspace, w, (uint16_t)n_inputs);
}
#endif // !SYMBEX_HAL_RP2350 && !SYMBEX_U8_EXTERNAL

#endif // !CONFIG_IDF_TARGET_ESP32S3
