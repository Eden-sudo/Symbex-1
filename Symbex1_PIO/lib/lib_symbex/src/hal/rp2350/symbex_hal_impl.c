/*
 * symbex_hal_impl.c (rp2350) -- the first-layer kernels (uint8 inputs x 1-bit weights) of the Raspberry Pi Pico 2
 * (RP2350), for both of its core types. The bit counts (bit x bit layers) stay in hal/generic: lut16 on the
 * Cortex-M33 and the native cpop on Hazard3 were the fastest methods measured there (tools/dev/count_bench.py), so
 * only symbex_u8_prepare / symbex_u8_sum_bits / symbex_u8_sum_bits_rows live here (hal/generic leaves them out when
 * SYMBEX_HAL_RP2350 is set, see symbex_hal.h).
 *
 * Two methods, same integer as hal/generic:
 *   USADA8 : Cortex-M33 (DSP). The SWAR method of ops/symbex_swar.h, but the 4 masked bytes are summed by ONE
 *            usada8 (sum of absolute differences against 0) into a 32-bit accumulator: no 16-bit lanes to split
 *            and empty. Workspace: the inputs in natural order, zero padded.
 *   PLANES : Hazard3 (RISC-V, Zbb). symbex_u8_prepare splits the inputs into their 8 bit planes (word p of plane q,
 *            bit b = bit q of input 32p + b), the 8 plane words of the same 32 inputs side by side (workspace
 *            word 8p + q: one pointer, sequential loads); the sum is then sum_q count(plane q AND w) << q, the 8
 *            counts in registers, one cpop each. hal/generic's symbex_and_planes_rows (planes apart, counts in an
 *            array) gave only 1.15x over SWAR on Hazard3 for this layer (measured).
 * Default: USADA8 on the M33, PLANES on Hazard3 (chosen in symbex_hal.h). Flags: -DSYMBEX_RP2350_U8_PLANES /
 * * -DSYMBEX_RP2350_U8_USADA8 force one (to measure both on the M33; USADA8 on a core without DSP uses a C stand-in,
 * for the PC tests only).
 * -DSYMBEX_HAL_TEST_RP2350 builds this file on a PC (tests/run_tests.py).
 */
#include <stdint.h>
#include "hal/symbex_hal.h"
#include "core/symbex_core.h"
#include "ops/symbex_swar.h"

#if defined(SYMBEX_HAL_RP2350)

#if defined(SYMBEX_RP2350_U8_USADA8_ON)               /* chosen in symbex_hal.h */
  #define U8_USADA8 1
#else
  #define U8_PLANES 1
#endif

void symbex_u8_prepare(const uint8_t* x, int n_inputs, uint8_t* workspace, int n_words) {
#if defined(U8_PLANES)
    /* 8 planes of n_words words = n_words * 32 bytes, the size every HAL's workspace has */
    uint32_t* planes = (uint32_t*)workspace;
    for (int p = 0; p < n_words; p++) {
        uint32_t pl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int g = 0; g < 8; g++) {                    /* 4 inputs at a time, one per byte of v */
            const int i = p * 32 + g * 4;
            uint32_t v = 0;
            for (int k = 0; k < 4; k++) v |= (uint32_t)(i + k < n_inputs ? x[i + k] : 0) << (8 * k);
            for (int q = 0; q < 8; q++) {
                /* bit q of the 4 bytes (bits 0, 8, 16, 24) -> 4 adjacent bits: the products land on distinct
                   bits (no carries), the 4 wanted ones on bits 28-31 */
                const uint32_t b = (v >> q) & 0x01010101u;
                pl[q] |= ((b * 0x10204080u) >> 28) << (4 * g);
            }
        }
        for (int q = 0; q < 8; q++) planes[p * 8 + q] = pl[q];
    }
#else
    const int bytes = n_words * 32;                      /* natural order, zero padding */
    for (int i = 0; i < bytes; i++) workspace[i] = (i < n_inputs) ? x[i] : 0;
#endif
}

#if defined(U8_USADA8)
#if defined(__ARM_FEATURE_SIMD32)
#include <arm_acle.h>
#define USADA8(v, acc) __usada8((v), 0u, (acc))
#else
static inline uint32_t usada8_c(uint32_t v, uint32_t acc) {      /* the PC stand-in of usada8(v, 0, acc) */
    return acc + (v & 0xFFu) + ((v >> 8) & 0xFFu) + ((v >> 16) & 0xFFu) + (v >> 24);
}
#define USADA8(v, acc) usada8_c((v), (acc))
#endif

/* the padding of the workspace is 0, so whole words are summed: n_inputs rounded up to 32 */
static inline int32_t usada8_row(const uint32_t* xw, const uint32_t* w, int words) {
    uint32_t acc = 0;
    for (int p = 0; p < words; p++, xw += 8) {
        const uint32_t wv = SYMBEX_READ_W32(&w[p]);
        acc = USADA8(xw[0] & symbex_swar_spread4[wv & 15u], acc);
        acc = USADA8(xw[1] & symbex_swar_spread4[(wv >> 4) & 15u], acc);
        acc = USADA8(xw[2] & symbex_swar_spread4[(wv >> 8) & 15u], acc);
        acc = USADA8(xw[3] & symbex_swar_spread4[(wv >> 12) & 15u], acc);
        acc = USADA8(xw[4] & symbex_swar_spread4[(wv >> 16) & 15u], acc);
        acc = USADA8(xw[5] & symbex_swar_spread4[(wv >> 20) & 15u], acc);
        acc = USADA8(xw[6] & symbex_swar_spread4[(wv >> 24) & 15u], acc);
        acc = USADA8(xw[7] & symbex_swar_spread4[wv >> 28], acc);
    }
    return (int32_t)acc;
}
#endif

#if defined(U8_PLANES)
#if defined(__riscv_zbb) || defined(__POPCNT__) || defined(__aarch64__)
#define POP(x) __builtin_popcount(x)                    /* one instruction (cpop on Hazard3) */
#else
#define POP(x) symbex_hal_popcount32(x)                 /* the M33 with -DSYMBEX_RP2350_U8_PLANES, PC tests */
#endif

/* the padding of the workspace is 0, so only the words that hold inputs are visited */
static inline int32_t planes_row(const uint32_t* x, const uint32_t* w, int words) {
    uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    for (int p = 0; p < words; p++, x += 8) {
        const uint32_t wv = SYMBEX_READ_W32(&w[p]);
        c0 += POP(x[0] & wv); c1 += POP(x[1] & wv); c2 += POP(x[2] & wv); c3 += POP(x[3] & wv);
        c4 += POP(x[4] & wv); c5 += POP(x[5] & wv); c6 += POP(x[6] & wv); c7 += POP(x[7] & wv);
    }
    return (int32_t)(c0 + (c1 << 1) + (c2 << 2) + (c3 << 3) + (c4 << 4) + (c5 << 5) + (c6 << 6) + (c7 << 7));
}
#endif

int32_t symbex_u8_sum_bits(const uint8_t* workspace, const uint32_t* w, int n_words) {
#if defined(U8_PLANES)
    return planes_row((const uint32_t*)workspace, w, n_words);
#else
    return usada8_row((const uint32_t*)workspace, w, n_words);
#endif
}

void symbex_u8_sum_bits_rows(const uint8_t* workspace, int n_inputs, const uint32_t* w, int stride, int n_rows,
                             int32_t* out) {
    if (n_inputs > stride * 32) n_inputs = stride * 32;
    const int words = (n_inputs + 31) / 32;
#if defined(U8_PLANES)
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = planes_row((const uint32_t*)workspace, w, words);
#else
    for (int r = 0; r < n_rows; r++, w += stride) out[r] = usada8_row((const uint32_t*)workspace, w, words);
#endif
}

#endif // SYMBEX_HAL_RP2350
