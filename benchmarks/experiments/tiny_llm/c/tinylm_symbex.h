/*
 * tinylm_symbex.h -- tinylm.c's linear layers on lib_symbex's 8-bit x 1-bit kernel (the first-layer kernel of the
 * SYMBEX benchmarks): symbex_u8_prepare + symbex_u8_sum_bits_rows (PIE on the ESP32-S3, SWAR / C elsewhere).
 * Include it before tinylm.c:
 *     #include "tinylm_symbex.h"
 *     #include "tinylm.c"
 * Build with lib_symbex (include/ on the path, its src/ compiled). On the PC (generic HAL), from c/:
 *     L=../../../../Symbex1_PIO/lib/lib_symbex
 *     gcc -O2 -Wall -Wextra -include tinylm_symbex.h -I$L/include -o check_pc_symbex check_pc.c tinylm.c \
 *         $L/src/hal/generic/symbex_hal_impl.c $L/src/ops/symbex_*.c && ./check_pc_symbex ../runs/binary_d384l8_s42
 */
#ifndef TINYLM_SYMBEX_H
#define TINYLM_SYMBEX_H

#include <stdint.h>
#include "hal/symbex_hal.h"

#define TINYLM_U8_MAX_IN 1024

/* the prepared inputs (x + 128, the layout symbex_u8_sum_bits needs), 16-byte aligned for PIE: one per state of the
   whole program (tinylm keeps one TinyState) */
static uint8_t tinylm_u8_workspace[TINYLM_U8_MAX_IN] __attribute__((aligned(16)));

static inline void tinylm_u8_prepare(const int8_t *q, int n_in) {
    uint8_t u[TINYLM_U8_MAX_IN];
    for (int i = 0; i < n_in; i++) u[i] = (uint8_t)(q[i] + 128);
    symbex_u8_prepare(u, n_in, tinylm_u8_workspace, (n_in + 31) / 32);
}

/* the 1-bit input layers (tinylm.c linear_bits): XOR + popcount rows; the input copied to a 16-byte aligned buffer
   (PIE needs it, and the rows: model.bin keeps them aligned) */
static uint32_t tinylm_xor_input[TINYLM_U8_MAX_IN / 32] __attribute__((aligned(16)));

static inline void tinylm_xor_rows(const uint32_t *a, const uint32_t *w, int words, int n_rows, int32_t *out) {
    for (int j = 0; j < words; j++) tinylm_xor_input[j] = a[j];
    symbex_xor_popcount_rows(tinylm_xor_input, w, words, n_rows, out);
}

#define TINYLM_XOR_ROWS(a, w, words, n_rows, out) tinylm_xor_rows((a), (w), (words), (n_rows), (out))
#define TINYLM_U8_PREPARE(q, n_in) tinylm_u8_prepare((q), (n_in))
#define TINYLM_U8_SUM_ROWS(n_in, w, words_in, n_out, out) \
    symbex_u8_sum_bits_rows(tinylm_u8_workspace, (n_in), (w), (words_in), (n_out), (out))

#endif
