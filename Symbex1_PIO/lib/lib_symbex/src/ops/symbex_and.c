/*
 * symbex_and.c -- AND arithmetic for binary layers (see include/ops/symbex_and.h). Portable: the hardware
 * kernels live in the HAL (symbex_and_popcount_rows, symbex_u8_prepare, symbex_u8_sum_bits_rows).
 * Each layer is ONE call to the HAL (*_rows): the per-neuron setup is paid once per layer.
 */
#include <stdint.h>
#include "ops/symbex_and.h"
#include "hal/symbex_hal.h"

/* count(a) over the first n_in bits only */
static int32_t count_input(const uint32_t* a, uint16_t n_in) {
    int32_t c = 0;
    const uint16_t full = n_in / 32, rest = n_in % 32;
    for (uint16_t p = 0; p < full; p++) c += symbex_hal_popcount32(a[p]);
    if (rest) c += symbex_hal_popcount32(a[full] & ((1UL << rest) - 1UL));
    return c;
}

void symbex_and_raw(const uint32_t* a, uint16_t n_in, const uint32_t* w, uint16_t stride, uint16_t n_out,
                    const int32_t* w_count, int32_t* raw) {
    const int32_t ca = count_input(a, n_in);
    /* w padding bits are 0, so the bits of a past n_in do not count */
    symbex_and_popcount_rows(a, w, stride, n_out, raw);
    for (uint16_t j = 0; j < n_out; j++) raw[j] = 4 * raw[j] - 2 * ca - 2 * w_count[j] + (int32_t)n_in;
}

void symbex_and_input_u8(const uint8_t* x, uint16_t n_in, const uint32_t* w, uint16_t stride, uint16_t n_out,
                         uint8_t* work, int32_t* raw) {
    int32_t total = 0;
    for (uint16_t i = 0; i < n_in; i++) total += x[i];
    symbex_u8_prepare(x, n_in, work, stride);
    symbex_u8_sum_bits_rows(work, n_in, w, stride, n_out, raw);
    for (uint16_t j = 0; j < n_out; j++) raw[j] = 2 * raw[j] - total;
}
