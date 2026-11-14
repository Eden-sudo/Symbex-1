/* symbex_fomo.h -- centers of 8-connected same-class cell groups. */
#ifndef SYMBEX_FOMO_H
#define SYMBEX_FOMO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SymbexFomoCenter {
    uint8_t class_id;
    uint32_t n;
    uint32_t sum_r;
    uint32_t sum_c;
};

/*
 * cells is a row-major h by w grid, with zero as background. h * w must
 * not exceed UINT16_MAX; stack has h * w entries and is used as scratch.
 * Buffers must not overlap. The total group count is returned, even when
 * only the first max_out groups fit in out. out may be NULL if max_out is 0.
 * Empty grids return 0; invalid arguments return -1 without writing buffers.
 * Pixel centers: x = ((double)sum_c / n + 0.5) * cell_size,
 *                y = ((double)sum_r / n + 0.5) * cell_size.
 */
int symbex_fomo_centers(const uint8_t* cells, uint16_t h, uint16_t w,
                        struct SymbexFomoCenter* out, int max_out, uint16_t* stack);

#ifdef __cplusplus
}
#endif

#endif // SYMBEX_FOMO_H
