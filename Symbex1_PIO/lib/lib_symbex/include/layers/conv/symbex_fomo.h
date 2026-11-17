/* symbex_fomo.h -- centers of 8-connected class cells. */
#ifndef SYMBEX_FOMO_H
#define SYMBEX_FOMO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SymbexFomoCenter {
    uint8_t class_id;
    uint16_t n;
    uint32_t sum_r;
    uint32_t sum_c;
};

/*
 * cells is a row-major h by w grid; zero is background. Groups of equal
 * nonzero class are emitted in order of their first cell. Returns the total
 * number of groups, writing at most max_out, or -1 for invalid arguments.
 * stack provides h*w scratch entries; buffers must not overlap.
 *
 * Pixel center (x, y) is (((sum_c / n) + 0.5) * cell_size,
 * ((sum_r / n) + 0.5) * cell_size), using real division, not C integer division.
 */
int symbex_fomo_centers(const uint8_t* cells, uint16_t h, uint16_t w,
                        struct SymbexFomoCenter* out, int max_out, uint16_t* stack);

#ifdef __cplusplus
}
#endif

#endif /* SYMBEX_FOMO_H */
