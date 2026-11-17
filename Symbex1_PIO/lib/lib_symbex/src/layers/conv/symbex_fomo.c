#include "layers/conv/symbex_fomo.h"

int symbex_fomo_centers(const uint8_t* cells, uint16_t h, uint16_t w,
                        struct SymbexFomoCenter* out, int max_out, uint16_t* stack)
{
    uint32_t size = (uint32_t)h * (uint32_t)w;
    int total = 0;

    if (max_out < 0 || size > UINT16_MAX || (size && (!cells || !stack)) ||
        (max_out > 0 && !out)) return -1;
    for (uint32_t i = 0; i < size; ++i) stack[i] = UINT16_MAX;

    for (uint32_t seed = 0; seed < size; ++seed) {
        if (!cells[seed] || stack[seed] != UINT16_MAX) continue;
        uint8_t class_id = cells[seed];
        uint32_t top = seed, n = 0, sum_r = 0, sum_c = 0;
        stack[seed] = (uint16_t)seed;
        while (top != UINT16_MAX) {
            uint32_t p = top, next = stack[p];
            int32_t r = (int32_t)(p / w), c = (int32_t)(p % w);
            top = next == p ? UINT16_MAX : next;
            ++n;
            sum_r += (uint32_t)r;
            sum_c += (uint32_t)c;
            for (int32_t dr = -1; dr <= 1; ++dr) {
                for (int32_t dc = -1; dc <= 1; ++dc) {
                    int32_t nr = r + dr, nc = c + dc;
                    if (nr < 0 || nc < 0 || nr >= h || nc >= w) continue;
                    uint32_t q = (uint32_t)nr * w + (uint32_t)nc;
                    if (cells[q] == class_id && stack[q] == UINT16_MAX) {
                        stack[q] = (uint16_t)(top == UINT16_MAX ? q : top);
                        top = q;
                    }
                }
            }
        }
        if (total < max_out) {
            out[total].class_id = class_id;
            out[total].n = (uint16_t)n;
            out[total].sum_r = sum_r;
            out[total].sum_c = sum_c;
        }
        ++total;
    }
    return total;
}
