#include "layers/conv/symbex_fomo.h"

static uint16_t root(uint16_t* parents, uint16_t i)
{
    uint16_t r = i;
    while (parents[r] != r) r = parents[r];
    while (parents[i] != i) {
        uint16_t next = parents[i];
        parents[i] = r;
        i = next;
    }
    return r;
}

static void join(uint16_t* parents, uint16_t a, uint16_t b)
{
    a = root(parents, a);
    b = root(parents, b);
    if (a < b) parents[b] = a;
    else parents[a] = b;
}

int symbex_fomo_centers(const uint8_t* cells, uint16_t h, uint16_t w,
                        struct SymbexFomoCenter* out, int max_out, uint16_t* stack)
{
    uint32_t count = (uint32_t)h * w;
    uint32_t i;
    int total = 0;
    if (max_out < 0 || count > UINT16_MAX) return -1;
    if (count == 0) return 0;
    if (!cells || !stack || (max_out > 0 && !out)) return -1;

    for (i = 0; i < count; ++i)
        stack[i] = cells[i] ? (uint16_t)i : UINT16_MAX;
    for (i = 0; i < count; ++i) {
        uint32_t r = i / w, c = i % w;
        if (!cells[i]) continue;
        if (c && cells[i - 1] == cells[i]) join(stack, (uint16_t)i, (uint16_t)(i - 1));
        if (r && c && cells[i - w - 1] == cells[i]) join(stack, (uint16_t)i, (uint16_t)(i - w - 1));
        if (r && cells[i - w] == cells[i]) join(stack, (uint16_t)i, (uint16_t)(i - w));
        if (r && c + 1 < w && cells[i - w + 1] == cells[i]) join(stack, (uint16_t)i, (uint16_t)(i - w + 1));
    }
    for (i = 0; i < count; ++i)
        if (stack[i] != UINT16_MAX) stack[i] = root(stack, (uint16_t)i);
    for (i = 0; i < count; ++i) {
        uint16_t parent = stack[i];
        int group;
        if (parent == UINT16_MAX) continue;
        if (parent == i) {
            group = total++;
            stack[i] = (uint16_t)group;
            if (group < max_out) {
                out[group].class_id = cells[i];
                out[group].n = 0;
                out[group].sum_r = 0;
                out[group].sum_c = 0;
            }
        } else {
            group = stack[parent];
        }
        if (group < max_out) {
            ++out[group].n;
            out[group].sum_r += i / w;
            out[group].sum_c += i % w;
        }
    }
    return total;
}
