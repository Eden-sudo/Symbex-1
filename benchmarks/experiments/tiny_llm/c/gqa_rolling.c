#include "gqa_rolling.h"
#include <stdlib.h>
#include <string.h>

static int64_t floor_q14(int64_t n) {
    return n >= 0 ? n / 16384 : -((-n + 16383) / 16384);
}

static int valid_input(const int32_t *x, int heads) {
    for (int i = 0; i < heads * 64; ++i)
        if (x[i] < -(1 << 19) || x[i] > (1 << 19)) return 0;
    return 1;
}

static void quantize(const int32_t *x, int heads, const int32_t *c, const int32_t *s,
                     int8_t *q, int32_t *maxima) {
    for (int h = 0; h < heads; ++h) {
        int64_t values[64], maximum = 1;
        for (int d = 0; d < 64; d += 2) {
            int64_t a = x[h * 64 + d], b = x[h * 64 + d + 1];
            values[d] = c ? floor_q14(a * c[d/2] - b * s[d/2] + 8192) : a;
            values[d+1] = c ? floor_q14(a * s[d/2] + b * c[d/2] + 8192) : b;
        }
        for (int d = 0; d < 64; ++d) {
            int64_t magnitude = values[d] < 0 ? -values[d] : values[d];
            if (magnitude > maximum) maximum = magnitude;
        }
        maxima[h] = (int32_t)maximum;
        for (int d = 0; d < 64; ++d) {
            int64_t n = values[d] * 127;
            q[h * 64 + d] = (int8_t)(n < 0 ? -((-n + maximum/2) / maximum)
                                                         : (n + maximum/2) / maximum);
        }
    }
}

int gqa_ring_step(gqa_ring *r, const int32_t *q, const int32_t *k,
                  const int32_t *v, const uint16_t *exp_table, int64_t *out) {
    if (!r || r->heads < 1 || r->heads > 64 || r->kv_heads < 1 ||
        r->heads % r->kv_heads || r->window < 1 || r->window > 512 ||
        r->max_positions < 1 || r->position >= (uint64_t)r->max_positions ||
        !r->k || !r->v || !r->km || !r->vm || !r->cos || !r->sin ||
        !q || !k || !v || !exp_table || !exp_table[0] || !out) return -1;
    if (!valid_input(q, r->heads) || !valid_input(k, r->kv_heads) || !valid_input(v, r->kv_heads)) return -1;
    const int32_t *c = r->cos + (size_t)r->position * 32;
    const int32_t *s = r->sin + (size_t)r->position * 32;
    for (int i = 0; i < 32; ++i)
        if (c[i] < -16384 || c[i] > 16384 || s[i] < -16384 || s[i] > 16384) return -1;
    int length = r->position + 1 < (uint64_t)r->window ? (int)r->position + 1 : r->window;
    size_t rows = (size_t)r->kv_heads * length;
    int8_t *keys = malloc(rows * 64), *values = malloc(rows * 64);
    int32_t *km = malloc(rows * sizeof(*km)), *vm = malloc(rows * sizeof(*vm));
    if (!keys || !values || !km || !vm) {
        free(keys); free(values); free(km); free(vm);
        return -2;
    }
    int8_t qq[64 * 64], kq[64 * 64], vq[64 * 64];
    int32_t qm[64], new_km[64], new_vm[64];
    quantize(q, r->heads, c, s, qq, qm);
    quantize(k, r->kv_heads, c, s, kq, new_km);
    quantize(v, r->kv_heads, NULL, NULL, vq, new_vm);
    uint64_t start = r->position + 1 - length;
    for (int h = 0; h < r->kv_heads; ++h) {
        for (int t = 0; t < length; ++t) {
            size_t dest = (size_t)h * length + t;
            size_t src = (size_t)h * r->window + (start + t) % r->window;
            int newest = t == length - 1;
            memcpy(keys + dest * 64, newest ? kq + h * 64 : r->k + src * 64, 64);
            memcpy(values + dest * 64, newest ? vq + h * 64 : r->v + src * 64, 64);
            km[dest] = newest ? new_km[h] : r->km[src];
            vm[dest] = newest ? new_vm[h] : r->vm[src];
        }
    }
    int status = gqa_attention(r->heads, r->kv_heads, length, qq, qm, keys, km, values, vm, exp_table, out);
    if (status == 0) {
        for (int h = 0; h < r->kv_heads; ++h) {
            size_t row = (size_t)h * r->window + r->position % r->window;
            memcpy(r->k + row * 64, kq + h * 64, 64);
            memcpy(r->v + row * 64, vq + h * 64, 64);
            r->km[row] = new_km[h];
            r->vm[row] = new_vm[h];
        }
        ++r->position;
    }
    free(keys); free(values); free(km); free(vm);
    return status;
}
