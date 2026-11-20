#include "gqa_attention.h"
#include <stddef.h>

static int valid_values(const int8_t *x, size_t n) {
    for (size_t i = 0; i < n; ++i) if (x[i] == INT8_MIN) return 0;
    return 1;
}

static int valid_scales(const int32_t *x, size_t n) {
    for (size_t i = 0; i < n; ++i) if (x[i] < 1 || x[i] > (1 << 20)) return 0;
    return 1;
}

static int64_t rounded_div(int64_t n, int64_t d) {
    return n < 0 ? -((-n + d / 2) / d) : (n + d / 2) / d;
}

int gqa_attention(int heads, int kv_heads, int length,
                  const int8_t *q, const int32_t *qm,
                  const int8_t *k, const int32_t *km,
                  const int8_t *v, const int32_t *vm,
                  const uint16_t *exp_table, int64_t *out) {
    if (heads < 1 || heads > 64 || kv_heads < 1 || heads % kv_heads ||
        length < 1 || length > 512 || !q || !qm || !k || !km || !v || !vm ||
        !exp_table || !out || !exp_table[0]) return -1;
    size_t count = (size_t)kv_heads * length;
    if (!valid_values(q, (size_t)heads * 64) || !valid_values(k, count * 64) ||
        !valid_values(v, count * 64) || !valid_scales(qm, heads) ||
        !valid_scales(km, count) || !valid_scales(vm, count)) return -1;
    for (int h = 0; h < heads; ++h) {
        int kh = h / (heads / kv_heads);
        int64_t scores[512], maximum = INT64_MIN, total = 0;
        uint16_t probabilities[512];
        for (int t = 0; t < length; ++t) {
            size_t row = (size_t)kh * length + t;
            int64_t dot = 0;
            for (int d = 0; d < 64; ++d)
                dot += (int32_t)q[h * 64 + d] * k[row * 64 + d];
            scores[t] = dot * km[row];
            if (scores[t] > maximum) maximum = scores[t];
        }
        for (int t = 0; t < length; ++t) {
            /* Domain limits keep this product below 2^62. */
            int64_t index = (maximum - scores[t]) * qm[h] / (INT64_C(16129) << 20);
            probabilities[t] = index < 2048 ? exp_table[index] : 0;
            total += probabilities[t];
        }
        for (int d = 0; d < 64; ++d) {
            int64_t acc = 0;
            for (int t = 0; t < length; ++t) {
                size_t row = (size_t)kh * length + t;
                acc += (int64_t)probabilities[t] * vm[row] * v[row * 64 + d];
            }
            out[h * 64 + d] = rounded_div(acc, total * 127);
        }
    }
    return 0;
}
