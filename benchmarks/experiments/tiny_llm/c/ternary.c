#include "ternary.h"

int ternary_matvec_base3(const uint8_t *packed, size_t packed_len,
                        size_t rows, size_t cols, const int8_t *x, int64_t *out) {
    if (!packed || !x || !out || !rows || !cols || cols > INT64_MAX / 128)
        return -1;
    const size_t stride = cols / 5 + (cols % 5 != 0);
    if (rows > SIZE_MAX / stride || packed_len != rows * stride)
        return -1;
    unsigned tail_limit = 1;
    for (size_t i = 0; i < cols % 5; ++i) tail_limit *= 3;
    /* Validate every row before the first output write. */
    for (size_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < stride; ++b) {
            const unsigned v = packed[r * stride + b];
            if (v >= 243 || (cols % 5 && b == stride - 1 && v >= tail_limit))
                return -1;
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        int64_t sum = 0;
        for (size_t b = 0; b < stride; ++b) {
            unsigned v = packed[r * stride + b];
            for (size_t j = 0; j < 5 && b * 5 + j < cols; ++j) {
                const int weight = (int)(v % 3) - 1;
                sum += (int64_t)weight * x[b * 5 + j];
                v /= 3;
            }
        }
        out[r] = sum;
    }
    return 0;
}
