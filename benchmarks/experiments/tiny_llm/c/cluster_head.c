#include "cluster_head.h"
#include <stddef.h>

static int64_t dot(const int8_t *a, const int8_t *b, int dim) {
    int64_t sum = 0;
    for (int j = 0; j < dim; ++j)
        sum += (int64_t)a[j] * (int64_t)b[j];
    return sum;
}

int symbex_cluster_head(const int8_t *h, const int8_t *weights,
                        const int32_t *labels, const int8_t *centers,
                        int vocab, int dim, int clusters, int k,
                        int32_t *ids, int64_t *scores, int32_t *selected,
                        int32_t *count) {
    if (!h || !weights || !labels || !centers || !ids || !scores ||
        !selected || !count || vocab < 1 || vocab > 65536 ||
        dim < 1 || dim > 65536 || clusters < 1 || clusters > 256 ||
        k < 1 || k > clusters)
        return -1;
    if ((size_t)vocab > SIZE_MAX / (size_t)dim)
        return -1;
    for (int i = 0; i < vocab; ++i)
        if (labels[i] < 0 || labels[i] >= clusters)
            return -1;

    int64_t routing[256];
    unsigned char chosen[256] = {0};
    int32_t order[256];
    for (int c = 0; c < clusters; ++c)
        routing[c] = dot(h, centers + (size_t)c * (size_t)dim, dim);
    for (int n = 0; n < k; ++n) {
        int best = -1;
        for (int c = 0; c < clusters; ++c)
            if (!chosen[c] && (best < 0 || routing[c] > routing[best]))
                best = c;
        order[n] = best;
        chosen[best] = 1;
    }
    int n = 0;
    for (int i = 0; i < vocab; ++i)
        n += chosen[labels[i]] != 0;
    if (!n)
        return -2;

    n = 0;
    for (int i = 0; i < vocab; ++i) {
        if (chosen[labels[i]]) {
            ids[n] = i;
            scores[n] = dot(h, weights + (size_t)i * (size_t)dim, dim);
            ++n;
        }
    }
    for (int c = 0; c < k; ++c)
        selected[c] = order[c];
    *count = n;
    return 0;
}
