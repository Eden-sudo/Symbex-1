#ifndef SYMBEX_CLUSTER_HEAD_H
#define SYMBEX_CLUSTER_HEAD_H
#include <stdint.h>

/* Host reference, unscaled int8 rows; not a firmware/export ABI.
 * h[dim], weights[vocab*dim], labels[vocab], centers[clusters*dim].
 * Caller provides ids[vocab], scores[vocab], selected[k], count[1].
 * All arrays must be valid, correctly sized, and mutually nonoverlapping.
 * Returns 0 on success, -1 on invalid dimensions/labels/null pointers,
 * -2 on an empty shortlist. Outputs remain unchanged on either failure.
 * Limits: vocab <= 65536, dim <= 65536, clusters <= 256, 1 <= k <= clusters.
 * Scores are int64 dot products; ties choose lower cluster/token IDs.
 * IDs are ascending, selected clusters are in descending routing-score order.
 */
int symbex_cluster_head(const int8_t *h, const int8_t *weights,
                        const int32_t *labels, const int8_t *centers,
                        int vocab, int dim, int clusters, int k,
                        int32_t *ids, int64_t *scores, int32_t *selected,
                        int32_t *count);
#endif
