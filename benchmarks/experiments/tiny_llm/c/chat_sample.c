#include "chat_sample.h"
#include "tinylm_sample.h"

int chat_sample(const int64_t *scores, int n, int k, uint32_t temperature,
                uint32_t top_p, uint32_t *state, int *scratch, int capacity)
{
    int count = 0, effective, end;
    uint64_t total = 0, mass = 0, target, draw;
    if (!scores || n <= 0 || k < 0 || !top_p || top_p > 65536 ||
        !state || !*state || !scratch) return -1;
    effective = !k || k > n ? n : k;
    if (capacity < effective) return -1;
    for (int i = 0; i < n; ++i) {
        int pos = 0;
        while (pos < count && scores[scratch[pos]] >= scores[i]) ++pos;
        if (pos >= effective) continue;
        if (count < effective) ++count;
        for (int j = count - 1; j > pos; --j) scratch[j] = scratch[j - 1];
        scratch[pos] = i;
    }
    if (!temperature) return scratch[0];
    for (int i = 0; i < count; ++i)
        total += tinylm_sample_exp_q((uint64_t)scores[scratch[0]] -
                                    (uint64_t)scores[scratch[i]], temperature);
    /* total <= INT_MAX * 65536, so even total * top_p fits uint64. */
    target = (total * top_p + 65535) / 65536;
    for (end = 0; end < count; ++end) {
        mass += tinylm_sample_exp_q((uint64_t)scores[scratch[0]] -
                                   (uint64_t)scores[scratch[end]], temperature);
        if (mass >= target) { ++end; break; }
    }
    *state = tinylm_sample_xorshift32(*state);
    draw = *state % mass;
    for (int i = 0; i < end; ++i) {
        uint32_t w = tinylm_sample_exp_q((uint64_t)scores[scratch[0]] -
                                       (uint64_t)scores[scratch[i]], temperature);
        if (draw < w) return scratch[i];
        draw -= w;
    }
    return -1;
}
