#include "chat_penalty.h"
#include <stdint.h>
int chat_penalize(int64_t *scores, int n, const int *history, int count, int64_t penalty)
{
    if (!scores || n <= 0 || count < 0 || (count && !history) || penalty < 0) return -1;
    for (int i=0;i<count;++i) if (history[i]<0 || history[i]>=n) return -1;
    for (int i=0;i<count;++i) {
        int duplicate=0, t=history[i];
        for (int j=0;j<i;++j) if (history[j]==t) { duplicate=1; break; }
        if (!duplicate)
            scores[t] = scores[t] < INT64_MIN+penalty ? INT64_MIN : scores[t]-penalty;
    }
    return 0;
}
