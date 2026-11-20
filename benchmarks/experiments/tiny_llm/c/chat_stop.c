#include "chat_stop.h"
#include <string.h>
int chat_stop_init(ChatStop *s, int eos, uint32_t limit,
                   const unsigned char *const *strings, const unsigned *lengths, unsigned n)
{
    ChatStop fresh = {0};
    if (!s || eos < 0 || n > 8 || (n && (!strings || !lengths))) return -1;
    for (unsigned i=0;i<n;++i)
        if (!strings[i] || !lengths[i] || lengths[i]>64) return -1;
    fresh.eos=eos; fresh.limit=limit; fresh.n=n; fresh.reason=limit ? 0 : 3;
    for (unsigned i=0;i<n;++i) {
        fresh.lengths[i]=lengths[i];
        memcpy(fresh.strings[i],strings[i],lengths[i]);
    }
    *s=fresh;
    return 0;
}
int chat_stop_feed(ChatStop *s, int token, const unsigned char *piece, size_t size)
{
    if (!s || token<0 || (size && !piece)) return -1;
    if (s->reason) return s->reason;
    if (token==s->eos) return s->reason=1;
    ++s->count;
    for (size_t j=0;j<size;++j) {
        if (s->used==64) { memmove(s->tail,s->tail+1,63); --s->used; }
        s->tail[s->used++]=piece[j];
        for (unsigned i=0;i<s->n;++i)
            if (s->used>=s->lengths[i] &&
                !memcmp(s->tail+s->used-s->lengths[i],s->strings[i],s->lengths[i]))
                return s->reason=2;
    }
    if (s->count>=s->limit) s->reason=3;
    return s->reason;
}
