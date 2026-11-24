#include "chat_memory.h"
#include <string.h>
static int valid(ChatIds s)
{
    if (s.count && !s.ids) return 0;
    /* Every retained span must fit the fixed model context. */
    if (s.count>256) return 0;
    for (size_t i=0;i<s.count;++i) if (s.ids[i]<0) return 0;
    return 1;
}
int chat_retain(ChatIds system, const ChatIds *turns, size_t count, ChatIds suffix,
                unsigned reserve, unsigned context, int *out, size_t capacity, size_t *dropped)
{
    size_t first=count, available, used, pos=0;
    if (!out || !dropped || (count && !turns) || !context || context>256 ||
        reserve>context || !valid(system) || !valid(suffix)) return -1;
    for (size_t i=0;i<count;++i) {
        /* Old oversized turns are droppable; validate IDs without a size cap. */
        if (turns[i].count && !turns[i].ids) return -1;
        for (size_t j=0;j<turns[i].count;++j) if (turns[i].ids[j]<0) return -1;
    }
    used=system.count+suffix.count;
    if (used>context-reserve) return -1;
    available=context-reserve-used;
    if (count && turns[count-1].count>available) return -1;
    while (first && turns[first-1].count<=available) {
        --first; available-=turns[first].count; used+=turns[first].count;
    }
    if (capacity<used) return -1;
    if (system.count) memcpy(out,system.ids,system.count*sizeof(int));
    pos=system.count;
    for (size_t i=first;i<count;++i) {
        if (turns[i].count) memcpy(out+pos,turns[i].ids,turns[i].count*sizeof(int));
        pos+=turns[i].count;
    }
    if (suffix.count) memcpy(out+pos,suffix.ids,suffix.count*sizeof(int));
    *dropped=first;
    return (int)used;
}
