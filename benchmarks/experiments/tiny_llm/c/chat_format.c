#include "chat_format.h"
#include <limits.h>
#include <string.h>
static int valid(ChatIds span, int vocab)
{
    if (span.count && !span.ids) return 0;
    if (span.count > INT_MAX) return 0;
    for (size_t i=0;i<span.count;++i) if (span.ids[i]<0 || span.ids[i]>=vocab) return 0;
    return 1;
}
int chat_build(const ChatIds markers[3], const ChatTurn *turns, size_t count,
               int vocab, int *out, size_t capacity)
{
    size_t needed=1, pos=0;
    if (!markers || (count && !turns) || !out || vocab<3 || count>INT_MAX) return -1;
    for (int i=0;i<3;++i) if (!markers[i].count || !valid(markers[i],vocab)) return -1;
    needed+=markers[2].count;
    for (size_t i=0;i<count;++i) {
        if (turns[i].role>2 || !valid(turns[i].content,vocab)) return -1;
        size_t extra=markers[turns[i].role].count+turns[i].content.count+1;
        if (extra>INT_MAX || needed>(size_t)INT_MAX-extra) return -1;
        needed+=extra;
    }
    if (needed>capacity || needed>INT_MAX) return -1;
    out[pos++]=1;
    for (size_t i=0;i<count;++i) {
        ChatIds marker=markers[turns[i].role], text=turns[i].content;
        memcpy(out+pos,marker.ids,marker.count*sizeof(int)); pos+=marker.count;
        if (text.count) memcpy(out+pos,text.ids,text.count*sizeof(int));
        pos+=text.count; out[pos++]=2;
    }
    memcpy(out+pos,markers[2].ids,markers[2].count*sizeof(int));
    return (int)needed;
}
