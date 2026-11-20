#include "chat_pieces.h"
#include <string.h>
static uint32_t u32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
int chat_decoder_init(ChatDecoder *d, const unsigned char *blob, size_t size)
{
    if (!d || !blob || size<12 || memcmp(blob,"TCP1",4)) return -1;
    uint32_t n=u32(blob+4);
    if (!n || n>65536) return -1;
    size_t start=8+4*((size_t)n+1);
    if (start>size || u32(blob+8)!=0 || u32(blob+8+4*(size_t)n)!=size-start) return -1;
    for (uint32_t i=0;i<n;++i) if (u32(blob+8+4*(size_t)i)>u32(blob+12+4*(size_t)i)) return -1;
    *d=(ChatDecoder){blob+8,blob+start,n,0};
    return 0;
}
int chat_decode(ChatDecoder *d, int token, const unsigned char **bytes, size_t *size)
{
    if (!d || !bytes || !size || token<0 || (uint32_t)token>=d->vocab) return -1;
    uint32_t a=u32(d->offsets+4*(size_t)token), b=u32(d->offsets+4*((size_t)token+1));
    const unsigned char *p=d->payload+a;
    size_t n=b-a;
    if (n && !d->started) {
        d->started=1;
        if (*p==' ') { ++p; --n; }
    }
    *bytes=p; *size=n;
    return 0;
}
