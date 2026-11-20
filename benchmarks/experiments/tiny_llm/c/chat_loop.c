#include "chat_loop.h"
#include "chat_sample.h"
#include "chat_penalty.h"
int chat_generate(ChatOptions *o, const int *prompt, int count, ChatStop *stop,
                  ChatForward forward, ChatPiece piece, ChatEmit emit, void *ctx,
                  int64_t *logits, int *scratch)
{
    int history[256], used=count;
    if (!o || !prompt || !stop || !forward || !emit || !logits || !scratch ||
        o->vocab<=0 || o->context<1 || o->context>256 || count<1 || count>o->context ||
        o->k<0 || !o->top_p || o->top_p>65536 || !o->seed || o->penalty<0 ||
        stop->eos<0 || stop->eos>=o->vocab || stop->count || (stop->n && !piece)) return -1;
    for (int i=0;i<count;++i) {
        if (prompt[i]<0 || prompt[i]>=o->vocab) return -1;
        history[i]=prompt[i];
    }
    if (stop->reason) return stop->reason;
    if (used==o->context) return 4;
    for (int i=0;i<count;++i) if (forward(ctx,prompt[i],logits)<0) return -1;
    while (used<o->context) {
        const unsigned char *bytes=0; size_t size=0;
        if (chat_penalize(logits,o->vocab,history,used,o->penalty)) return -1;
        int token=chat_sample(logits,o->vocab,o->k,o->temperature,o->top_p,&o->seed,scratch,o->vocab);
        if (token<0) return -1;
        if (token!=stop->eos && piece && piece(ctx,token,&bytes,&size)) return -1;
        int reason=chat_stop_feed(stop,token,bytes,size);
        if (reason<0) return -1;
        if (reason==1) return 1;
        if (emit(ctx,token)) return -1;
        history[used++]=token;
        if (reason) return reason;
        if (used==o->context) return 4;
        if (forward(ctx,token,logits)<0) return -1;
    }
    return 4;
}
