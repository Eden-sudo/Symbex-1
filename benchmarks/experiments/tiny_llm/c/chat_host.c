/* Host token-ID chat driver. Input lines are complete retained/framed prompts,
 * not raw text. Each line resets/replays KV state, then emits generated IDs.
 * Usage: chat_host trusted-model.bin [max_tokens=32]. No model downloads.
 * Plain host tinylm only (not the chip allocation/PIE variants). */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include "tinylm.h"
#include "chat_loop.h"
typedef struct { TinyModel model; TinyState state; } Host;
static int forward(void *ctx, int token, int64_t *logits)
{
    Host *h=ctx; int64_t maximum;
    return tinylm_step(&h->model,&h->state,token,logits,&maximum);
}
static int emit(void *ctx, int token)
{
    (void)ctx;
    return printf("%d ",token)<0 ? -1 : 0;
}
int main(int argc, char **argv)
{
    Host h={0}; FILE *f=NULL; unsigned char *blob=NULL;
    int64_t *logits=NULL; int *scratch=NULL; int result=1;
    long size, limit=32; char *end, line[8192];
    if (argc<2 || argc>3) { fprintf(stderr,"usage: chat_host model.bin [max_tokens]\n"); return 1; }
    if (argc==3) {
        errno=0; limit=strtol(argv[2],&end,10);
        if (errno || !*argv[2] || *end || limit<0 || limit>256) return 1;
    }
    f=fopen(argv[1],"rb");
    if (!f || fseek(f,0,SEEK_END) || (size=ftell(f))<32 || size>16*1024*1024 || fseek(f,0,SEEK_SET)) goto done;
    blob=malloc((size_t)size);
    if (!blob || fread(blob,1,(size_t)size,f)!=(size_t)size) goto done;
    fclose(f); f=NULL;
    if (tinylm_load(&h.model,blob,size) || h.model.seq>256 || h.model.vocab<3 ||
        h.model.vocab>65536 || tinylm_state_init(&h.state,&h.model)) goto done;
    logits=malloc((size_t)h.model.vocab*sizeof(*logits));
    scratch=malloc((size_t)h.model.vocab*sizeof(*scratch));
    if (!logits || !scratch) goto done;
    while (fgets(line,sizeof(line),stdin)) {
        int prompt[256], count=0; char *p=line;
        /* Reject split/overlong lines instead of treating them as another turn. */
        size_t length=0; while(line[length]) ++length;
        if (length==sizeof(line)-1 && line[length-1]!='\n') goto done;
        while (*p) {
            while (isspace((unsigned char)*p)) ++p;
            if (!*p) break;
            errno=0; long token=strtol(p,&end,10);
            if (errno || end==p || (*end && !isspace((unsigned char)*end)) ||
                token<0 || token>=h.model.vocab || count==256) goto done;
            prompt[count++]=(int)token; p=end;
        }
        if (!count) continue;
        h.state.pos=0; /* Replay overwrites every live KV position. */
        ChatStop stop;
        if (chat_stop_init(&stop,2,(uint32_t)limit,NULL,NULL,0)) goto done;
        ChatOptions options={h.model.vocab,h.model.seq,40,1,58982,1,0};
        int reason=chat_generate(&options,prompt,count,&stop,forward,NULL,emit,&h,logits,scratch);
        if (reason<0 || putchar('\n')==EOF || fflush(stdout)) goto done;
        fprintf(stderr,"stop=%d tokens=%u\n",reason,stop.count);
    }
    if (ferror(stdin)) goto done;
    result=0;
done:
    if (result) fprintf(stderr,"chat_host: invalid input/model or runtime failure\n");
    if (f) fclose(f);
    free(logits); free(scratch); tinylm_state_free(&h.state);
    free(h.model.blocks); free(blob);
    return result;
}
