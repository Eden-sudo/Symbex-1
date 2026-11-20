import subprocess
from tiny_runtime_support import compile_c, run, LM


def test_loop(tmp_path):
    exe=compile_c(tmp_path,r'''
#include <stdio.h>
#include <string.h>
#include "chat_loop.h"
typedef struct { int calls, emitted, mode; } Stub;
static int forward(void *ctx,int token,int64_t *z) {
    Stub *s=ctx; (void)token;
    ++s->calls;
    for(int i=0;i<8;++i) z[i]=0;
    z[s->mode==1 ? 2 : 3]=100;
    return s->mode==2 ? -1 : 0;
}
static int emit(void *ctx,int token) {
    Stub *s=ctx; if(token!=3) return -1; ++s->emitted; return 0;
}
static int piece(void *ctx,int token,const unsigned char **p,size_t *n) {
    (void)ctx; (void)token; *p=(const unsigned char*)"a"; *n=1; return 0;
}
int main(void) {
    int prompt[]={1,4},scratch[8]; int64_t z[8];
    for(int mode=0;mode<6;++mode) {
        Stub stub={0,0,mode}; ChatStop stop;
        const unsigned char *strings[]={(const unsigned char*)"aa"}; unsigned lens[]={2};
        if(chat_stop_init(&stop,2,mode==4?0:3,strings,lens,mode==3?1:0)) return 1;
        ChatOptions o={8,mode==5?3:8,0,0,65536,1,0};
        int reason=chat_generate(&o,prompt,2,&stop,forward,piece,emit,&stub,z,scratch);
        printf("%d %d %d\n",reason,stub.calls,stub.emitted);
    }
    return 0;
}
''',["chat_loop.c","chat_sample.c","chat_penalty.c","chat_stop.c"])
    assert run(exe).splitlines()==["3 4 3","1 2 0","-1 1 0","2 3 2","3 0 0","4 2 1"]


def test_host_build(tmp_path):
    # No synthetic model fixture exists in repository tests; build real backend,
    # exercise inference only through the tiny stub above.
    exe=tmp_path/"chat_host"
    files=["chat_host.c","chat_loop.c","chat_sample.c","chat_penalty.c","chat_stop.c","tinylm.c"]
    subprocess.run(["/usr/bin/clang","-std=c99","-O1","-Wall","-Wextra","-Werror",
                    *(str(LM/"c"/f) for f in files),"-o",str(exe)],
                   check=True,capture_output=True,text=True,timeout=15)
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=5)
    assert result.returncode==1 and "usage:" in result.stderr
