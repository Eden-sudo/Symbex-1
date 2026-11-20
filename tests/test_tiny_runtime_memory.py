import random
from tiny_runtime_support import compile_c, module, run


def test_memory(tmp_path):
    py=module("chat_memory")
    exe=compile_c(tmp_path,r'''
#include <stdio.h>
#include "chat_memory.h"
int main(void) {
    unsigned context,reserve,a,b,n,lens[8]; int ids[300],out[256];
    for(int i=0;i<300;++i) ids[i]=i;
    while(scanf("%u%u%u%u%u",&context,&reserve,&a,&b,&n)==5) {
        ChatIds turns[8]; size_t dropped=999;
        for(unsigned i=0;i<n;++i) {
            if(scanf("%u",&lens[i])!=1) return 1;
            turns[i]=(ChatIds){ids,lens[i]};
        }
        out[0]=999;
        int size=chat_retain((ChatIds){ids,a},turns,n,(ChatIds){ids,b},reserve,context,out,256,&dropped);
        if(size<0 && (out[0]!=999 || dropped!=999)) return 2;
        printf("%d %zu",size,dropped);
        for(int i=0;i<size;++i) printf(" %d",out[i]);
        puts("");
    }
    return 0;
}
''',["chat_memory.c"])
    rng=random.Random(4)
    cases=[(256,1,3,2,[80,80,80,80]),(256,0,256,0,[]),(256,1,0,0,[256]),
           (256,1,1,1,[300,2]),(256,1,1,1,[0,0])]
    cases += [(rng.randint(1,256),rng.randint(0,10),rng.randint(0,30),rng.randint(0,10),
               [rng.randint(0,100) for _ in range(rng.randint(0,8))]) for _ in range(100)]
    payload=''.join(' '.join(map(str,[c,r,a,b,len(ls),*ls]))+'\n' for c,r,a,b,ls in cases)
    for line,(c,r,a,b,ls) in zip(run(exe,payload).splitlines(),cases):
        actual=list(map(int,line.split()))
        try:
            ids,dropped=py.retain(list(range(a)),[list(range(n)) for n in ls],list(range(b)),r,c)
        except ValueError:
            assert actual==[-1,999]
        else:
            assert actual==[len(ids),dropped,*ids]
    assert py.retain([1],[[10]*100,[20]*100,[30]*100],[9])[1]==1
