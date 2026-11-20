import random
import pytest
from tiny_runtime_support import compile_c, module, run


def test_penalty(tmp_path):
    py = module("chat_penalty")
    exe = compile_c(tmp_path, r'''
#include <stdio.h>
#include <inttypes.h>
#include "chat_penalty.h"
int main(void) {
    int n,h,ids[32]; int64_t p,z[16];
    while(scanf("%d%d%" SCNd64,&n,&h,&p)==3) {
        for(int i=0;i<n;++i) if(scanf("%" SCNd64,&z[i])!=1) return 1;
        for(int i=0;i<h;++i) if(scanf("%d",&ids[i])!=1) return 2;
        printf("%d",chat_penalize(z,n,ids,h,p));
        for(int i=0;i<n;++i) printf(" %" PRId64,z[i]);
        puts("");
    }
    return 0;
}
''', ["chat_penalty.c"])
    rng=random.Random(14)
    cases=[([rng.randrange(-100,100) for _ in range(16)],
            [rng.randrange(16) for _ in range(32)], rng.randrange(100)) for _ in range(100)]
    cases += [([-(1<<63), -(1<<63)+2, (1<<63)-1, 0], [0,1,2,2], (1<<63)-1),
              ([5,-5], [0,0,1], 3), ([5,-5], [], 10)]
    data=''.join(f'{len(z)} {len(h)} {p}\n'+ ' '.join(map(str,z+h))+'\n' for z,h,p in cases)
    actual=[list(map(int,line.split())) for line in run(exe,data).splitlines()]
    assert actual==[[0]+py.penalize(z,h,p) for z,h,p in cases]
    assert py.penalize([5,-5],[0,0,1],3)==[2,-8]
    assert run(exe,"2 2 3\n5 -5 0 2\n").strip()=="-1 5 -5"
    with pytest.raises(ValueError): py.penalize([5,-5],[0,2],3)
