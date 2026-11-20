import random
import pytest
from tiny_runtime_support import compile_c, module, run


def test_sample(tmp_path):
    py = module("sample_int")
    exe = compile_c(tmp_path, r'''
#include <stdio.h>
#include <inttypes.h>
#include "chat_sample.h"
int main(void) {
    int n, k, ids[65], token;
    uint32_t temp, p, state;
    int64_t z[64];
    while (scanf("%d%d%" SCNu32 "%" SCNu32 "%" SCNu32, &n,&k,&temp,&p,&state)==5) {
        for (int i=0;i<n;++i) if(scanf("%" SCNd64,&z[i])!=1) return 1;
        ids[64]=999;
        token=chat_sample(z,n,k,temp,p,&state,ids,64);
        if(ids[64]!=999) return 2;
        printf("%d %" PRIu32 "\n",token,state);
    }
    return 0;
}
''', ["chat_sample.c"])
    rng = random.Random(73)
    cases = []
    for _ in range(300):
        z = [rng.randrange(-100, 101) for _ in range(rng.randrange(1, 65))]
        cases.append((z, rng.choice([0, 1, 5, 99]), rng.choice([0, 1, 7, 0xffffffff]),
                      rng.choice([1, 32768, 65535, 65536]), rng.randrange(1, 1 << 32)))
    cases += [([-(1 << 63), (1 << 63)-1, 0], 0, t, 65536, 1) for t in (0, 1, 0xffffffff)]
    cases += [([0]*5, 0, 1, p, s) for p in (1, 32768, 65536) for s in (1, 42)]
    payload = ''.join(f'{len(z)} {k} {t} {p} {s}\n'+' '.join(map(str,z))+'\n'
                      for z,k,t,p,s in cases)
    actual = [tuple(map(int, line.split())) for line in run(exe, payload).splitlines()]
    assert actual == [py.sample(*case) for case in cases]
    assert py.sample([2, 2, 1], 0, 0, 65536, 42) == (0, 42)
    assert py.sample([0]*5, 0, 1, 1, 1) == (0, 270369)
    for z,k,t,p,s in cases:
        if t and p == 65536:
            assert py.sample(z,k,t,p,s) == py.sample_topk(z,k or len(z),t,s)
    invalid = [([],0,1,1,1), ([1],-1,1,1,1), ([1],0,1,0,1), ([1],0,1,65537,1),
               ([1],0,1,1,0)]
    for args in invalid:
        with pytest.raises(ValueError):
            py.sample(*args)
    # Invalid C calls preserve state, including insufficient scratch.
    bad = "1 -1 1 1 42\n0\n1 0 1 0 42\n0\n1 0 1 65537 42\n0\n"
    assert run(exe, bad).splitlines() == ["-1 42"]*3
