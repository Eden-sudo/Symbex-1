from tiny_runtime_support import compile_c, module, run


def test_stop(tmp_path):
    py=module("chat_stop")
    exe=compile_c(tmp_path,r'''
#include <stdio.h>
#include "chat_stop.h"
int main(void) {
    ChatStop s; unsigned lens[]={4,3};
    const unsigned char *stops[]={(const unsigned char*)"abab",(const unsigned char*)"x\0y"};
    unsigned limit,n,byte; int token;
    while(scanf("%u",&limit)==1) {
        if(chat_stop_init(&s,2,limit,stops,lens,2)) return 1;
        for(int i=0;i<5;++i) {
            unsigned char piece[128];
            if(scanf("%d%u",&token,&n)!=2 || n>128) return 2;
            for(unsigned j=0;j<n;++j) { if(scanf("%u",&byte)!=1) return 3; piece[j]=(unsigned char)byte; }
            int reason=chat_stop_feed(&s,token,piece,n);
            printf("%d %u\n",reason,s.count);
        }
    }
    return 0;
}
''',["chat_stop.c"])
    cases=[(10,[(1,b"a"),(1,b"ba"),(1,b"b!"),(1,b"x"),(2,b"")]),
           (10,[(1,b"x"),(1,b"\0"),(1,b"y"),(1,b""),(2,b"")]),
           (2,[(1,b""),(1,b""),(1,b"abab"),(2,b""),(1,b"z")]),
           (0,[(1,b"a")]*5), (10,[(2,b"abab")]*5),
           (1,[(1,b"q"*100+b"abab")]*5)]
    payload=""; expected=[]
    for limit,items in cases:
        payload+=f"{limit}\n"
        s=py.Stop(2,limit,[b"abab",b"x\0y"])
        for token,piece in items:
            payload+=f"{token} {len(piece)} "+' '.join(map(str,piece))+"\n"
            reason=s.feed(token,piece)
            expected.append((reason,s.count))
    actual=[tuple(map(int,line.split())) for line in run(exe,payload).splitlines()]
    assert actual==expected
    assert expected[:5]==[(0,1),(0,2),(2,3),(2,3),(2,3)]
    assert expected[10:15]==[(0,1),(3,2),(3,2),(3,2),(3,2)]
