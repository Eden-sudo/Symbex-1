import pytest
from tiny_runtime_support import compile_c, module, run


class Vocab:
    # Synthetic byte vocabulary: all IDs already exist, role markers multi-token.
    def encode(self, text): return [b+3 for b in text.encode()]
    def decode(self, ids): return bytes(t-3 for t in ids).decode()
    def unk_id(self): return 0
    def bos_id(self): return 1
    def eos_id(self): return 2


def test_format(tmp_path):
    py=module("chat_format"); sp=Vocab()
    turns=[("system","Be brief."),("user","héllo"),("assistant","Hi!"),("user","")]
    markers=[sp.encode(py.MARKERS[r]) for r in ("system","user","assistant")]
    encoded=[sp.encode(text) for _,text in turns]
    arrays='\n'.join(f'int m{i}[]={{'+','.join(map(str,ids))+'};' for i,ids in enumerate(markers))
    arrays+='\n'+'\n'.join(f'int t{i}[]={{'+','.join(map(str,ids or [0]))+'};' for i,ids in enumerate(encoded))
    specs=','.join('{'+f'{i%3},{{t{i},{len(ids)}}}'+'}' for i,ids in enumerate(encoded))
    # Last turn is user, not system.
    specs=specs.rsplit('{0,',1)[0]+'{1,'+specs.rsplit('{0,',1)[1]
    src=r'''
#include <stdio.h>
#include "chat_format.h"
int main(void) {
ARRAYS
    ChatIds markers[3]={{m0,7},{m1,5},{m2,10}};
    ChatTurn turns[]={SPECS}; int out[256];
    out[0]=999;
    if(chat_build(markers,turns,4,259,out,1)!=-1 || out[0]!=999) return 1;
    int n=chat_build(markers,turns,4,259,out,256);
    if(n<0) return 2;
    for(int i=0;i<n;++i) printf("%d ",out[i]);
    turns[3].role=3;
    if(chat_build(markers,turns,4,259,out,256)!=-1) return 3;
    return 0;
}
'''.replace('ARRAYS',arrays).replace('SPECS',specs)
    exe=compile_c(tmp_path,src,["chat_format.c"])
    assert list(map(int,run(exe).split()))==py.build_chat(sp,turns)
    for role,text in turns:
        assert py.decode_turn(sp,role,py.encode_turn(sp,role,text))==text
    with pytest.raises(ValueError): py.decode_turn(sp,"user",[1,2])
    with pytest.raises(ValueError): py.encode_turn(sp,"tool","x")
