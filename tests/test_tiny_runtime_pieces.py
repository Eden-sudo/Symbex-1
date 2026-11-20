import struct
import subprocess
import pytest
from tiny_runtime_support import compile_c, module


class SyntheticSP:
    pieces=["<unk>","<s>","</s>","▁Hello","▁","<0xC3>","<0xA9>","!","<0x00>"]
    def get_piece_size(self): return len(self.pieces)
    def id_to_piece(self,t): return self.pieces[t]
    def is_control(self,t): return t in (1,2)
    def is_unknown(self,t): return t==0
    def is_byte(self,t): return t in (5,6,8)


def test_pieces(tmp_path):
    py=module("chat_pieces")
    blob=py.export(SyntheticSP())
    assert py.pack(py.unpack(blob))==blob
    path=tmp_path/"pieces.bin"; path.write_bytes(blob)
    exe=compile_c(tmp_path,r'''
#include <stdio.h>
#include "chat_pieces.h"
int main(int argc,char **argv) {
    unsigned char blob[1024]; ChatDecoder d;
    if(argc!=2) return 1;
    FILE *f=fopen(argv[1],"rb"); if(!f) return 2;
    size_t n=fread(blob,1,sizeof(blob),f); fclose(f);
    if(chat_decoder_init(&d,blob,n)) return 3;
    int token;
    while(scanf("%d",&token)==1) {
        const unsigned char *p; size_t size;
        if(chat_decode(&d,token,&p,&size)) return 4;
        if(fwrite(p,1,size,stdout)!=size) return 5;
    }
    return 0;
}
''',["chat_pieces.c"])
    for ids,expected in [([1,3,4,5,6,7,2],"Hello é!".encode()),
                         ([1,4,4,3],b"  Hello"),([1,8,2],b"\0"),([1,2],b"")]:
        decoder=py.Decoder(blob)
        assert b''.join(decoder.feed(t) for t in ids)==expected
        result=subprocess.run([str(exe),str(path)],input=' '.join(map(str,ids)).encode(),
                              capture_output=True,timeout=5,check=True)
        assert result.stdout==expected
    corrupt=[b"",blob[:10],b"BAD!"+blob[4:],blob+b"x",
             blob[:8]+struct.pack('<I',1)+blob[12:]]
    for bad in corrupt:
        with pytest.raises(ValueError): py.unpack(bad)
        path.write_bytes(bad)
        assert subprocess.run([str(exe),str(path)],capture_output=True,timeout=5).returncode==3
