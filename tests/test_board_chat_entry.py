import subprocess
from tiny_import_helpers import TINY


def test_host_entry(tmp_path):
    driver = tmp_path / "driver.c"
    driver.write_text(r'''
#include "board_chat.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
typedef struct { int ids[256], n, resets, fail; char text[1024]; size_t used; } State;
static int reset(void *p) { State *s=p; s->n=0; ++s->resets; return 0; }
static int forward(void *p, int token, int64_t *z) {
    State *s=p; s->ids[s->n++]=token;
    for (int i=0;i<8;++i) z[i]=0;
    z[6]=9; return s->fail ? -1 : 0;
}
static int write_bytes(void *p, const char *b, size_t n) {
    State *s=p; assert(s->used+n<sizeof s->text);
    memcpy(s->text+s->used,b,n); s->used+=n; s->text[s->used]=0; return 0;
}
int main(void) {
    int sm=3, um=4, am=5, content=7;
    ChatIds markers[3]={{&sm,1},{&um,1},{&am,1}};
    ChatTurn turns[3]={{0,{&content,1}},{1,{&content,1}},{1,{&content,1}}};
    ChatOptions o={8,10,0,0,65536,1,0};
    int64_t logits[8]; int scratch[8]; size_t dropped=99; State s={0};
    assert(board_chat_turn(markers,turns,3,&o,2,42,reset,forward,write_bytes,
                           &s,logits,scratch,&dropped)==3);
    assert(dropped==1 && s.resets==1 && s.n==9 && o.seed==1);
    int expected[]={1,3,7,2,4,7,2,5,6};
    assert(memcmp(s.ids,expected,sizeof expected)==0);
    assert(strcmp(s.text,"CHAT1 BEGIN 42\nCHAT1 TOKEN 0 6\nCHAT1 TOKEN 1 6\nCHAT1 END 3 2\n")==0);
    puts(s.text);
    s.used=0; s.fail=1;
    assert(board_chat_turn(markers,turns,3,&o,2,42,reset,forward,write_bytes,
                           &s,logits,scratch,&dropped)==-1);
    assert(strstr(s.text,"END")==NULL);
    int before=s.resets; turns[2].role=0;
    assert(board_chat_turn(markers,turns,3,&o,2,42,reset,forward,write_bytes,
                           &s,logits,scratch,&dropped)==-1);
    assert(s.resets==before);
    return 0;
}
''')
    exe = tmp_path / "driver"
    sources = ["board_chat", "chat_format", "chat_memory", "chat_loop",
               "chat_sample", "chat_penalty", "chat_stop"]
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-I", str(TINY / "c"), str(driver),
                    *(str(TINY / "c" / (s + ".c")) for s in sources), "-o", str(exe)],
                   check=True, capture_output=True, timeout=15)
    result = subprocess.run([str(exe)], check=True, capture_output=True, timeout=5)
    assert b"CHAT1 END 3 2" in result.stdout
