import binascii
import struct
import subprocess
import pytest
from tiny_runtime_support import compile_c, module


def test_c_byte_exact_and_streaming(tmp_path):
    codec = module("chat_serial")
    exe = compile_c(tmp_path, r'''
#include <stdio.h>
#include "chat_serial.h"
int main(void) {
    unsigned kind, size, value; unsigned char p[512], out[CHAT_SERIAL_MAX_FRAME];
    while (scanf("%u%u", &kind, &size) == 2) {
        if (size > 512) return 1;
        for (unsigned i=0;i<size;++i) {
            if(scanf("%u", &value)!=1) return 2;
            p[i]=(unsigned char)value;
        }
        out[0]=99;
        if(chat_serial_encode(kind,p,size,out,1)!=0 || out[0]!=99) return 3;
        size_t n=chat_serial_encode(kind,p,size,out,sizeof(out));
        if(!n) return 4;
        if(fwrite(out,1,n,stdout)!=n) return 5;
    }
    return 0;
}
''', ["chat_serial.c"])
    events = [(1, struct.pack("<I", n)) for n in (0, 0x7e7d, 0x7fffffff)]
    events += [(2, bytes(range(256))*2), (2, b"\xc3"), (2, b"\xa9"), (3, b"\x01")]
    payload = "".join(f"{kind} {len(data)} " + " ".join(map(str, data)) + "\n" for kind, data in events)
    actual = subprocess.run([str(exe)], input=payload.encode(), capture_output=True, check=True, timeout=5).stdout
    expected = b"".join(codec.encode(*event) for event in events)
    assert actual == expected
    assert binascii.crc_hqx(b"123456789", 0xffff) == 0x29b1
    for chunk_size in (1, 2, 7, len(actual)):
        decoder, received = codec.Decoder(), []
        for start in range(0, len(actual), chunk_size):
            received.extend(decoder.feed(actual[start:start+chunk_size]))
        decoder.finish()
        assert received == events
    for reason in range(1, 6):
        frame = codec.encode(3, bytes([reason]))
        decoder = codec.Decoder()
        assert decoder.feed(frame) == [(3, bytes([reason]))]
        decoder.finish()
    # Rejected C inputs keep the output untouched (driver checks capacity first).
    for kind, data in [(0,b"x"),(1,b"x"),(1,b"\0\0\0\x80"),(2,b""),(3,b"\0")]:
        raw = f"{kind} {len(data)} " + " ".join(map(str, data)) + "\n"
        result = subprocess.run([str(exe)], input=raw.encode(), capture_output=True, timeout=5)
        assert result.returncode == 4


def test_corruption_truncation_and_bounds():
    codec = module("chat_serial")
    token = codec.encode(1, struct.pack("<I", 42))
    end = codec.encode(3, b"\x01")
    corrupt = bytearray(token); corrupt[5] ^= 1
    bad = [bytes(corrupt), b"x", b"\x7e\x7d\x7e", b"\x7e\x7d\x00",
           b"\x7e" + b"a"*519, end+token, b"\x7ex\x7e"]
    # CRC-valid malformed version, length, kind and payload.
    for body in (b"\x02\x03\x01\x00\x01", b"\x01\x03\x02\x00\x01",
                 b"\x01\x09\x01\x00\x01", b"\x01\x03\x01\x00\x00"):
        raw = body + struct.pack("<H", binascii.crc_hqx(body, 0xffff))
        raw = raw.replace(b"\x7d", b"\x7d\x5d").replace(b"\x7e", b"\x7d\x5e")
        bad.append(b"\x7e" + raw + b"\x7e")
    for wire in bad:
        decoder = codec.Decoder()
        with pytest.raises(ValueError): decoder.feed(wire)
        with pytest.raises(ValueError): decoder.feed(end)
    for wire in (b"", token, end[:-1], b"\x7e\x7d"):
        decoder = codec.Decoder(); decoder.feed(wire)
        with pytest.raises(ValueError): decoder.finish()
    for kind, data in [(0,b"x"),(1,b"x"),(1,b"\0\0\0\x80"),(2,b""),(2,b"a"*513),(3,b"\0")]:
        with pytest.raises(ValueError): codec.encode(kind, data)
