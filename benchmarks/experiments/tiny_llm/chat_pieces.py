"""Export SentencePiece piece bytes: python chat_pieces.py tokenizer.model pieces.bin.

Format little-endian: 'TCP1', uint32 vocab, uint32 offsets[vocab+1], payload.
Offsets relative to payload; control pieces empty, byte fallback raw byte,
U+2581 becomes ASCII space. Streaming drops one leading dummy-prefix space.
UNK uses SentencePiece's default unknown surface U+2047 surrounded by spaces.
This targets prepare.py's identity-normalized BPE, not arbitrary SP normalizers.
"""
import struct


def pack(pieces):
    if not pieces or len(pieces)>65536 or any(type(p) is not bytes for p in pieces):
        raise ValueError("expected 1..65536 byte pieces")
    offsets=[0]
    for p in pieces:
        offsets.append(offsets[-1]+len(p))
    if offsets[-1]>0xffffffff:
        raise ValueError("piece payload too large")
    return b"TCP1"+struct.pack("<I",len(pieces))+struct.pack(f"<{len(offsets)}I",*offsets)+b"".join(pieces)


def unpack(blob):
    if len(blob)<12 or blob[:4]!=b"TCP1":
        raise ValueError("invalid table header")
    n=struct.unpack_from("<I",blob,4)[0]
    start=8+4*(n+1)
    if not 1<=n<=65536 or start>len(blob):
        raise ValueError("invalid table size")
    offsets=struct.unpack_from(f"<{n+1}I",blob,8)
    if offsets[0] or offsets[-1]!=len(blob)-start or any(a>b for a,b in zip(offsets,offsets[1:])):
        raise ValueError("invalid offsets")
    return [blob[start+a:start+b] for a,b in zip(offsets,offsets[1:])]


def export(sp):
    pieces=[]
    for token in range(sp.get_piece_size()):
        p=sp.id_to_piece(token)
        if sp.is_control(token):
            data=b""
        elif sp.is_unknown(token):
            data=" ⁇ ".encode()
        elif sp.is_byte(token):
            if len(p)!=6 or not p.startswith("<0x") or p[-1]!=">":
                raise ValueError("invalid byte piece")
            data=bytes([int(p[3:5],16)])
        else:
            data=p.replace("▁"," ").encode("utf-8")
        pieces.append(data)
    return pack(pieces)


class Decoder:
    def __init__(self, blob):
        self.pieces=unpack(blob)
        self.started=False

    def feed(self, token):
        if type(token) is not int or not 0<=token<len(self.pieces):
            raise ValueError("invalid token")
        data=self.pieces[token]
        if data and not self.started:
            self.started=True
            if data.startswith(b" "):
                data=data[1:]
        return data


if __name__=="__main__":
    import argparse
    from pathlib import Path
    import sentencepiece as spm
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output")
    args=parser.parse_args()
    Path(args.output).write_bytes(export(spm.SentencePieceProcessor(model_file=args.model)))
