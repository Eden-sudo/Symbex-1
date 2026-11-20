"""Export a restricted identity-normalized SentencePiece BPE table for ASCII prompts."""
import argparse
import json
from pathlib import Path
import struct


def fields(data):
    """Read protobuf wire fields without requiring the optional protobuf package."""
    pos = 0

    def varint():
        nonlocal pos
        value = 0
        for shift in range(0, 70, 7):
            if pos >= len(data):
                raise ValueError("truncated model")
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("invalid model varint")

    while pos < len(data):
        tag = varint()
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError("invalid model field")
        if wire == 0:
            value = varint()
        elif wire in (1, 2, 5):
            size = varint() if wire == 2 else (8 if wire == 1 else 4)
            if size > len(data) - pos:
                raise ValueError("truncated model field")
            value = data[pos:pos + size]
            pos += size
        else:
            raise ValueError("unsupported model wire type")
        yield number, value


def export(model, destination):
    top = list(fields(Path(model).read_bytes()))
    sections = dict(top)
    t, n = dict(fields(sections.get(2, b""))), dict(fields(sections.get(3, b"")))
    if (t.get(3, 1) != 2 or not t.get(35, 0) or t.get(24, 0)
            or n.get(1) != b"identity" or n.get(2, b"") or n.get(3, 1)
            or n.get(4, 1) or not n.get(5, 1)):
        raise ValueError("encoder requires identity BPE, byte fallback and unchanged whitespace")
    entries = []
    for token_id, blob in enumerate(value for number, value in top if number == 1):
        piece = dict(fields(blob))
        text = piece[1].decode("utf-8")
        score = struct.unpack("<f", piece.get(2, b"\0" * 4))[0]
        kind = piece.get(3, 1)
        if kind not in (1, 4, 6):
            continue
        if kind == 6:
            text = chr(int(text[3:5], 16))
            if not 32 <= ord(text) <= 126:
                continue
            kind = 2
        else:
            text = text.replace("▁", " ")
            if any(not 32 <= ord(c) <= 126 for c in text):
                continue  # targets may be Unicode; the prompt domain is ASCII only
            kind = int(kind == 4)
        score_literal = format(score, ".9g")
        if "." not in score_literal and "e" not in score_literal:
            score_literal += ".0"
        entries.append(f"    {{{json.dumps(text)}, {token_id}, {score_literal}f, {kind}}},")
    header = ('/* Generated from SentencePiece; ASCII prompt domain only. */\n'
              '#include "encoder.h"\nstatic const sensor_piece sensor_pieces[] = {\n'
              + "\n".join(entries) + '\n};\n'
              'const sensor_vocab sensor_exported_vocab = {sensor_pieces, sizeof(sensor_pieces) / sizeof(sensor_pieces[0])};\n')
    with Path(destination).open("x", encoding="ascii") as stream:
        stream.write(header)
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
