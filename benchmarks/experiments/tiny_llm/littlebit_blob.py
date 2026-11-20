"""LittleBit integer blob v1, all fields little-endian, no implicit alignment.

Header (40 bytes): magic 'LBT1', uint32 version=1, n_in, rank, n_out,
path_count (1 or 2), multiplier_shift=20, output_shift=12, float64 input_scale.
Each path: V^T packed [rank, ceil(n_in/8)] then U [n_out, ceil(rank/8)],
then signed int32 mg[n_in], m1[rank], m2[n_out]. Packed rows are LSB-first;
1 means +1, 0 means -1; high padding bits are zero. No trailing bytes.
Header input_scale is metadata only: q represents q*input_scale. Execution
uses integer multipliers only and returns Q12 values. Not a model.bin format.
"""
import importlib.util
from pathlib import Path
import struct

spec = importlib.util.spec_from_file_location("littlebit_int", Path(__file__).with_name("littlebit_int.py"))
integer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integer)
HEADER = struct.Struct("<4s7Id")


def _padding(data, rows, columns):
    stride = (columns + 7) // 8
    if columns % 8 and any(data[j * stride + stride - 1] >> (columns % 8) for j in range(rows)):
        raise ValueError("nonzero sign padding")


def dumps(layer):
    integer.validate_integer_layer(layer)
    n, r, o = (layer[key] for key in ("n_in", "rank", "n_out"))
    output = bytearray(HEADER.pack(b"LBT1", 1, n, r, o, len(layer["paths"]), 20, 12, layer["input_scale"]))
    for path in layer["paths"]:
        _padding(path["v"], r, n)
        _padding(path["u"], o, r)
        output.extend(path["v"])
        output.extend(path["u"])
        for key in ("mg", "m1", "m2"):
            output.extend(struct.pack(f"<{len(path[key])}i", *path[key]))
    return bytes(output)


def loads(blob):
    if len(blob) < HEADER.size:
        raise ValueError("truncated header")
    magic, version, n, r, o, count, shift, qf, scale = HEADER.unpack_from(blob)
    if (magic, version, shift, qf) != (b"LBT1", 1, 20, 12):
        raise ValueError("unsupported blob header")
    if not 1 <= r <= min(n, o) or count not in (1, 2):
        raise ValueError("invalid dimensions or path count")
    nv, nu = r * ((n + 7) // 8), o * ((r + 7) // 8)
    expected = HEADER.size + count * (nv + nu + 4 * (n + r + o))
    if len(blob) != expected:
        raise ValueError("blob size mismatch")
    result = {"n_in": n, "rank": r, "n_out": o, "input_scale": scale, "paths": []}
    offset = HEADER.size
    for _ in range(count):
        path = {"v": bytes(blob[offset:offset + nv]), "u": bytes(blob[offset + nv:offset + nv + nu])}
        offset += nv + nu
        _padding(path["v"], r, n)
        _padding(path["u"], o, r)
        for key, length in (("mg", n), ("m1", r), ("m2", o)):
            path[key] = list(struct.unpack_from(f"<{length}i", blob, offset))
            offset += 4 * length
        result["paths"].append(path)
    integer.validate_integer_layer(result)
    return result
