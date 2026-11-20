"""LBM1 projection container: 32-byte header, JSON TOC, 16-byte aligned payloads.

Header <4sIII16s>: magic, version, TOC byte count, entry count, reserved zeros.
TOC is SHA256-protected (32 bytes immediately after header); each entry records
name, kind, offset, size and SHA256. Factors embed LBT1. Binary payloads are
<II> (outputs, inputs), packed row signs, then float32 row scales. Auxiliary
model state is little-endian float32 with shape in TOC (including rope tables).
This is an archival Python format, not a runnable tinylm.c model.bin.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("littlebit_blob", Path(__file__).with_name("littlebit_blob.py"))
blob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blob)
HEADER = struct.Struct("<4sIII16s")
ALIGN = 16


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def dumps(model, *, input_bound=1.0):
    """Export every block projection, embedding/head weights, norms and buffers.

    Reject unsupported block projections instead of silently omitting them.
    Readers return named payloads, not an automatically reconstructed model.
    """
    entries, payloads, covered = [], [], set()
    def add(name, kind, payload, **metadata):
        entries.append(dict(name=name, kind=kind, size=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest(), **metadata))
        payloads.append(payload)
    for index, block in enumerate(model.blocks):
        for name in ("q", "k", "v", "o", "w1", "w3", "w2"):
            layer = getattr(block, name)
            key = f"blocks.{index}.{name}"
            if hasattr(layer, "paths") and hasattr(layer, "rank"):
                add(key, "factor", blob.dumps(blob.integer.integerize(layer, input_bound)))
            elif (isinstance(layer, torch.nn.Linear) and layer.bias is None
                  and getattr(layer, "kind", None) == "binary"
                  and getattr(layer, "acts", 8) == 8 and not getattr(layer, "had_blk", 0)):
                w = layer.weight.detach().float().cpu()
                if not torch.isfinite(w).all():
                    raise ValueError("nonfinite binary weights")
                signs = blob.integer._pack_rows(w.tolist())
                scales = w.abs().mean(1).clamp(min=1e-5).numpy().astype("<f4").tobytes()
                add(key, "binary", struct.pack("<II", *w.shape) + signs + scales)
            else:
                raise ValueError(f"unsupported projection: {key}")
            covered.add(key + ".")
    state = dict(model.state_dict())
    state.update(dict(model.named_buffers()))
    for name, value in state.items():
        if any(name.startswith(prefix) for prefix in covered):
            continue
        array = value.detach().float().cpu().numpy().astype("<f4")
        if not np.isfinite(array).all():
            raise ValueError("nonfinite auxiliary state")
        add(name, "float32", array.tobytes(), shape=list(array.shape))
    # Offsets depend on decimal widths in the JSON TOC; iterate to a fixed point.
    toc = b""
    while True:
        offset = _align(HEADER.size + 32 + len(toc))
        for entry in entries:
            entry["offset"] = offset
            offset = _align(offset + entry["size"])
        updated = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        if updated == toc:
            break
        toc = updated
    output = bytearray(HEADER.pack(b"LBM1", 1, len(toc), len(entries), bytes(16)))
    output.extend(hashlib.sha256(toc).digest())
    output.extend(toc)
    for entry, payload in zip(entries, payloads):
        output.extend(bytes(entry["offset"] - len(output)))
        output.extend(payload)
    return bytes(output)


def loads(data):
    """Validate integrity, bounds, alignment, unique names and typed payloads."""
    if len(data) < HEADER.size + 32:
        raise ValueError("truncated container")
    magic, version, size, count, reserved = HEADER.unpack_from(data)
    if (magic, version, reserved) != (b"LBM1", 1, bytes(16)):
        raise ValueError("unsupported container header")
    end = HEADER.size + 32 + size
    toc = data[HEADER.size + 32:end]
    if end > len(data) or hashlib.sha256(toc).digest() != data[HEADER.size:HEADER.size + 32]:
        raise ValueError("TOC checksum mismatch")
    try:
        entries = json.loads(toc)
        if not isinstance(entries, list) or len(entries) != count:
            raise ValueError("entry count mismatch")
        result, previous = {}, end
        for e in entries:
            name, kind, offset, length = (e[k] for k in ("name", "kind", "offset", "size"))
            if (not isinstance(name, str) or not name or name in result
                    or type(offset) is not int or type(length) is not int or length <= 0
                    or offset != _align(previous) or offset + length > len(data)
                    or any(data[previous:offset])):
                raise ValueError("invalid TOC entry")
            payload = data[offset:offset + length]
            if hashlib.sha256(payload).hexdigest() != e["sha256"]:
                raise ValueError("payload checksum mismatch")
            if kind == "factor":
                value = blob.loads(payload)
            elif kind == "binary":
                o, n = struct.unpack_from("<II", payload)
                signs_size = o * ((n + 7) // 8)
                if not n or not o or length != 8 + signs_size + 4 * o:
                    raise ValueError("invalid binary dimensions")
                signs = bytes(payload[8:8 + signs_size])
                blob._padding(signs, o, n)
                scales = np.frombuffer(payload[8 + signs_size:], dtype="<f4").copy()
                if not np.isfinite(scales).all() or (scales <= 0).any():
                    raise ValueError("invalid binary scales")
                value = dict(n_in=n, n_out=o, signs=signs, scales=scales)
            elif kind == "float32":
                shape = e["shape"]
                if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape):
                    raise ValueError("invalid state shape")
                import math
                if math.prod(shape) * 4 != length:
                    raise ValueError("invalid state size")
                value = np.frombuffer(payload, dtype="<f4").copy().reshape(shape)
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite state")
            else:
                raise ValueError("unknown layer kind")
            result[name] = {"kind": kind, "value": value}
            previous = offset + length
        if previous != len(data):
            raise ValueError("trailing container bytes")
        return result
    except (KeyError, TypeError, struct.error, UnicodeError) as exc:
        raise ValueError("malformed container") from exc


def export(model, path, **kwargs):
    Path(path).write_bytes(dumps(model, **kwargs))


def read(path):
    return loads(Path(path).read_bytes())
