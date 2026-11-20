"""Minimal little-endian GGUF v3 reader; no architecture mapping or downloads.

GGUF dimensions are fastest-axis first. Returned NumPy shapes reverse them.
Q8_0 stores blocks of 32 int8 values preceded by a little-endian fp16 scale.
"""
import math
import struct
from pathlib import Path

import numpy as np


# ggml_type IDs, including removed slots. Unknown vendor IDs remain numeric;
# a model marketed as "1-bit" does not establish its on-disk encoding.
TYPE_NAMES = dict(enumerate([
    "F32", "F16", "Q4_0", "Q4_1", "Q4_2 (removed)", "Q4_3 (removed)",
    "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
    "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
    "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64",
    "IQ1_M", "BF16", "Q4_0_4_4 (removed)", "Q4_0_4_8 (removed)",
    "Q4_0_8_8 (removed)", "TQ1_0", "TQ2_0", "IQ4_NL_4_4 (removed)",
    "IQ4_NL_4_8 (removed)", "IQ4_NL_8_8 (removed)", "MXFP4", "NVFP4", "Q1_0",
]))
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf(path):
    """Return metadata, tensor infos and decoded tensors; reject malformed data.

    Only supported types are decoded. Offsets must be aligned, nonoverlapping
    and in-file. Length checks precede reads/allocations for untrusted inputs.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        def read(n):
            if n < 0 or n > size - f.tell():
                raise ValueError("truncated GGUF")
            data = f.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF")
            return data

        def unpack(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

        def string():
            try:
                return read(unpack("Q")).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("invalid GGUF UTF-8 string") from exc

        def value(kind, nested=False):
            if kind in SCALARS:
                if kind == 7:
                    b = unpack("B")
                    if b not in (0, 1):
                        raise ValueError("invalid GGUF boolean")
                    return bool(b)
                return unpack(SCALARS[kind])
            if kind == 8:
                return string()
            if kind == 9 and not nested:
                element, count = unpack("I"), unpack("Q")
                if element == 9:
                    raise ValueError("nested GGUF arrays are not supported")
                if count > size - f.tell():
                    raise ValueError("truncated GGUF array")
                if element not in SCALARS and element != 8:
                    raise ValueError(f"unsupported GGUF metadata type {element}")
                return [value(element, True) for _ in range(count)]
            raise ValueError(f"unsupported GGUF metadata type {kind}")

        if read(4) != b"GGUF" or unpack("I") != 3:
            raise ValueError("expected little-endian GGUF version 3")
        count, kv_count = unpack("Q"), unpack("Q")
        if count > size // 24 or kv_count > size // 12:
            raise ValueError("impossible GGUF entry count")
        metadata, metadata_types = {}, {}
        for _ in range(kv_count):
            key, kind = string(), unpack("I")
            if key in metadata:
                raise ValueError(f"duplicate GGUF metadata key {key}")
            metadata[key], metadata_types[key] = value(kind), kind
        alignment = metadata.get("general.alignment", 32)
        if ("general.alignment" in metadata and metadata_types["general.alignment"] != 4) or \
                not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a positive power-of-two UINT32")
        infos = {}
        for _ in range(count):
            name, rank = string(), unpack("I")
            if name in infos or not 1 <= rank <= 4:
                raise ValueError("duplicate tensor name or invalid GGUF rank")
            dims = tuple(unpack("Q") for _ in range(rank))
            kind, offset = unpack("I"), unpack("Q")
            label = TYPE_NAMES.get(kind, f"UNKNOWN_{kind}")
            if kind not in (0, 1, 8):
                raise ValueError(f"unsupported GGUF tensor type {label} ({kind}) for {name}; "
                                 "only F32, F16 and Q8_0 are decoded; ternary/1-bit needs its own decoder")
            if not all(dims) or offset % alignment:
                raise ValueError("invalid GGUF dimensions or unaligned tensor offset")
            if kind == 8 and dims[0] % 32:
                raise ValueError("Q8_0 row width must be divisible by 32")
            n = math.prod(dims)
            nbytes = n * (4 if kind == 0 else 2) if kind != 8 else n // 32 * 34
            infos[name] = dict(dimensions=dims, shape=dims[::-1], type=label,
                               type_id=kind, offset=offset, nbytes=nbytes)
        start = (f.tell() + alignment - 1) // alignment * alignment
        end = start
        for info in sorted(infos.values(), key=lambda i: i["offset"]):
            pos = start + info["offset"]
            if pos < end or pos + info["nbytes"] > size:
                raise ValueError("overlapping or truncated GGUF tensor data")
            end = pos + info["nbytes"]
        tensors = {}
        for name, info in infos.items():
            f.seek(start + info["offset"])
            raw = read(info["nbytes"])
            if info["type_id"] == 8:
                blocks = np.frombuffer(raw, dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]))
                tensor = blocks["q"].astype(np.float32) * blocks["d"].astype(np.float32)[:, None]
            else:
                tensor = np.frombuffer(raw, dtype="<f4" if info["type_id"] == 0 else "<f2").astype(np.float32)
            tensors[name] = tensor.reshape(info["shape"])
    return {"metadata": metadata, "tensor_infos": infos, "tensors": tensors}
