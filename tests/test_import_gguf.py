import struct

import numpy as np
import pytest

from tiny_import_helpers import load

reader = load("importers/gguf")


def string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, metadata=(), alignment=32):
    kv = [("general.alignment", 4, struct.pack("<I", alignment)), *metadata]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    header += b"".join(string(k) + struct.pack("<I", t) + value for k, t, value in kv)
    payload = bytearray()
    for name, dims, kind, data in tensors:
        offset = len(payload)
        header += string(name) + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", kind, offset)
        payload.extend(data)
        payload.extend(b"\x00" * (-len(payload) % alignment))
    path.write_bytes(header + b"\x00" * (-len(header) % alignment) + payload)


@pytest.mark.parametrize("alignment", [32, 64])
def test_tensors_and_all_metadata_types(tmp_path, alignment):
    path = tmp_path / "toy.gguf"
    formats = [(0, "B", 255), (1, "b", -8), (2, "H", 65535), (3, "h", -123),
               (4, "I", 123456), (5, "i", -123456), (6, "f", 1.25), (7, "?", True),
               (10, "Q", 2**40), (11, "q", -2**40), (12, "d", -3.125)]
    meta = [(f"key{kind}", kind, struct.pack("<" + fmt, v)) for kind, fmt, v in formats]
    meta += [("text", 8, string("llama")),
             ("vocab", 9, struct.pack("<IQ", 8, 2) + string("a") + string("▁b")),
             ("ints", 9, struct.pack("<IQii", 5, 2, -1, 2)),
             ("empty", 9, struct.pack("<IQ", 6, 0))]
    a = np.arange(6, dtype="<f4").reshape(2, 3) - 2
    b = np.array([.5, -2., 4., 0.], dtype="<f2")
    q = np.arange(-32, 32, dtype=np.int8).reshape(2, 32)
    packed = struct.pack("<e", .5) + q[0].tobytes() + struct.pack("<e", -.25) + q[1].tobytes()
    write_gguf(path, [("a", (3, 2), 0, a.tobytes()), ("b", (4,), 1, b.tobytes()),
                      ("q", (32, 2), 8, packed)], meta, alignment)
    result = reader.read_gguf(path)
    for kind, _, v in formats:
        assert result["metadata"][f"key{kind}"] == v
    assert result["metadata"]["vocab"] == ["a", "▁b"]
    assert result["metadata"]["ints"] == [-1, 2]
    assert result["metadata"]["empty"] == []
    assert result["metadata"]["text"] == "llama"
    np.testing.assert_array_equal(result["tensors"]["a"], a)
    np.testing.assert_array_equal(result["tensors"]["b"], b)
    np.testing.assert_array_equal(result["tensors"]["q"], q * np.array([.5, -.25])[:, None])
    assert result["tensor_infos"]["a"]["dimensions"] == (3, 2)


@pytest.mark.parametrize("kind,name", [(2, "Q4_0"), (19, "IQ1_S"), (29, "IQ1_M"),
                                         (34, "TQ1_0"), (35, "TQ2_0"), (41, "Q1_0"), (999, "UNKNOWN_999")])
def test_unsupported_named(tmp_path, kind, name):
    path = tmp_path / "unsupported.gguf"
    write_gguf(path, [("weight", (32,), kind, b"x" * 128)])
    with pytest.raises(ValueError, match=name):
        reader.read_gguf(path)


@pytest.mark.parametrize("case,match", [("version", "version 3"), ("truncated", "truncated"),
    ("rank", "rank"), ("qwidth", "row width"), ("duplicate", "duplicate"),
    ("metadata", "metadata type"), ("alignment", "alignment")])
def test_bad_files(tmp_path, case, match):
    path = tmp_path / "bad.gguf"
    tensors = [("a", (32,), 0, b"\x00" * 128)]
    meta = []
    if case == "rank":
        tensors = [("a", (), 0, b"")]
    if case == "qwidth":
        tensors = [("a", (31,), 8, b"\x00" * 34)]
    if case == "duplicate":
        tensors *= 2
    if case == "metadata":
        meta = [("bad", 99, b"")]
    write_gguf(path, tensors, meta)
    data = bytearray(path.read_bytes())
    if case == "version":
        data[4:8] = struct.pack("<I", 2)
    if case == "truncated":
        data = data[:-1]
    if case == "alignment":
        # alignment payload follows header, key string and uint32 metadata type.
        offset = 24 + 8 + len("general.alignment") + 4
        data[offset:offset + 4] = struct.pack("<I", 3)
    path.write_bytes(data)
    with pytest.raises(ValueError, match=match):
        reader.read_gguf(path)
