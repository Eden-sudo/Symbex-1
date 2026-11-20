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
                                         (41, "Q1_0"), (999, "UNKNOWN_999")])
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


def ternary_block(values, scale, kind):
    """Synthetic encoder from the block layout, independent of the decoder."""
    digits = [int(v) + 1 for v in values]
    payload = bytearray()
    if kind == 34:
        for start, width, planes in ((0, 32, 5), (160, 16, 5), (240, 4, 4)):
            for lane in range(width):
                number = sum(digits[start + plane * width + lane] * 3**(4 - plane)
                             for plane in range(planes))
                payload.append((number * 256 + 242) // 243)
    else:
        for start in (0, 128):
            for lane in range(32):
                payload.append(sum(digits[start + plane * 32 + lane] << (2 * plane)
                                   for plane in range(4)))
    return bytes(payload) + struct.pack("<e", scale)


@pytest.mark.parametrize("kind,size", [(34, 54), (35, 66)])
def test_ternary_blocks_and_row_order(tmp_path, kind, size):
    rng = np.random.default_rng(42)
    codes = rng.integers(-1, 2, (4, 256))
    codes[0] = -1
    codes[1] = 0
    # Distinct values across the 160/240/256 and row boundaries.
    codes[2, [0, 31, 32, 159, 160, 239, 240, 255]] = [1, 0, -1, 1, -1, 0, 1, -1]
    scales = [0.5, 2., -0.25, 0.]
    raw = b"".join(ternary_block(row, scale, kind) for row, scale in zip(codes, scales))
    assert len(raw) == size * 4
    path = tmp_path / "ternary.gguf"
    write_gguf(path, [("w", (512, 2), kind, raw)])
    result = reader.read_gguf(path)
    expected = (codes * np.array(scales)[:, None]).reshape(2, 512)
    assert result["tensors"]["w"].dtype == np.float32
    np.testing.assert_array_equal(result["tensors"]["w"], expected)
    assert result["tensor_infos"]["w"]["nbytes"] == size * 4


def test_tq1_all_five_trit_patterns():
    # Every base-3 five-digit combination, in every lane/plane.
    for number in range(243):
        digits = [(number // 3**(4 - j)) % 3 for j in range(5)]
        values = np.array([d - 1 for d in digits for _ in range(32)] +
                          [d - 1 for d in digits for _ in range(16)] +
                          [d - 1 for d in digits[:4] for _ in range(4)])
        raw = ternary_block(values, 1., 34)
        np.testing.assert_array_equal(reader.dequant_ternary(raw, 34)[0], values)


@pytest.mark.parametrize("kind,size", [(34, 54), (35, 66)])
def test_ternary_truncation_and_row_validation(tmp_path, kind, size):
    path = tmp_path / "bad-ternary.gguf"
    write_gguf(path, [("w", (128, 2), kind, bytes(size))])
    with pytest.raises(ValueError, match="row width"):
        reader.read_gguf(path)
    write_gguf(path, [("w", (256,), kind, bytes(size))])
    # Remove alignment padding as well as one actual block byte.
    path.write_bytes(path.read_bytes()[:-(32 - size % 32 + 1)])
    with pytest.raises(ValueError, match="truncated"):
        reader.read_gguf(path)
    with pytest.raises(ValueError, match="truncated"):
        reader.dequant_ternary(bytes(size - 1), kind)


def test_tq2_reserved_code_follows_ggml():
    raw = bytes([255] * 64) + struct.pack("<e", .5)
    np.testing.assert_array_equal(reader.dequant_ternary(raw, 35), np.ones((1, 256)))
