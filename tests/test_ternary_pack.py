import importlib.util
import itertools
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/ternary_pack.py"
spec = importlib.util.spec_from_file_location("ternary_pack", PATH)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


@pytest.mark.parametrize("format,group", [("2bit", 4), ("base3", 5)])
def test_exhaustive_single_byte_and_tails(format, group):
    for count in range(group + 1):
        for values in itertools.product((-1, 0, 1), repeat=count):
            data = codec.pack(values, format)
            assert len(data) == codec.payload_bytes(count, format)
            assert codec.unpack(data, count, format) == list(values)
    for count in (7, 17, 33, 256):
        values = [(-1, 1, 0, 1, -1)[i % 5] for i in range(count)]
        assert codec.unpack(codec.pack(values, format), count, format) == values


def test_known_bytes_and_accounting():
    assert codec.pack([-1, 0, 1, -1], "2bit") == bytes([36])
    assert codec.pack([-1, 0, 1, -1, 1]) == bytes([183])
    assert codec.pack([1] * 5) == bytes([242])
    assert codec.bytes_per_weight(100, "base3") == .2
    assert codec.bytes_per_weight(100, "2bit") == .25
    assert codec.bytes_per_weight(7, "base3", 4) == 6 / 7
    assert codec.bytes_per_weight(0) == 0


@pytest.mark.parametrize("data,count,format", [(b"\xff", 5, "base3"),
    (b"\x03", 4, "2bit"), (b"\x04", 1, "2bit"), (b"\x03", 1, "base3"),
    (b"", 1, "base3"), (b"\x00\x00", 5, "base3")])
def test_reject_malformed(data, count, format):
    with pytest.raises(ValueError):
        codec.unpack(data, count, format)


def test_reject_invalid_arguments():
    for values in ([2], [-2]):
        with pytest.raises(ValueError):
            codec.pack(values)
    with pytest.raises(TypeError):
        codec.pack([.5])
    for count, format in [(-1, "base3"), (1, "unknown")]:
        with pytest.raises(ValueError):
            codec.payload_bytes(count, format)
