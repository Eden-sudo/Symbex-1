import ctypes as c
import importlib.util
from pathlib import Path
import random
import subprocess

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Record(c.Structure):
    _fields_ = [("source", c.c_int), ("class_name", c.c_char_p),
                ("duration_seconds", c.c_int), ("heart_rate_before", c.c_int),
                ("heart_rate_after", c.c_int), ("count", c.c_int)]


def test_clang_python_byte_equality_and_buffers(tmp_path):
    lib = tmp_path / "prompt.so"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-shared", "-fPIC", str(DIR / "prompt.c"), "-o", str(lib)],
                   check=True, capture_output=True, timeout=10)
    fn = c.CDLL(str(lib)).sensor_prompt
    fn.argtypes = [c.POINTER(c.c_char), c.c_size_t, c.POINTER(Record)]
    fn.restype = c.c_int
    p, d = load("prompt.py"), load("make_dataset.py")
    rng = random.Random(412)
    events = []
    for index, (source, labels) in enumerate(d.load_source_classes().items()):
        for _ in range(200):
            events.append((index, d.make_event(source, rng.choice(labels), rng)))
        for extreme in (2, 3):
            events.append((index, {"source": source, "class": "A" * 64,
                                  **{f[0]: f[extreme] for f in p.FIELDS[source]}}))
    for index, event in events:
        record = Record(index, event["class"].encode(), *(event.get(k, 0) for k in
                        ("duration_seconds", "heart_rate_before", "heart_rate_after", "count")))
        expected = p.serialize(event).encode()
        buf = c.create_string_buffer(len(expected) + 1)
        assert fn(buf, len(buf), c.byref(record)) == len(expected)
        assert buf.raw == expected + b"\0"
        # Red zone must survive even when output is truncated.
        small = c.create_string_buffer(b"x" * (len(expected) + 2))
        assert fn(small, len(expected), c.byref(record)) == -1
        assert small[0] == b"\0" and small[len(expected)] == b"x"
    for label in (b"", b"1bad", b"bad|class", b"A" * 65, b"\xff", None):
        record = Record(0, label, 1, 0, 0, 0)
        buf = c.create_string_buffer(256)
        assert fn(buf, len(buf), c.byref(record)) == -1
    for record in (Record(3, b"N", 1, 40, 40, 0), Record(0, b"N", 0, 0, 0, 0),
                   Record(1, b"N", 0, 39, 40, 0), Record(1, b"N", 0, 40, 181, 0),
                   Record(2, b"N", 0, 0, 0, 6)):
        assert fn(buf, len(buf), c.byref(record)) == -1
    assert fn(buf, len(buf), None) == -1
    assert fn(None, 0, None) == -1
    buf[0] = b"x"
    assert fn(buf, 0, None) == -1 and buf[0] == b"x"
