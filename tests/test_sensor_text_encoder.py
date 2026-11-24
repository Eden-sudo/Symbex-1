import ctypes as c
import importlib.util
from pathlib import Path
import random
import subprocess

import pytest

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Vocab(c.Structure):
    _fields_ = [("pieces", c.c_void_p), ("count", c.c_size_t)]


@pytest.mark.parametrize("environment", [False, True])
def test_clang_ids_match_sentencepiece(environment, tmp_path):
    t, exporter, pre = load("tokenizer.py"), load("export_encoder.py"), load("pretokenize.py")
    sp = t.train_tokenizer(tmp_path / "tok", rows=48, environment=environment)
    count = exporter.export(tmp_path / "tok/tokenizer.model", tmp_path / "vocab.c")
    assert 0 < count <= sp.vocab_size()
    library = tmp_path / "encoder.so"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-shared", "-fPIC", "-I", str(DIR), str(DIR / "encoder.c"),
                    str(tmp_path / "vocab.c"), "-o", str(library)], check=True, capture_output=True, timeout=10)
    lib = c.CDLL(str(library))
    vocab = Vocab.in_dll(lib, "sensor_exported_vocab")
    fn = lib.sensor_encode
    fn.argtypes = [c.POINTER(Vocab), c.c_char_p, c.POINTER(c.c_int), c.c_size_t]
    fn.restype = c.c_int
    b = t.prompt.bilingual
    events = [dict(r, language=language) for r in b.environment.generate_records(128, 314)
              for language in ("en", "es")]
    rng = random.Random(9)
    for source, labels in t.dataset.load_source_classes().items():
        events.extend(t.dataset.make_event(source, rng.choice(labels), rng) for _ in range(32))
    # Test both formats even with a tokenizer trained on only one domain.
    texts = [t.prompt.serialize(event) for event in events]
    texts += ["", "  007 -40 +85 100000  ", "<sensor_v2>|<text>", "~@#$%^&*()"]
    for text in texts:
        expected = sp.encode(text)
        output = (c.c_int * (len(expected) + 2))(*([-777] * (len(expected) + 2)))
        n = fn(c.byref(vocab), text.encode(), output, len(expected))
        assert n == len(expected), text
        assert list(output)[:n] == expected, text
        assert output[n] == -777
        if n:
            output[0] = -777
            assert fn(c.byref(vocab), text.encode(), output, n - 1) == -1
            assert output[0] == -777
    for bad in (b"\n", "á".encode(), b"x" * 4097):
        assert fn(c.byref(vocab), bad, output, len(output)) == -1
    row = {"event": events[0], "text": b.render(events[0])}
    ids = pre.encode_row(row, sp)
    assert sp.decode(ids[1:-1]) == t.prompt.example(row["event"], row["text"])


def test_export_rejects_incompatible_normalization(tmp_path):
    t, exporter = load("tokenizer.py"), load("export_encoder.py")
    t.train_tokenizer(tmp_path / "tok", rows=41)
    bad = tmp_path / "bad.model"
    original = (tmp_path / "tok/tokenizer.model").read_bytes()
    # Append a replacement NormalizerSpec: name=identity, add_dummy_prefix=true.
    normalizer = b"\x0a\x08identity\x18\x01"
    bad.write_bytes(original + b"\x1a" + bytes([len(normalizer)]) + normalizer)
    with pytest.raises(ValueError, match="identity BPE"):
        exporter.export(bad, tmp_path / "bad.c")
    assert not (tmp_path / "bad.c").exists()
