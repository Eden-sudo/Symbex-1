import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text/tokenizer.py"


def load():
    spec = importlib.util.spec_from_file_location("sensor_tokenizer_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_numeric_value_and_field_format(tmp_path):
    t = load()
    sp = t.train_tokenizer(tmp_path / "tiny", rows=82)
    assert 300 <= sp.vocab_size() <= 512
    for source, fields in t.prompt.FIELDS.items():
        for label in t.dataset.load_source_classes()[source]:
            event = {"source": source, "class": label, **{f[0]: f[2] for f in fields}}
            for name, _, low, high in fields:
                for value in range(low, high + 1):
                    event[name] = value
                    text = t.prompt.serialize(event)
                    ids = sp.encode(text)
                    assert sp.unk_id() not in ids
                    assert sp.decode(ids) == text
    for text in ("0 1 9 10 99 100 120 180", "  007 -1 +2 1.50  ",
                 "unknown unicode: °C", t.prompt.START, t.prompt.ANSWER, t.prompt.END):
        assert sp.decode(sp.encode(text)) == text
    for token in (t.prompt.START, t.prompt.ANSWER, t.prompt.END):
        assert len(sp.encode(token)) == 1
    # Same generated corpus and training order produce the same pieces/encodings.
    other = t.train_tokenizer(tmp_path / "again", rows=82)
    assert [sp.id_to_piece(i) for i in range(sp.vocab_size())] == [
        other.id_to_piece(i) for i in range(other.vocab_size())]
    corpus = (tmp_path / "tiny/corpus.txt").read_text()
    assert "initially" not in corpus  # held-out ECG template c
    assert all(sp.decode(sp.encode(line)) == line for line in corpus.splitlines())


def test_bad_settings_and_existing_output(tmp_path):
    t = load()
    with pytest.raises(ValueError):
        t.train_tokenizer(tmp_path / "bad", vocab_size=100)
    with pytest.raises(FileExistsError):
        t.train_tokenizer(tmp_path, rows=41)
