import importlib.util
from pathlib import Path

import numpy as np
import pytest

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_real_load_split_layout(tmp_path, monkeypatch):
    t, p, train = load("tokenizer.py"), load("pretokenize.py"), load("../tiny_llm/train.py")
    sp = t.train_tokenizer(tmp_path / "tok", rows=41)
    splits, meta = t.dataset.generate_dataset([41, 2, 0], 7)
    t.dataset.write_dataset(tmp_path / "json", splits, meta)
    counts = p.prepare(tmp_path / "json", tmp_path / "tok/tokenizer.model", tmp_path / "bin")
    monkeypatch.setattr(train, "DATA", str(tmp_path / "bin"))
    for split, rows in splits.items():
        expected = [v for row in rows for v in p.encode_row(row, sp)]
        assert train.load_split(split, "cpu").long().tolist() == expected
        assert counts[split] == len(expected)
        assert (tmp_path / "bin" / f"{split}.bin").stat().st_size == 2 * len(expected)
    row = splits["train"][0]
    ids = p.encode_row(row, sp)
    assert ids[0] == sp.bos_id() and ids[-1] == sp.eos_id()
    assert sp.decode(ids[1:-1]) == p.prompt.example(row["event"], row["text"])
    with pytest.raises(FileExistsError):
        p.write_split([], sp, tmp_path / "bin/train.bin")


def test_reject_signed_overflow(tmp_path):
    p = load("pretokenize.py")
    class TooLarge:
        def vocab_size(self):
            return 32769
    with pytest.raises(ValueError, match="int16"):
        p.encode_row({}, TooLarge())
