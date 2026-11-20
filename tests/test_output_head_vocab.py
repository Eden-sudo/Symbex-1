import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_vocab", ROOT / "benchmarks/experiments/tiny_llm/prepare_vocab.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("vocab", m.VOCABS)
def test_bounded_retokenization(vocab, tmp_path, monkeypatch):
    stories = [f"A small fox tells story {i} beside the blue river." for i in range(105)]
    prefix = (m.DELIMITER.join(stories) + m.DELIMITER).encode()
    source = tmp_path / "train.txt"
    source.write_bytes(prefix + b"THIS MUST NOT ENTER THE SLICE" * 100)
    seen = []
    original = spm.SentencePieceTrainer.train

    def capture(**kwargs):
        items = list(kwargs["sentence_iterator"])
        seen.extend(items)
        return original(**{**kwargs, "sentence_iterator": iter(items)})

    monkeypatch.setattr(spm.SentencePieceTrainer, "train", capture)
    output = m.retokenize(source, vocab=vocab, max_bytes=len(prefix), tmp_parent=tmp_path)
    info = json.loads((output / "prepare.json").read_text())
    sp = spm.SentencePieceProcessor(model_file=str(output / f"tok{vocab}.model"))
    assert info["requested_vocab"] == vocab
    assert info["actual_vocab"] == sp.get_piece_size() <= vocab
    assert info["bytes_read"] == len(prefix) and info["truncated"]
    assert seen == [s for i, s in enumerate(stories) if i % 100]
    for name, items in (("train", seen), ("val", stories[::100])):
        expected = [t for s in items for t in [1, *sp.encode(s)]]
        np.testing.assert_array_equal(np.fromfile(output / f"{name}.bin", dtype="<u2"), expected)
        assert info["tokens"][name] == len(expected)
    assert not (output / "test.bin").exists()
    assert source.read_bytes() == prefix + b"THIS MUST NOT ENTER THE SLICE" * 100


def test_rejections_and_partial_story(tmp_path):
    source = tmp_path / "text"
    source.write_text("first<|endoftext|>second café", encoding="utf-8")
    for kwargs in ({"vocab": 512}, {"max_bytes": 0}, {"max_bytes": m.MAX_BYTES + 1}):
        with pytest.raises(ValueError):
            m.retokenize(source, **kwargs)
    with pytest.raises(ValueError, match="two complete"):
        m.retokenize(source, max_bytes=source.stat().st_size - 1)
    output = m.retokenize(source, tmp_parent=tmp_path)
    assert not json.loads((output / "prepare.json").read_text())["truncated"]
