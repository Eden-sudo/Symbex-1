"""Tiny, local SentencePiece coverage for the TinyStories preparation script."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

try:
    import numpy as np
except ImportError as exc:
    raise ImportError("NumPy is required for the tiny LLM preparation tests") from exc

try:
    import sentencepiece as spm
except ImportError as exc:
    raise ImportError("SentencePiece is required for the tiny LLM preparation tests") from exc

import pytest


@pytest.fixture
def prepare(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/prepare.py"
    spec = importlib.util.spec_from_file_location(f"tiny_prepare_{uuid.uuid4().hex}", path)
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_sp(tmp_path):
    prefix = str(tmp_path / "toy")
    sentences = [f"A small story number {i} has a happy ending." for i in range(40)]
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences), model_prefix=prefix, model_type="bpe",
        vocab_size=60, byte_fallback=False, hard_vocab_limit=False, num_threads=1,
        unk_id=0, bos_id=3, eos_id=2, pad_id=-1, minloglevel=2,
    )
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def expected_tokens(sp, items):
    return [token for item in items for token in [sp.bos_id(), *sp.encode(item)]]


@pytest.mark.parametrize("text, expected", [
    ("", []),
    (" \n\t ", []),
    ("<|endoftext|><|endoftext|>", []),
    ("  single story  ", ["single story"]),
    ("<|endoftext|>  café\n  inside  <|endoftext|><|endoftext|> \n "
     "<|endoftext|> second story <|endoftext|>", ["café\n  inside", "second story"]),
])
def test_stories_splits_strips_and_discards_empty_pieces(prepare, tmp_path, text, expected):
    path = tmp_path / "stories.txt"
    path.write_text(text, encoding="utf-8")
    assert prepare.stories(path) == expected


def test_encode_writes_bos_payloads_and_uint16(prepare, tiny_sp, tmp_path):
    items = ["A small story", "", "A happy ending", "number 123"]
    payloads = [tiny_sp.encode(item) for item in items]
    path = tmp_path / "tokens.bin"
    count = prepare.encode(tiny_sp, items, path, 1)
    assert tiny_sp.bos_id() != 1
    assert count == sum(len(payload) + 1 for payload in payloads)
    assert path.stat().st_size == count * np.dtype(np.uint16).itemsize
    actual = np.fromfile(path, dtype=np.uint16)
    np.testing.assert_array_equal(actual, expected_tokens(tiny_sp, items))
    offset = 0
    for payload in payloads:
        assert actual[offset] == tiny_sp.bos_id()
        offset += len(payload) + 1


def test_encode_empty_items_truncates_output(prepare, tiny_sp, tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"old tokens")
    assert prepare.encode(tiny_sp, [], path, 1) == 0
    assert path.is_file() and path.stat().st_size == 0


def test_main_partitions_training_and_keeps_official_valid_separate(prepare, tmp_path, monkeypatch):
    training = [f"Training story number {i} has a small happy ending." for i in range(205)]
    official = [f"Official valid story number {i} is separate." for i in range(4)]
    delimiter = "<|endoftext|>"
    (tmp_path / "train.txt").write_text(
        delimiter + (delimiter + " \n " + delimiter).join(training) + delimiter, encoding="utf-8")
    (tmp_path / "valid.txt").write_text(delimiter.join(official), encoding="utf-8")
    indices = {0, 100, 200}
    val = [story for i, story in enumerate(training) if i in indices]
    train = [story for i, story in enumerate(training) if i not in indices]
    assert len(val) == 3 and len(train) == 202
    assert set(train).isdisjoint(val) and set(train) | set(val) == set(training)
    assert set(training).isdisjoint(official)
    assert prepare.VAL_EVERY == 100

    sample_seen = []
    real_train = spm.SentencePieceTrainer.train

    def train_toy(**kwargs):
        sample = list(kwargs["sentence_iterator"])
        sample_seen.extend(sample)
        kwargs.update(sentence_iterator=iter(sample), byte_fallback=False,
                      hard_vocab_limit=False, num_threads=1)
        return real_train(**kwargs)

    monkeypatch.setattr(prepare, "VOCAB", 60)
    monkeypatch.setattr(prepare, "SP_STORIES", 32)
    monkeypatch.setattr(prepare, "spm", SimpleNamespace(
        SentencePieceTrainer=SimpleNamespace(train=train_toy),
        SentencePieceProcessor=spm.SentencePieceProcessor,
    ))
    monkeypatch.setattr(prepare, "os", SimpleNamespace(path=os.path, cpu_count=lambda: 1))
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--data", str(tmp_path)])
    prepare.main()

    assert len(sample_seen) == len(set(sample_seen)) == 32
    assert set(sample_seen) <= set(train)
    assert set(sample_seen).isdisjoint(val + official)
    model = tmp_path / "tok60.model"
    assert model.is_file() and (tmp_path / "tok60.vocab").is_file()
    sp = spm.SentencePieceProcessor(model_file=str(model))
    splits = {"train": train, "val": val, "test": official}
    token_counts = {}
    for name, items in splits.items():
        expected = expected_tokens(sp, items)
        path = tmp_path / f"{name}.bin"
        assert path.is_file() and path.stat().st_size == 2 * len(expected)
        np.testing.assert_array_equal(np.fromfile(path, dtype=np.uint16), expected)
        token_counts[name] = len(expected)
    info = json.loads((tmp_path / "prepare.json").read_text(encoding="utf-8"))
    assert info == {"vocab": 60, "val_every": 100, "sp_stories": 32,
                    "stories": {name: len(items) for name, items in splits.items()},
                    "tokens": token_counts}
