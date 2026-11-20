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


def test_number_trie_blocks_composition_and_handles_minimum_logits(tmp_path):
    t, d = load("tokenizer.py"), load("decoding.py")
    sp = t.train_tokenizer(tmp_path / "tok", rows=41)
    event = {"source": "ecg_mitbih", "class": "N", "heart_rate_before": 40, "heart_rate_after": 140}
    copy = d.NumberCopy(event, sp)
    # Exhaustively traverse legal continuations: no digit recombinations, decimals,
    # extra digits or premature termination can escape the record's two values.
    stack, outputs = [()], set()
    while stack:
        prefix = stack.pop()
        if copy.complete(prefix):
            outputs.add(sp.decode(list(prefix)))
            assert copy.allowed(prefix) == []
        else:
            stack.extend(prefix + (token,) for token in copy.allowed(prefix))
    assert outputs == {"40 ", "140 "}
    rng = np.random.default_rng(5)
    for dtype in (np.int8, np.int32, np.int64):
        for use_minimum in (False, True):
            prefix = []
            while not copy.complete(prefix):
                logits = (np.full(sp.vocab_size(), np.iinfo(dtype).min, dtype=dtype) if use_minimum
                          else rng.integers(-100, 100, sp.vocab_size()).astype(dtype))
                original = logits.copy()
                masked, legal = copy.mask(logits, prefix)
                assert masked.dtype == logits.dtype and np.array_equal(original, logits)
                assert np.array_equal(masked[legal], logits[legal])
                token = copy.choose(logits, prefix)
                assert legal[token]
                prefix.append(token)
            assert sp.decode(prefix) in outputs
    field_copy = d.NumberCopy(event, sp, field="heart_rate_before")
    assert {sp.decode(list(seq)) for seq in field_copy.sequences} == {"40 "}
    with pytest.raises(ValueError, match="complete"):
        copy.mask(np.zeros(sp.vocab_size(), dtype=np.int64), prefix)
    with pytest.raises(ValueError, match="prefix"):
        copy.allowed(sp.encode("999"))
    with pytest.raises(ValueError, match="numeric"):
        d.NumberCopy(event, sp, field="count")
    with pytest.raises(ValueError, match="integer"):
        copy.mask(np.zeros(sp.vocab_size(), dtype=float))
    logits = np.arange(sp.vocab_size(), dtype=np.int64)
    masked, legal = copy.mask(logits, enabled=False)
    assert legal.all() and np.array_equal(masked, logits)
    assert copy.choose(logits, enabled=False) == sp.vocab_size() - 1
