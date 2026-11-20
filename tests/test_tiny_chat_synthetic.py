from collections import Counter
import random
import re
import pytest
from tiny_runtime_support import LM, module
from test_tiny_runtime_format import Vocab


def test_seed_coverage_and_formatter(monkeypatch):
    monkeypatch.syspath_prepend(str(LM))
    generator, data = module("chat_synthetic"), module("chat_data")
    state = random.getstate()
    rows = generator.generate(42, 2)
    assert random.getstate() == state
    assert rows == generator.generate(42, 2)
    assert rows != generator.generate(43, 2)
    expected = {(lang, intent): 2 for lang in ("es", "en")
                for intent in ("greeting", "sensor_state", "yes", "no", "thanks", "bye")}
    assert Counter((r["language"], r["intent"]) for r in rows) == expected
    sp = Vocab()
    for row in rows:
        turns = row["turns"]
        assert [role for role, _ in turns] == ["system", "user", "assistant"]
        ids, mask = data.format_dialogue(sp, turns)
        assert [t for t, m in zip(ids, mask) if m] == sp.encode(turns[-1][1]) + [2]
        answer = turns[-1][1]
        if row["intent"] == "sensor_state":
            assert re.search(r"\d+", answer)[0] == re.search(r"\d+", turns[0][1])[0]
        if row["intent"] == "yes":
            assert answer.startswith("Sí" if row["language"] == "es" else "Yes")
        if row["intent"] == "no":
            assert answer.startswith("No")


def test_invalid_counts():
    generator = module("chat_synthetic")
    for n in (0, -1, 1001, True, 1.5):
        with pytest.raises(ValueError):
            generator.generate(42, n)
