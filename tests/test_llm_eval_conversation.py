import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"
spec = importlib.util.spec_from_file_location("conversation", ROOT / "eval_conversation.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_bilingual_cases_and_turn_alignment():
    data = json.loads((ROOT / "eval_conversations.json").read_text())
    result = ev.score(data, {"memory-es": ["ANA", "Lima Lima"], "memory-en": ["Sam", "blue"]})
    assert result["macro"]["fact_recall"] == 1
    assert result["macro"]["keyword_recall"] == 1
    assert result["macro"]["length_ok"] == 1
    assert result["turns"][1]["repetition_rate"] == .5
    with pytest.raises(ValueError, match="one answer"):
        ev.score(data, {"memory-es": ["Ana"], "memory-en": ["Sam", "blue"]})


def test_phrase_boundaries_unicode_empty_and_partial():
    data = {"schema_version": 1, "cases": [{"id": "a", "language": "es", "turns": [
        {"prompt": "Describe.", "keywords": ["sí", "cat"],
         "facts": [["azul", "blue"], ["dos gatos"]], "min_words": 1, "max_words": 4}]}]}
    row = ev.score(data, {"a": ["SI\u0301 blue concatenate"]})["turns"][0]
    assert row["keyword_recall"] == .5  # cat must not match concatenate
    assert row["fact_recall"] == .5
    assert row["length_ok"]
    row = ev.score(data, {"a": [""]})["turns"][0]
    assert row["fact_recall"] == 0 and row["repetition_rate"] == 0
    assert not row["length_ok"]
    for field, value in [("facts", [[]]), ("keywords", [""]), ("max_words", -1)]:
        bad = copy.deepcopy(data)
        bad["cases"][0]["turns"][0][field] = value
        with pytest.raises(ValueError):
            ev.validate(bad)
    with pytest.raises(ValueError, match="IDs"):
        ev.score(data, {"unknown": ["blue"]})
