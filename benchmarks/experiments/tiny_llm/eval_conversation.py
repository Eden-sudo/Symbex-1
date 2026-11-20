"""Versioned bilingual conversation cases and deterministic lexical scoring."""
import argparse
import json
from pathlib import Path
import re
import unicodedata


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def contains(answer, phrase):
    needle = words(phrase)
    return any(answer[i:i + len(needle)] == needle for i in range(len(answer) - len(needle) + 1))


def validate(data):
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("expected schema_version 1 and nonempty cases")
    seen = set()
    for case in data["cases"]:
        if not isinstance(case.get("id"), str) or not case["id"] or case["id"] in seen:
            raise ValueError("case IDs must be nonempty and unique")
        seen.add(case["id"])
        if case.get("language") not in ("es", "en") or not case.get("turns"):
            raise ValueError("cases require es/en language and turns")
        for turn in case["turns"]:
            if not isinstance(turn.get("prompt"), str) or not turn["prompt"].strip():
                raise ValueError("turn requires a nonempty user prompt")
            keywords, facts = turn.get("keywords", []), turn.get("facts", [])
            if not isinstance(keywords, list) or not isinstance(facts, list):
                raise ValueError("keywords and facts must be lists")
            groups = [[kw] for kw in keywords] + facts
            if any(not isinstance(g, list) or not g or
                   any(not isinstance(p, str) or not words(p) for p in g) for g in groups):
                raise ValueError("facts are nonempty lists of alternative phrases")
            low, high = turn.get("min_words", 0), turn.get("max_words")
            if type(low) is not int or type(high) is not int or not 0 <= low <= high:
                raise ValueError("turn requires valid min_words/max_words bounds")
    return data


def score(data, answers):
    """answers maps case ID to one answer per user turn, in order.

    Facts are OR-groups of literal phrases, not semantic entailment. Repetition
    is repeated word occurrences / word count. Empty requirements score 1.
    """
    validate(data)
    if set(answers) != {c["id"] for c in data["cases"]}:
        raise ValueError("answer IDs must exactly match case IDs")
    rows = []
    for case in data["cases"]:
        responses = answers[case["id"]]
        if not isinstance(responses, list) or len(responses) != len(case["turns"]):
            raise ValueError("provide exactly one answer per turn")
        for index, (turn, answer) in enumerate(zip(case["turns"], responses)):
            if not isinstance(answer, str):
                raise ValueError("answers must be strings")
            tokens = words(answer)
            keywords, facts = turn.get("keywords", []), turn.get("facts", [])
            kh = sum(contains(tokens, k) for k in keywords)
            fh = sum(any(contains(tokens, alias) for alias in fact) for fact in facts)
            rows.append({"id": case["id"], "turn": index, "language": case["language"],
                         "keyword_hits": kh, "keyword_count": len(keywords),
                         "fact_hits": fh, "fact_count": len(facts),
                         "keyword_recall": kh / len(keywords) if keywords else 1.0,
                         "fact_recall": fh / len(facts) if facts else 1.0,
                         "words": len(tokens),
                         "length_ok": turn.get("min_words", 0) <= len(tokens) <= turn["max_words"],
                         "repetition_rate": (len(tokens) - len(set(tokens))) / len(tokens) if tokens else 0.0})
    return {"schema_version": 1, "scope": "lexical recall, not semantic truth", "turns": rows,
            "macro": {key: sum(row[key] for row in rows) / len(rows)
                      for key in ("keyword_recall", "fact_recall", "length_ok", "repetition_rate")}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases")
    p.add_argument("answers")
    a = p.parse_args(argv)
    result = score(json.loads(Path(a.cases).read_text(encoding="utf-8")),
                   json.loads(Path(a.answers).read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
