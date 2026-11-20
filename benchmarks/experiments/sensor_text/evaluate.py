"""Conservative field-aware evaluation of the synthetic domain's sentence grammar."""
import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import re
import string

_spec = importlib.util.spec_from_file_location("sensor_modules", Path(__file__).with_name("_modules.py"))
_modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modules)
prompt = _modules.load("prompt.py")
dataset = _modules.load("make_dataset.py")
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBERS = re.compile(r"(?<![\w.])" + NUMBER + r"(?!\w)")


def _extract(source, sentence):
    for template in dataset.TEMPLATES[source]:
        parts = []
        for literal, field, _, _ in string.Formatter().parse(template):
            parts.append(re.escape(literal))
            if field:
                value = r"[A-Za-z][A-Za-z0-9_]{0,63}" if field == "class" else NUMBER
                parts.append(f"(?P<{field}>{value})")
        match = re.fullmatch("".join(parts), sentence)
        if match:
            return match.groupdict()
    return None


def score(event, generated, reference):
    """Field accuracy includes class and each numeric field with its correct unit.

    Only the four declared sentence forms per source are parsed. Unknown wording
    (including extra clauses or wrong units) scores zero, with supported=False.
    Numeric hallucinations count occurrences of values absent from the record;
    swapped ECG values are field errors, not numeric hallucinations.
    """
    if event.get("source") == "environment":
        return score_environment(event, generated, reference)
    record = prompt.canonical_record(event)
    fields = [field[0] for field in prompt.FIELDS[record["source"]]]
    allowed = {Decimal(record[name]) for name in fields}
    hallucinations = sum(Decimal(match.group()) not in allowed for match in NUMBERS.finditer(generated))
    extracted = _extract(record["source"], generated)
    correct = 0
    if extracted is not None:
        correct = int(extracted["class"] == record["class"])
        correct += sum(Decimal(extracted[name]) == Decimal(record[name]) for name in fields)
    total = len(fields) + 1
    return {"exact_match": generated == reference, "field_fact_accuracy": correct / total,
            "correct_fields": correct, "total_fields": total,
            "all_facts_correct": correct == total and hallucinations == 0,
            "hallucination_count": hallucinations, "supported": extracted is not None,
            "missed_alerts": 0, "false_alerts": 0, "alerts_supported": True}


def score_environment(event, generated, reference):
    b = prompt.bilingual
    record = b.environment.canonical_record(event)
    facts = b.facts(record)
    language = record["language"]
    # Match whole declared clauses, not substrings that could credit a negation.
    clauses = generated[:-1].split("; ") if generated.endswith(".") else []
    correct = 0
    remainder = generated
    for name, values in facts.items():
        templates = b.ANALOG_TEMPLATES if name in b.environment.ANALOG else b.STATE_TEMPLATES
        matches = [template.format(**values) for template in templates[language]]
        # Some analog forms contain semicolons internally; use bounded full phrases.
        found = [text for text in matches if re.search(r"(?:^|; )" + re.escape(text) + r"(?=; |\.$)", generated)]
        if len(found) == 1:
            correct += len(values) - 1  # label selects field; unit is a fact too
            remainder = remainder.replace(found[0], "", 1)
    total = sum(len(v) - 1 for v in facts.values())
    alert_score = b.alerts.score(record, generated)
    allowed = {Decimal(record[name]) for name in b.environment.ANALOG} | {Decimal(record["window"])}
    hallucinations = sum(Decimal(m.group()) not in allowed for m in NUMBERS.finditer(generated))
    # Unsupported wording is explicitly flagged, including extra/contradictory clauses.
    expected_alert = b.alerts.render(record)
    remainder = remainder.replace(expected_alert, "", 1)
    supported = not remainder.strip("; .") and correct == total
    return {"exact_match": generated == reference, "field_fact_accuracy": correct / total,
            "correct_fields": correct, "total_fields": total,
            "all_facts_correct": supported and not hallucinations and alert_score["alerts_supported"]
                                 and not alert_score["missed_alerts"] and not alert_score["false_alerts"],
            "hallucination_count": hallucinations, "supported": supported, **alert_score}


def evaluate(rows):
    results = [score(row["event"], row["generated"], row["text"]) for row in rows]
    if not results:
        raise ValueError("evaluation needs at least one example")
    return {"examples": len(results),
            "exact_match": sum(r["exact_match"] for r in results) / len(results),
            "field_fact_accuracy": sum(r["correct_fields"] for r in results) / sum(r["total_fields"] for r in results),
            "hallucination_count": sum(r["hallucination_count"] for r in results),
            "missed_alerts": sum(r["missed_alerts"] for r in results),
            "false_alerts": sum(r["false_alerts"] for r in results),
            "unsupported_alert_count": sum(not r["alerts_supported"] for r in results),
            "unsupported_count": sum(not r["supported"] for r in results), "rows": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSONL: event, text (reference), generated")
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8") as stream:
        print(json.dumps(evaluate(json.loads(line) for line in stream), indent=2))


if __name__ == "__main__":
    main()
