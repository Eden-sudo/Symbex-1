import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text/evaluate.py"
    spec = importlib.util.spec_from_file_location("sensor_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ecg_identity_units_and_hallucinations():
    e = load()
    event = {"source": "ecg_mitbih", "class": "N", "heart_rate_before": 72, "heart_rate_after": 90}
    ref = "Beat class N was recorded with heart rate changing from 72 to 90 beats per minute."
    good = e.score(event, ref, ref)
    assert good["exact_match"] and good["field_fact_accuracy"] == 1
    swapped = ref.replace("72 to 90", "90 to 72")
    result = e.score(event, swapped, ref)
    assert result["field_fact_accuracy"] == pytest.approx(1 / 3)
    assert result["hallucination_count"] == 0 and not result["all_facts_correct"]
    wrong = e.score(event, ref.replace("90", "900"), ref)
    assert wrong["field_fact_accuracy"] == pytest.approx(2 / 3)
    assert wrong["hallucination_count"] == 1
    assert e.score(event, ref.replace("beats per minute", "seconds"), ref)["field_fact_accuracy"] == 0
    for value in ("-90", "90.5", "9e2"):
        assert e.score(event, ref.replace("90", value), ref)["hallucination_count"] == 1
    assert e.score(event, ref + " Also 999 and 999.", ref)["hallucination_count"] == 2
    assert e.score(event, "Everything is fine.", ref)["field_fact_accuracy"] == 0


def test_templates_class_counts_and_aggregate():
    e = load()
    events = [{"source": "har_ultracompact", "class": "Walking", "duration_seconds": 10},
              {"source": "fomo_voc_person", "class": "person", "count": 0},
              {"source": "ecg_mitbih", "class": "N", "heart_rate_before": 40, "heart_rate_after": 180}]
    rows = []
    for event in events:
        reference = e.dataset.render_event(event, event["source"] + ".a")
        for suffix in "abcd":
            generated = e.dataset.render_event(event, event["source"] + "." + suffix)
            result = e.score(event, generated, reference)
            assert result["all_facts_correct"] and result["hallucination_count"] == 0
            rows.append(dict(event=event, text=reference, generated=generated))
    result = e.evaluate(rows)
    assert result["field_fact_accuracy"] == 1 and result["exact_match"] == 0.25
    assert result["unsupported_count"] == 0
    result = e.score(events[1], "Detected dog objects: 0.", "Detected person objects: 0.")
    assert result["field_fact_accuracy"] == 0.5
    with pytest.raises(ValueError):
        e.evaluate([])
