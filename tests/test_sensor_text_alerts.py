import importlib.util
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(b, co2, battery, sound, language):
    r = dict(b.environment.generate_records(1, 0)[0], language=language)
    for name, value in (("co2", co2), ("battery", battery), ("sound", sound)):
        r[name] = value
        r["history"][name] = [value] * r["window"]
        r["trends"][name] = "steady"
    return r


def test_thresholds_bilingual_mentions_and_evaluation():
    e = load("evaluate.py")
    b = e.prompt.bilingual
    rows = []
    for language in ("en", "es"):
        quiet = record(b, 999, 21, 84, language)
        active = record(b, 1000, 20, 85, language)
        assert b.alerts.labels(quiet) == []
        assert b.alerts.labels(active) == ["co2_high", "battery_low", "sound_high"]
        for r in (quiet, active):
            for phrasing in range(3):
                text = b.render(r, phrasing)
                result = e.score(r, text, text)
                assert result["all_facts_correct"] and result["field_fact_accuracy"] == 1
                assert result["missed_alerts"] == result["false_alerts"] == 0
                rows.append(dict(event=r, text=text, generated=text))
        ref = b.render(active)
        missing = ref.replace(b.alerts.render(active), b.alerts.render(quiet))
        assert e.score(active, missing, ref)["missed_alerts"] == 3
        false = b.render(quiet).replace(b.alerts.render(quiet), b.alerts.render(active))
        assert e.score(quiet, false, b.render(quiet))["false_alerts"] == 3
        unsupported = e.score(active, "No high CO2 or low battery.", ref)
        assert unsupported["missed_alerts"] == 3 and not unsupported["alerts_supported"]
        assert not e.score(active, ref + " Contradiction.", ref)["all_facts_correct"]
    result = e.evaluate(rows)
    assert result["missed_alerts"] == result["false_alerts"] == result["unsupported_count"] == 0


def test_round_one_evaluation_still_supported():
    e = load("evaluate.py")
    event = {"source": "fomo_voc_person", "class": "person", "count": 1}
    ref = "Detected person objects: 1."
    assert e.score(event, ref, ref)["all_facts_correct"]
