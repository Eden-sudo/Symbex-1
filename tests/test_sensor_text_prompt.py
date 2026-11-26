import importlib.util
from pathlib import Path
import random

import pytest

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round_trip():
    p, d = load("prompt.py"), load("make_dataset.py")
    rng = random.Random(91)
    for source, labels in d.load_source_classes().items():
        for label in labels:
            for _ in range(3):
                event = d.make_event(source, label, rng)
                assert p.parse(p.serialize(event)) == event
                assert p.serialize(dict(reversed(list(event.items())))) == p.serialize(event)
                assert p.serialize(dict(event, template_id="held-out")) == p.serialize(event)
    event = dict(source="ecg_mitbih", **{"class": "N"}, heart_rate_before=72, heart_rate_after=90)
    assert p.serialize(event) == "<sensor_v1>|source=ecg_mitbih|class=N|heart_rate_before=72 bpm|heart_rate_after=90 bpm|<text>"


def test_reject_noncanonical():
    p = load("prompt.py")
    event = {"source": "har_ultracompact", "class": "Walking", "duration_seconds": 1}
    text = p.serialize(event)
    for bad in (text + "\n", text.replace("1 s", "01 s"), text.replace("1 s", "+1 s"),
                text.replace("1 s", "1 seconds"), text.replace("|class=Walking", "|class=Walking|class=Walking")):
        with pytest.raises(ValueError):
            p.parse(bad)
    for value in (True, 1.0, 0, 121, -1):
        with pytest.raises(ValueError):
            p.serialize(dict(event, duration_seconds=value))
    for changes in ({"extra": 1}, {"class": "bad|class"}, {"source": "unknown"}):
        with pytest.raises(ValueError):
            p.serialize(dict(event, **changes))
    assert p.example(event, "Activity Walking lasted 1 seconds.").endswith(".</text>")
