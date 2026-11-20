import importlib.util
from pathlib import Path
import copy

import pytest

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name="environment.py"):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ranges_determinism_and_endpoint_trends():
    e = load()
    records = e.generate_records(128, 7)
    assert records == e.generate_records(128, 7)
    assert records != e.generate_records(128, 8)
    assert len(records) == 128
    directions = set()
    for r in records:
        assert e.canonical_record(r) == r
        for name, _, low, high in e.FIELDS:
            assert low <= r[name] <= high
            if name in e.ANALOG:
                values = r["history"][name]
                assert len(values) == r["window"] and values[-1] == r[name]
                assert all(low <= v <= high for v in values)
                expected = "rising" if values[-1] > values[0] else "falling" if values[-1] < values[0] else "steady"
                assert r["trends"][name] == expected
                directions.add(expected)
    assert directions == {"rising", "falling", "steady"}
    assert e.trend([3, 9, 3]) == "steady"
    assert e.generate_records(0, 0) == []


def test_invalid_and_boundary_records():
    e = load()
    r = e.generate_records(1, 0)[0]
    for name, _, low, high in e.FIELDS:
        for value in (low, high):
            good = copy.deepcopy(r)
            good[name] = value
            if name in e.ANALOG:
                good["history"][name] = [value] * good["window"]
                good["trends"][name] = "steady"
            assert e.canonical_record(good) == good
        for value in (low - 1, high + 1, True, 1.5):
            with pytest.raises(ValueError):
                e.canonical_record(dict(r, **{name: value}))
    for change in ({"window": 1}, {"window": 17}, {"history": {}}, {"trends": {}}, {"extra": 1}):
        with pytest.raises(ValueError):
            e.canonical_record(dict(r, **change))
    bad = copy.deepcopy(r)
    bad["trends"]["co2"] = "unknown"
    with pytest.raises(ValueError):
        e.canonical_record(bad)
