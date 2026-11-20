import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"
spec = importlib.util.spec_from_file_location("results", ROOT / "eval_results.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
PROVENANCE = dict(evidence="measured", source="synthetic fixture", environment="host CPU",
                  split="synthetic", commit="test-fixture", engine="float")


def test_schema_and_statistics(tmp_path):
    path = tmp_path / "result.json"
    record = ev.write_result(path, [42, 43, 44], {"loss": {"unit": "nats/token", "values": [1, 2, 3]}}, **PROVENANCE)
    assert json.loads(path.read_text()) == record
    assert set(record) == {"schema_version", "seeds", "seed_count", "status", "evidence", "is_estimate",
                           "source", "environment", "split", "commit", "engine", "metrics"}
    assert record["metrics"]["loss"] == {"unit": "nats/token", "values": [1, 2, 3],
                                          "mean": 2, "std": math.sqrt(2 / 3), "std_ddof": 0}
    assert record["status"] == "multi_seed" and not record["is_estimate"]
    with pytest.raises(FileExistsError):
        ev.write_result(path, [42], {"loss": {"unit": "nats/token", "values": [4]}}, **PROVENANCE)
    assert json.loads(path.read_text()) == record


def test_estimates_and_seed_labels():
    for seeds, status in [([42], "ranking"), ([42, 43], "preliminary")]:
        r = ev.result(seeds, {"bytes": {"unit": "bytes", "values": [10] * len(seeds)}},
                      **{**PROVENANCE, "evidence": "estimate", "source": "counted tensor payload"})
        assert r["status"] == status and r["is_estimate"]
        assert r["metrics"]["bytes"]["std"] == 0


@pytest.mark.parametrize("seeds,values", [([], []), ([1, 1], [2, 3]), ([True], [2]),
                                         ([1], []), ([1], [float("nan")]), ([1], [True])])
def test_reject_bad_measurements(tmp_path, seeds, values):
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        ev.write_result(path, seeds, {"loss": {"unit": "nats/token", "values": values}}, **PROVENANCE)
    assert not path.exists()


def test_cli_and_required_provenance(tmp_path):
    payload = dict(seeds=[42], metrics={"loss": {"unit": "nats/token", "values": [2]}}, **PROVENANCE)
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(payload))
    ev.main([str(source), str(output)])
    assert json.loads(output.read_text())["status"] == "ranking"
    for field, value in [("evidence", "unknown"), ("source", ""), ("split", "unknown")]:
        with pytest.raises(ValueError):
            ev.result(**{**payload, field: value})
