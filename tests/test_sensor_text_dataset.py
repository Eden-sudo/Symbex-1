"""CPU-only checks for the synthetic sensor text dataset."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/experiments/sensor_text/make_dataset.py"
SOURCE_PATHS = {
    "har_ultracompact": ("benchmarks/conv1d/har_ultracompact/har_data.py", "CLASSES"),
    "ecg_mitbih": ("benchmarks/conv1d/ecg_mitbih/ecg_data.py", "CLASSES"),
    "fomo_voc_person": ("benchmarks/conv2d/fomo_voc_person/fomo_classes.py", "VOC_CLASSES"),
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load module from {path}"
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.fail(f"dependency missing while loading {path}: {exc}")
    finally:
        sys.dont_write_bytecode = old
    return module


@pytest.fixture
def dataset():
    return load(SCRIPT, "sensor_text_test")


@pytest.fixture
def classes():
    return {source: list(getattr(load(ROOT / path, f"independent_{source}"), symbol))
            for source, (path, symbol) in SOURCE_PATHS.items()}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_reproducibility_counts_and_coverage(dataset, classes, tmp_path):
    minimum = sum(map(len, classes.values()))
    sizes = [minimum, 7, 0]
    for name, seed in (("one", 8), ("two", 8), ("other", 9)):
        rows, meta = dataset.generate_dataset(sizes, seed)
        dataset.write_dataset(tmp_path / name, rows, meta)
    names = {"train.jsonl", "val.jsonl", "test.jsonl", "meta.json"}
    assert {p.name for p in (tmp_path / "one").iterdir()} == names
    for name in names:
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    assert (tmp_path / "one/train.jsonl").read_bytes() != (tmp_path / "other/train.jsonl").read_bytes()
    meta = json.loads((tmp_path / "one/meta.json").read_text())
    assert meta["classes"] == classes
    assert meta["sizes"] == dict(zip(("train", "val", "test"), sizes))
    pairs = [(source, label) for source, labels in classes.items() for label in labels]
    for split, size in zip(("train", "val", "test"), sizes):
        rows = read_rows(tmp_path / "one" / f"{split}.jsonl")
        assert len(rows) == size
        assert [(r["event"]["source"], r["event"]["class"]) for r in rows] == [pairs[i % minimum] for i in range(size)]
        counts = {source: dict.fromkeys(labels, 0) for source, labels in classes.items()}
        for row in rows:
            event = row["event"]
            counts[event["source"]][event["class"]] += 1
        assert meta["counts_per_class"][split] == counts
    with pytest.raises(ValueError, match="train size"):
        dataset.generate_dataset([minimum - 1, 0, 0], 0)


def test_templates_and_event_contract(dataset, classes):
    minimum = sum(map(len, classes.values()))
    splits, meta = dataset.generate_dataset([minimum * 3, minimum * 2, minimum * 2], 13)
    ids = {split: set(sum(meta["template_ids"][split].values(), [])) for split in splits}
    assert ids["train"].isdisjoint(ids["val"] | ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    texts = {split: {row["text"] for row in rows} for split, rows in splits.items()}
    assert texts["train"].isdisjoint(texts["val"] | texts["test"])
    for split, rows in splits.items():
        for row in rows:
            assert set(row) == {"event", "text"}
            event, text = row["event"], row["text"]
            source = event["source"]
            assert event["class"] in classes[source]
            assert event["template_id"] in ids[split]
            fields = {"har_ultracompact": {"duration_seconds": (1, 120)},
                      "ecg_mitbih": {"heart_rate_before": (40, 180), "heart_rate_after": (40, 180)},
                      "fomo_voc_person": {"count": (0, 5)}}[source]
            assert set(event) == {"source", "class", "template_id"} | set(fields)
            for field, (low, high) in fields.items():
                assert type(event[field]) is int and low <= event[field] <= high
            assert set(map(int, re.findall(r"\b\d+\b", text))) <= {event[f] for f in fields}
            assert 3 <= len(text.split()) <= 25 and text.endswith(".") and "\n" not in text
    for source, labels in classes.items():
        for label in labels:
            event = {"source": source, "class": label, "duration_seconds": 1,
                     "heart_rate_before": 40, "heart_rate_after": 180, "count": 0}
            rendered = [dataset.render_event(event, f"{source}.{letter}") for letter in "abcd"]
            assert len(set(rendered)) == 4
            assert all(label in text and text.endswith(".") for text in rendered)
    with pytest.raises(ValueError, match="template"):
        dataset.render_event({"source": "har_ultracompact"}, "unknown")
    with pytest.raises(ValueError, match="template"):
        dataset.render_event({"source": "har_ultracompact"}, "ecg_mitbih.a")


def test_boundary_wording(dataset):
    for duration in (1, 120):
        e = {"source": "har_ultracompact", "class": "Walking", "duration_seconds": duration}
        assert dataset.render_event(e, "har_ultracompact.a") == f"Activity Walking lasted {duration} seconds."
    for count in (0, 1, 5):
        e = {"source": "fomo_voc_person", "class": "person", "count": count}
        assert dataset.render_event(e, "fomo_voc_person.c") == f"For class person, the recorded detection count was {count}."
    for before, after in ((40, 180), (180, 40), (40, 40)):
        e = {"source": "ecg_mitbih", "class": "N", "heart_rate_before": before, "heart_rate_after": after}
        assert dataset.render_event(e, "ecg_mitbih.d") == (f"Recorded beat class: N; heart rate readings: {before} "
                                                            f"then {after} beats per minute.")


def test_cli(dataset, classes, tmp_path):
    minimum = str(sum(map(len, classes.values())))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    def run(*args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=tmp_path,
                              env=env, capture_output=True, text=True, timeout=15)
    out = tmp_path / "out"
    result = run("--out", out, "--n", minimum, "2", "3", "--seed", "-1")
    assert result.returncode == 0, result.stderr
    assert {p.name for p in out.iterdir()} == {"train.jsonl", "val.jsonl", "test.jsonl", "meta.json"}
    sentinel = (out / "meta.json").read_bytes()
    result = run("--out", out, "--n", minimum, "0", "0")
    assert result.returncode == 1 and "error:" in result.stderr
    assert (out / "meta.json").read_bytes() == sentinel
    target = tmp_path / "missing"
    for args in ((), ("--out", target, "--n", minimum, "0"),
                 ("--out", target, "--n", minimum, "x", "0"),
                 ("--out", target, "--n", minimum, "0", "0", "--seed", "x"),
                 ("--out", target, "--n", minimum, "-1", "0"),
                 ("--out", target, "--n", str(int(minimum) - 1), "0", "0")):
        result = run(*args)
        assert result.returncode == 2 and "error:" in result.stderr
        assert not target.exists()
    parent = tmp_path / "file"
    parent.write_text("sentinel")
    result = run("--out", parent / "out", "--n", minimum, "0", "0")
    assert result.returncode == 1 and "error:" in result.stderr
    assert parent.read_text() == "sentinel"


def test_failures(dataset, classes, tmp_path, monkeypatch):
    sizes = [sum(map(len, classes.values())), 0, 0]
    rows, meta = dataset.generate_dataset(sizes, 0)
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"unchanged")
    original = Path.open
    def fail(self, *args, **kwargs):
        if self.name == "val.jsonl":
            raise OSError("injected write failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError, match="injected"):
        dataset.write_dataset(tmp_path / "partial", rows, meta)
    assert not (tmp_path / "partial/meta.json").exists()
    assert sentinel.read_bytes() == b"unchanged"
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    with pytest.raises(ImportError, match="cannot load source module"):
        dataset.generate_dataset(sizes, 0)
