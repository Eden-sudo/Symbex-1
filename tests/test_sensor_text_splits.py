import copy
import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text/environment_dataset.py"
    spec = importlib.util.spec_from_file_location("environment_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["seed", "template_holdout"])
def test_seed_splits_and_template_policy(mode, tmp_path):
    d = load()
    splits, meta = d.generate_dataset((32, 16, 16), 7, mode)
    assert (splits, meta) == d.generate_dataset((32, 16, 16), 7, mode)
    assert splits != d.generate_dataset((32, 16, 16), 8, mode)[0]
    assert len(set(meta["split_seeds"].values())) == 3
    keys = {name: {d.pair_key(row) for row in rows} for name, rows in splits.items()}
    for name, rows in splits.items():
        assert len(keys[name]) == meta["sizes"][name]
        assert {r["event"]["language"] for r in rows} == {"en", "es"}
        assert all(int(r["event"]["template_id"].split(".")[-1]) in meta["phrasings"][name] for r in rows)
        for other in splits:
            if other != name:
                assert keys[name].isdisjoint(keys[other])
    if mode == "template_holdout":
        assert set(meta["phrasings"]["test"]).isdisjoint(meta["phrasings"]["train"] + meta["phrasings"]["val"])
    else:
        assert all(v == [0, 1, 2] for v in meta["phrasings"].values())
    # Changing test size cannot perturb train or validation RNG streams.
    expanded, _ = d.generate_dataset((32, 16, 24), 7, mode)
    assert expanded["train"] == splits["train"] and expanded["val"] == splits["val"]
    d.dataset.write_dataset(tmp_path / "data", splits, meta)
    assert json.loads((tmp_path / "data/meta.json").read_text()) == meta


def test_leakage_ignores_template_metadata_and_rejects_bad_sizes():
    d = load()
    splits, _ = d.generate_dataset((1, 0, 0), -1)
    row = copy.deepcopy(splits["train"][0])
    row["event"]["template_id"] = "disguised"
    splits["test"].append(row)
    with pytest.raises(ValueError, match="crosses"):
        d.check_leakage(splits)
    for sizes in ((1, 2), (1, -1, 0), (True, 0, 0)):
        with pytest.raises(ValueError):
            d.generate_dataset(sizes)
    assert d.generate_dataset((0, 0, 0))[0] == {"train": [], "val": [], "test": []}
