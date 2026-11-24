import importlib.util
from pathlib import Path


def test_dataset_card_matches_recipe_and_fields():
    directory = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"
    spec = importlib.util.spec_from_file_location("card_dataset", directory / "environment_dataset.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    card = (directory / "sensor_text.md").read_text()
    sizes, seed, mode = d.generate_dataset.__defaults__
    assert sizes == (256, 64, 64) and seed == 0 and mode == "template_holdout"
    assert "256 train / 64 val / 64 test" in card
    for name, unit, low, high in d.bilingual.environment.FIELDS:
        assert f"| {name} | {unit} | {low}..{high} |" in card
    assert "entirely synthetic" in card and "v1 only" in card
    assert "not a board run" in card
    splits, meta = d.generate_dataset((2, 2, 2))
    assert meta["mode"] in card and meta["schema"] in card
    assert d.check_leakage(splits)["cross_split_pairs"] == 0
