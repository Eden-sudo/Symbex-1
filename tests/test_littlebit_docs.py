"""Keep round-2 documentation's module and test references resolvable."""
from pathlib import Path


def test_round_two_documented_entrypoints_exist():
    root = Path(__file__).resolve().parents[1]
    tiny = root / "benchmarks/experiments/tiny_llm"
    text = (tiny / "littlebit.md").read_text()
    for suffix in ("distill", "sensitivity", "container", "quality"):
        assert f"littlebit_{suffix}" in text
        assert (tiny / f"littlebit_{suffix}.py").is_file()
        assert (root / f"tests/test_littlebit_{suffix}.py").is_file()
    assert "c/tinylm_factor_popcount.h" in text
    assert (tiny / "c/tinylm_factor_popcount.h").is_file()
    assert "not a globally optimal" in text
    assert "not a runnable model" in text
    assert "no faster-runtime claim" in text
