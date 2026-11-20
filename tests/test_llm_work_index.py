"""Keep the single integration index's executable references honest."""
from pathlib import Path
import re


def test_index_references_existing_checks():
    root = Path(__file__).resolve().parents[1]
    index = root / "benchmarks/experiments/tiny_llm/LLM_WORK.md"
    text = index.read_text()
    for link in re.findall(r"\]\(([^)]+)\)", text):
        assert (index.parent / link).is_file(), link
    references = set(re.findall(r"tests/test_[a-z0-9_]+\.py", text))
    assert len(references) >= 50
    for reference in references:
        assert (root / reference).is_file(), reference
    for name in ("host", "chat", "sensor"):
        assert f"tests/test_llm_{name}_pipeline.py" in references
    for section in ("Core and attention", "LittleBit", "Runtime", "Importers",
                    "Sensor-to-text", "Energy and existing research"):
        assert f"## {section}" in text
    assert "Both histories use Python GQA forward" in text
    assert "remain unverified" in text
