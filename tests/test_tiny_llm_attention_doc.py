"""Execute the documentation's tiny generation example without downloads."""
import importlib.util
import sys
from pathlib import Path

import torch

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def test_documented_example(monkeypatch):
    for name in ("train", "intsim", "gqa_intsim", "gqa_rolling"):
        spec = importlib.util.spec_from_file_location(name, DIR / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    text = (DIR / "attention_memory.md").read_text()
    example = text.split("```python\n", 1)[1].split("```", 1)[0]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        namespace = {}
        exec(compile(example, "attention_memory.md", "exec"), namespace)
        assert namespace["state"].position == 12
        assert namespace["scores"].shape == (16,)
    finally:
        torch.set_num_threads(previous)
