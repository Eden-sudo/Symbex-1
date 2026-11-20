import importlib.util
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/experiments/tiny_llm"


def test_documented_examples(monkeypatch):
    text = (BASE / "output_head.md").read_text()
    for name in ("factorized_head", "cluster_head"):
        spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
    examples = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(examples) == 2
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            for example in examples:
                exec(compile(example, "output_head.md", "exec"), {})
    finally:
        torch.set_num_threads(previous)


def test_documented_checks_and_limits():
    text = (BASE / "output_head.md").read_text()
    references = set(re.findall(r"tests/test_output_head_\w+\.py", text))
    assert len(references) == 6
    assert all((ROOT / path).is_file() for path in references)
    for artifact in ("prepare_vocab.py", "factorized_head.py", "cluster_head.py",
                     "head_cost.py", "c/cluster_head.c", "c/cluster_head.h"):
        assert (BASE / artifact).is_file()
    assert "only if its winning token is shortlisted" in text
    assert "not bounds" in text
    assert "remain unverified" in text
