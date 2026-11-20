from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "benchmarks/experiments/tiny_llm/ternary.md"


def test_documented_projection_example(monkeypatch):
    monkeypatch.chdir(ROOT)
    text = DOC.read_text()
    snippets = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert len(snippets) == 1
    namespace = {}
    exec(compile(snippets[0], str(DOC), "exec"), namespace)
    assert namespace["acc"].dtype == torch.int64
    assert namespace["acc"].tolist() == [[169]]
    torch.testing.assert_close(namespace["y"], torch.tensor([[169 * 3 / 127 * .6]]))
    codec = namespace["codec"]
    assert codec.payload_bytes(384, "base3") + 4 == 81
    assert codec.payload_bytes(384, "2bit") + 4 == 100
    assert (384 + 7) // 8 + 4 == 52
    assert 54 / 256 == .2109375
    assert 66 / 256 == .2578125
    for name in re.findall(r"`(test_[a-z0-9_]+\.py)`", text):
        assert (ROOT / "tests" / name).is_file()
