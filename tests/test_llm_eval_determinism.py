import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"
spec = importlib.util.spec_from_file_location("determinism", ROOT / "eval_determinism.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


@pytest.mark.parametrize("temperature,top_p", [(0, 65536), (4, 65536), (4, 45000)])
def test_independent_histories_and_json(tmp_path, temperature, top_p):
    histories = []

    def scores(ids):
        histories.append(list(ids))
        return [((sum(ids) + j) % 3) for j in range(7)]

    output = tmp_path / "record.json"
    result = ev.check(scores, [1, 2], 12, 42, output, temperature=temperature, top_p=top_p)
    assert json.loads(output.read_text()) == result
    assert result["equal"]
    assert histories[:12] == histories[12:24] == histories[24:]
    assert len(result["c"]["ids"]) == 12
    state = 42
    for actual in result["c"]["states"]:
        if temperature:
            state = ev.loss.local("sample_int").xorshift32(state)
        assert actual == state


def test_invalid_inputs_do_not_write(tmp_path):
    output = tmp_path / "record.json"
    for prompt, steps, seed in [([], 2, 1), ([1], 0, 1), ([1], 2, 0), ([-1], 2, 1)]:
        with pytest.raises(ValueError):
            ev.check(lambda ids: [0, 0], prompt, steps, seed, output)
    assert not output.exists()


def test_mismatch_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "c_sampler", lambda directory: lambda *args: (1, args[-1]))
    result = ev.check(lambda ids: [3, 0], [0], 2, 7, tmp_path / "bad.json", temperature=0)
    assert result["equal"] is False
    assert result["python_first"]["ids"] == [0, 0]
    assert result["c"]["ids"] == [1, 1]
