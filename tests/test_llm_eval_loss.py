import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"
spec = importlib.util.spec_from_file_location("eval_loss", ROOT / "eval_loss.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def make_run(tmp_path, weights="binary"):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    args = dict(dim=64, layers=1, heads=1, hidden=64, seq=3, weights=weights)
    model = ev.local("train").TinyLM(9, 64, 1, 1, 64, 3, weights).eval()
    torch.save({"args": args, "model": model.state_dict()}, tmp_path / "model.pt")
    tokens = tmp_path / "tokens.bin"
    np.array([8, 1, 2, 3, 4, 5, 6, 65535], dtype="<u2").tofile(tokens)
    return model, tokens


@pytest.mark.parametrize("backend", ["float", "intsim"])
def test_loss_matches_manual(tmp_path, backend):
    _, tokens = make_run(tmp_path)
    model, _, sim, params = ev.load_run(tmp_path, backend == "intsim")
    total = 0
    with torch.no_grad():
        for ids in ([1, 2, 3, 4], [4, 5, 6]):
            x = torch.tensor([ids[:-1]])
            logits = model(x) if sim is None else sim.float_logits(*sim.forward(params, x))
            for pos, target in enumerate(ids[1:]):
                row = logits[0, pos].double()
                total += (torch.logsumexp(row, 0) - row[target]).item()
    result = ev.evaluate(tmp_path, tokens, 5, backend, offset=1)
    assert result["loss"] == pytest.approx(total / 5, abs=1e-12)
    assert result["perplexity"] == pytest.approx(math.exp(total / 5))
    assert (result["tokens"], result["windows"]) == (5, 2)


def test_bounds_and_fp32(tmp_path, capsys):
    _, tokens = make_run(tmp_path, "fp32")
    ev.main([str(tmp_path), str(tokens), "--max-tokens", "2"])
    assert '"tokens": 2' in capsys.readouterr().out
    with pytest.raises(ValueError, match="binary/ternary"):
        ev.evaluate(tmp_path, tokens, 2, "intsim")
    with pytest.raises(ValueError, match="vocabulary"):
        ev.evaluate(tmp_path, tokens, 7)
    for cap, offset in [(0, 0), (1, -1), (1, 20)]:
        with pytest.raises(ValueError):
            list(ev.token_windows(tokens, 3, cap, offset))
    tokens.write_bytes(b"x")
    with pytest.raises(ValueError, match="incomplete"):
        list(ev.token_windows(tokens, 3, 1))
