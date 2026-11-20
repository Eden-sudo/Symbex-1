import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_llm_eval_loss import make_run

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"
spec = importlib.util.spec_from_file_location("agreement", ROOT / "eval_agreement.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_report_matches_direct_model_outputs(tmp_path, capsys):
    _, tokens = make_run(tmp_path)
    model, _, sim, params = ev.loss.load_run(tmp_path, True)
    agree, differences = [], []
    with torch.no_grad():
        for ids in ([8, 1, 2], [3, 4]):
            x = torch.tensor([ids])
            f = model(x).double()
            i = sim.float_logits(*sim.forward(params, x))
            agree.extend((f.argmax(-1) == i.argmax(-1)).flatten().tolist())
            differences.append(float((f - i).abs().max()))
    result = ev.report(tmp_path, tokens, 2, 5)
    assert result["top1_matches"] == sum(agree)
    assert result["top1_agreement"] == sum(agree) / 5
    assert result["max_logit_difference"] == max(differences)
    assert result["max_logit_difference"] > 0
    assert (result["tokens"], result["windows"]) == (5, 2)
    ev.main([str(tmp_path), str(tokens), "--windows", "1", "--max-tokens", "6"])
    assert json.loads(capsys.readouterr().out)["tokens"] == 3


def test_short_file_and_invalid_caps(tmp_path):
    _, tokens = make_run(tmp_path)
    tokens.write_bytes(tokens.read_bytes()[:6])
    assert ev.report(tmp_path, tokens, 10, 30)["tokens"] == 2
    for windows, cap in [(0, 3), (1, 0), (-1, 4)]:
        with pytest.raises(ValueError):
            ev.report(tmp_path, tokens, windows, cap)
