import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("sensitivity", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit_sensitivity.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_probe_matches_direct_error_and_preserves_model():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 8, bias=False)).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    bpws = [m.lb.bits_per_weight(r, 8, 8) for r in (1, 2, 3)]
    result = m.probe(model, bpws)
    for name, options in result.items():
        for c in options:
            layer = m.lb.FactorLinear.from_dense(model[int(name)].weight, c["rank"])
            assert c["bits"] == 8 * layer.storage_bytes()
            assert c["error"] == pytest.approx(float((layer.dense() - model[int(name)].weight).double().square().sum().detach()))
    assert not model.training
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    selection, used = m.allocate(result, sum(c[0]["bits"] for c in result.values()))
    assert used == sum(c["bits"] for c in selection.values())


def test_exact_constructed_budget_beats_uniform_and_rejects_infeasible():
    # Equal rank costs, unequal sensitivity: rank 3 + rank 1 beats rank 2 + rank 2.
    curves = {"sensitive": [100., 60., 0.], "robust": [2., 1., 0.]}
    candidates = {name: [{"rank": r, "bits": r * 8, "error": error}
                          for r, error in enumerate(errors, 1)] for name, errors in curves.items()}
    result, used = m.allocate(candidates, 32)
    assert used == 32 and sum(c["bits"] for c in result.values()) == 32
    assert sum(c["error"] for c in result.values()) < 61
    assert result["sensitive"]["rank"] == 3
    assert m.allocate(candidates, 33)[1] == 32
    with pytest.raises(ValueError, match="cannot fit"):
        m.allocate(candidates, 15)
