import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("quality", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit_quality.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_script_runs_and_reports_all_combinations(capsys):
    rows = m.main(["--size", "128"])
    assert len(rows) == 17
    text = capsys.readouterr().out
    assert "weight_MSE output_MSE" in text and "infeasible" in text
    assert {r["target_bpw"] for r in rows[1:]} == {.1, .3, .55, 1.}
    assert any(r["rank"] is not None for r in rows[1:])


def test_constructed_rank_one_case_is_monotone_with_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Exact sign rank-one teacher, not a claim about random teachers.
        weight = torch.ones(512, 512, dtype=torch.double) / 512
        inputs = torch.ones(2, 512, dtype=torch.double)
        # Residual rank one does not fit 0.1 BPW at this size: honest infeasibility.
        rows = m.compare(weight, inputs, itq_iters=1)
        for residual in (False, True):
            for itq in (False, True):
                group = [r for r in rows[1:] if r["residual"] == residual and r["itq"] == itq
                         and r["rank"] is not None]
                assert len(group) >= 3
                for metric in ("weight_mse", "output_mse"):
                    assert all(b[metric] <= a[metric] + 1e-20 for a, b in zip(group, group[1:]))
                assert all(r["actual_bpw"] <= r["target_bpw"] for r in group)
    finally:
        torch.set_num_threads(previous)
