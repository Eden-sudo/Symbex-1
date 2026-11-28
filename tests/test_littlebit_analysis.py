import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit_analysis.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_baseline_arithmetic():
    row = a.estimate()
    weights = 8 * (4 * 384 * 384 + 3 * 384 * 1024)
    outputs = 8 * (5 * 384 + 2 * 1024)
    assert row["linear_weights"] == weights == 14155776
    assert row["linear_payload_bytes"] == weights // 8 + 2 * outputs
    assert row["integer_weight_bytes"] == weights // 8 + 4 * outputs + 384 * 4096 + 4 * 4096
    assert row["estimated_ops_per_token"] == weights + outputs + 2 * 384 * 4096 + 4096


def test_factored_arithmetic():
    row = a.estimate(0.3)
    assert row["ranks"] == [40, 40, 40, 40, 64, 64, 65]
    # Orientation changes padding: w2 can fit one extra latent channel.
    square = 768 * 40 // 8 + 2 * (768 + 40)
    rectangle = 1408 * 64 // 8 + 2 * (1408 + 64)
    down = 384 * 9 + 65 * 128 + 2 * (1408 + 65)
    assert row["linear_payload_bytes"] == 8 * (4 * square + 2 * rectangle + down)
    adds = 8 * (4 * 768 * 40 + 2 * 1408 * 64 + 1408 * 65)
    scales = 8 * (4 * 808 + 2 * 1472 + 1473)
    assert row["linear_signed_adds"] == adds
    assert row["linear_scale_multiplies"] == scales
    assert row["integer_weight_bytes"] == row["linear_payload_bytes"] + 2 * scales + 56 * 40 + 384 * 4096 + 4 * 4096
    assert row["estimated_ops_per_token"] == adds + scales + 2 * 384 * 4096 + 4096
    assert row["integer_weight_bytes"] < a.estimate()["integer_weight_bytes"]
    with pytest.raises(ValueError, match="rank one"):
        a.estimate(0.1, residual=True)


def test_script(capsys):
    a.main([])
    output = capsys.readouterr().out
    assert "Estimates only" in output and "1-bit" in output
    assert all(str(target) in output for target in (0.1, 0.3, 0.5))
    a.main(["--residual", "--targets", "0.1", "0.5"])
    assert "infeasible" in capsys.readouterr().out
