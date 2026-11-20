import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("budget", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit_budget.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_exact_payloads():
    assert b.storage_bytes(64, 384, 1024) == 14208
    assert b.bits_per_weight(64, 384, 1024) == 0.2890625
    assert b.bits_per_weight(64, 384, 1024, True) == 0.578125
    assert b.storage_bytes(3, 7, 11) == 56
    assert b.storage_bytes(64, 384, 1024, scale_bits=32) == 17152


@pytest.mark.parametrize("residual", [False, True])
def test_inverse(residual):
    for rank in range(1, 14):
        target = b.bits_per_weight(rank, 13, 17, residual)
        assert b.rank_for_bpw(target, 13, 17, residual) == rank
        if rank > 1:
            assert b.rank_for_bpw(target - 1e-8, 13, 17, residual) == rank - 1
    assert b.rank_for_bpw(100, 13, 17, residual) == 13
    with pytest.raises(ValueError, match="rank one"):
        b.rank_for_bpw(0.001, 13, 17, residual)


@pytest.mark.parametrize("target", [float("nan"), float("inf"), True, "1"])
def test_bad_target(target):
    with pytest.raises(ValueError):
        b.rank_for_bpw(target, 13, 17)
