import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("lb", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit.py")
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


@pytest.mark.parametrize("iters", [None, 5])
def test_residual_fits_error(iters):
    w = torch.randn(16, 12, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    one = lb.FactorLinear.from_dense(w, 3, itq_iters=iters)
    two = lb.FactorLinear.from_dense(w, 3, True, itq_iters=iters)
    primary = {k: v.clone() for k, v in two.paths[0].state_dict().items()}
    assert (w - two.dense()).norm() < (w - one.dense()).norm()
    expected = lb.FactorLinear.from_dense(w - one.dense().detach(), 3, itq_iters=iters)
    for key, value in expected.paths[0].state_dict().items():
        torch.testing.assert_close(two.paths[1].state_dict()[key], value)
    two.fit_residual_(w, itq_iters=iters)
    for key, value in primary.items():
        assert torch.equal(two.paths[0].state_dict()[key], value)
    with pytest.raises(ValueError, match="no residual"):
        one.fit_residual_(w)
    with pytest.raises(ValueError, match="shape"):
        two.fit_residual_(w.T)


def test_zero_residual():
    w = torch.zeros(4, 5)
    assert torch.equal(lb.FactorLinear.from_dense(w, 2, True).dense(), w)
