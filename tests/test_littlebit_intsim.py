import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lb, integer = load("littlebit"), load("littlebit_int")


@pytest.mark.parametrize("residual", [False, True])
def test_float_agreement(residual):
    generator = torch.Generator().manual_seed(42)
    layer = lb.FactorLinear.from_dense(torch.randn(7, 9, generator=generator) * 0.2, 3, residual)
    # Exercise learned negative/nonuniform scales, not just initializer defaults.
    with torch.no_grad():
        for path in layer.paths:
            path.g[::2].neg_()
            path.l[::2].neg_()
            path.h[::2].neg_()
    params = integer.integerize(layer, input_bound=2)
    for _ in range(8):
        q = torch.randint(-127, 128, (9,), generator=generator)
        expected = layer(q.float() * params["input_scale"]).detach()
        result = integer.forward_int(params, q.tolist())
        assert result == integer.forward_int(params, q.tolist())
        actual = torch.tensor(result) / 4096
        # Error budget includes two int8 rounding stages and Q20/Q12 rounding.
        torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.04)


def test_zero_and_rejections():
    layer = lb.FactorLinear.from_dense(torch.zeros(3, 4), 2, True)
    params = integer.integerize(layer)
    assert integer.forward_int(params, [-128, 0, 1, 127]) == [0, 0, 0]
    with pytest.raises(ValueError, match="x"):
        integer.forward_int(params, [128] * 4)
    for bound in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            integer.integerize(layer, bound)
    params["paths"][0]["m2"] = [2**31 - 1] * 3
    params["rank"] = 2
    assert integer._round_shift(-3, 1) == -2
    assert integer._requantize(-128, 1 << 20, 20) == -127
