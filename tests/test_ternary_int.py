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


integer, ternary, train = load("ternary_int"), load("ternary"), load("train")


@pytest.mark.parametrize("shape", [(17,), (3, 17), (2, 3, 17)])
def test_exact_accumulation_and_float_path(shape):
    g = torch.Generator().manual_seed(9)
    x = torch.randn(shape, generator=g)
    w = torch.randn(5, 17, generator=g)
    w[0].zero_()
    q, sx = integer.quantize_input(x)
    codes, sw = ternary.quantize(w)
    acc = integer.accumulate(q, codes)
    # Python unbounded integers are independent of tensor matmul arithmetic.
    oracle = [[sum(int(a) * int(b) for a, b in zip(row, weight))
               for weight in codes] for row in q.reshape(-1, 17)]
    assert acc.reshape(-1, 5).tolist() == oracle
    expected = train.quant_act8(x) @ train.quant_weight(w, "ternary").T
    torch.testing.assert_close(integer.dequantize(acc, sx, sw), expected, atol=3e-6, rtol=3e-6)


def test_extremes_zeros_and_errors():
    x = torch.tensor([-128, 127, -128, 127], dtype=torch.int8)
    w = torch.tensor([[-1, 1, -1, 1], [0, 0, 0, 0]], dtype=torch.int8)
    assert integer.accumulate(x, w).tolist() == [510, 0]
    q, scale = integer.quantize_input(torch.zeros(2, 4))
    assert q.eq(0).all() and torch.isfinite(scale).all()
    with pytest.raises(ValueError, match="int8"):
        integer.accumulate(x.float(), w)
    with pytest.raises(ValueError, match="ternary"):
        integer.accumulate(x, w + 1)
    with pytest.raises(ValueError, match="width"):
        integer.accumulate(x[:2], w)
    with pytest.raises(ValueError, match="scale shape"):
        integer.dequantize(torch.zeros(2), torch.ones(2), torch.ones(2, 1))
