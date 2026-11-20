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


ternary, train = load("ternary"), load("train")


def test_codes_scales_and_identity_gradient():
    w = torch.tensor([[-2., -1, -.5, 0, .5, 1, 2], [0.] * 7], requires_grad=True)
    codes, scale = ternary.quantize(w)
    assert codes.dtype == torch.int8
    assert codes.tolist() == [[-1, -1, 0, 0, 0, 1, 1], [0] * 7]
    torch.testing.assert_close(scale, torch.tensor([[1.], [1e-5]]))
    q = ternary.quant_weight(w)
    torch.testing.assert_close(q, codes * scale)
    upstream = torch.arange(14.).reshape_as(w)
    (q * upstream).sum().backward()
    torch.testing.assert_close(w.grad, upstream, rtol=0, atol=0)
    assert not scale.requires_grad


def test_opt_in_linear_and_toy_learning():
    layer = train.QLinear(3, 2, "ternary")
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[.3, -.4, .05], [-.2, .1, .4]]))
    x = torch.eye(3)
    target = torch.tensor([[1., -1.], [-1., 0.], [0., 1.]])
    expected = train.quant_act8(x) @ ternary.quant_weight(layer.weight).T
    torch.testing.assert_close(layer(x), expected)
    optimizer = torch.optim.SGD(layer.parameters(), lr=.6)
    initial = (layer(x) - target).square().mean().item()
    for _ in range(12):
        optimizer.zero_grad()
        loss = (layer(x) - target).square().mean()
        loss.backward()
        assert torch.isfinite(layer.weight.grad).all()
        optimizer.step()
    assert (layer(x) - target).square().mean().item() < initial * .6
    # Existing binary behavior remains distinct: a zero gets a positive sign.
    assert train.quant_weight(torch.zeros(1, 3), "binary").gt(0).all()


@pytest.mark.parametrize("w", [torch.zeros(3), torch.zeros(1, 0), torch.ones(2, 3, dtype=torch.int8)])
def test_invalid_shape_or_dtype(w):
    with pytest.raises(ValueError, match="matrix"):
        ternary.quantize(w)
