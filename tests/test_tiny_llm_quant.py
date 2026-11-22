"""Check the tiny language model quantizers on CPU."""
import importlib.util
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parent.parent
KINDS = ("binary", "ternary", "int4", "int2")


@pytest.fixture(scope="module")
def tiny_llm():
    path = REPO / "benchmarks" / "experiments" / "tiny_llm" / "train.py"
    spec = importlib.util.spec_from_file_location("tiny_llm_quant_under_test", path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"tiny_llm requires missing dependency: {exc.name}") from exc
    return mod


def random_tensor(shape):
    generator = torch.Generator(device="cpu").manual_seed(12345)
    return torch.randn(shape, generator=generator, device="cpu")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(1, 1), (3, 17), (4, 513)])
def test_quant_weight_properties(tiny_llm, kind, shape):
    w = random_tensor(shape) * torch.tensor([0.25, 1, 4, 16], device="cpu")[:shape[0], None]
    actual = tiny_llm.quant_weight(w, kind)
    assert actual.shape == w.shape and actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    if kind == "binary":
        codes = torch.where(w >= 0, 1.0, -1.0)
    elif kind == "ternary":
        codes = (w / scale).round().clamp(-1, 1)
    elif kind == "int4":
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / 7
        codes = (w / scale).round().clamp(-7, 7)
    else:
        codes = (w / scale).floor().clamp(-2, 1) + 0.5
    torch.testing.assert_close(actual, codes * scale, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual / scale, codes, rtol=1e-5, atol=1e-5)
    if kind == "int4":
        assert all(row.round().unique().numel() <= 15 for row in actual / scale)
        torch.testing.assert_close(actual.abs().amax(dim=1), w.abs().amax(dim=1))


@pytest.mark.parametrize("kind, expected", [
    ("binary", [[1e-5, 1e-5, 1e-5], [-1e-5, 1e-5, 1e-5]]),
    ("ternary", [[0, 0, 0], [0, 0, 0]]),
    ("int4", [[0, 0, 0], [-1e-5 / 7, 0, 1e-5 / 7]]),
    ("int2", [[5e-6, 5e-6, 5e-6], [-5e-6, 5e-6, 5e-6]]),
])
def test_quant_weight_boundaries(tiny_llm, kind, expected):
    w = torch.tensor([[0, 0, 0], [-1e-6, 0, 1e-6]], device="cpu")
    torch.testing.assert_close(tiny_llm.quant_weight(w, kind), torch.tensor(expected, dtype=w.dtype, device="cpu"),
                               rtol=1e-5, atol=1e-10)
    if kind == "int4":
        w = torch.tensor([[-7, -2.5, -1.5, 0, 1.5, 2.5, 7]], device="cpu")
        expected = [[-7, -2, -2, 0, 2, 2, 7]]
    else:
        w = torch.tensor([[-2, -1, -0.5, 0, 0.5, 1, 2]], device="cpu")
        expected = {
            "binary": [[-1, -1, -1, 1, 1, 1, 1]],
            "ternary": [[-1, -1, 0, 0, 0, 1, 1]],
            "int2": [[-1.5, -0.5, -0.5, 0.5, 0.5, 1.5, 1.5]],
        }[kind]
    torch.testing.assert_close(tiny_llm.quant_weight(w, kind), torch.tensor(expected, dtype=w.dtype, device="cpu"))


@pytest.mark.parametrize("shape", [(1,), (3, 17), (2, 3, 513)])
def test_quant_act8_properties(tiny_llm, shape):
    x = random_tensor(shape)
    factors = torch.arange(1, x.numel() // shape[-1] + 1, device="cpu").reshape(*shape[:-1], 1)
    x = x * factors
    actual = tiny_llm.quant_act8(x)
    assert actual.shape == x.shape and actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    scale = 127 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    codes = (x * scale).round().clamp(-128, 127)
    torch.testing.assert_close(actual, codes / scale, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual * scale, codes, rtol=1e-5, atol=1e-4)
    assert codes.min() >= -127 and codes.max() <= 127
    assert all(row.round().unique().numel() <= 255 for row in (actual * scale).reshape(-1, shape[-1]))
    torch.testing.assert_close(actual.abs().amax(dim=-1), x.abs().amax(dim=-1))


def test_quant_act8_boundaries(tiny_llm):
    x = torch.tensor([[0, 0, 0], [-1e-8, 0, 1e-8]], device="cpu")
    actual = tiny_llm.quant_act8(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(x), rtol=0, atol=1e-10)
    x = torch.tensor([[-127, -2.5, -1.5, 0, 1.5, 2.5, 127]], device="cpu")
    expected = torch.tensor([[-127, -2, -2, 0, 2, 2, 127]], dtype=x.dtype, device="cpu")
    torch.testing.assert_close(tiny_llm.quant_act8(x), expected)


@pytest.mark.parametrize("shape", [(1,), (3, 17), (2, 3, 7)])
def test_ste_forward_and_backward(tiny_llm, shape):
    generator = torch.Generator(device="cpu").manual_seed(12345)
    x = torch.randn(shape, generator=generator, device="cpu", requires_grad=True)
    q = torch.randn(shape, generator=generator, device="cpu", requires_grad=True)
    grad = torch.randn(shape, generator=generator, device="cpu")
    actual = tiny_llm.ste(x, q)
    torch.testing.assert_close(actual, q, rtol=1e-5, atol=1e-6)
    actual.backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
    assert q.grad is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape, n_out", [((7,), 3), ((2, 7), 5), ((2, 3, 11), 4)])
def test_qlinear_forward_and_backward(tiny_llm, kind, shape, n_out):
    layer = tiny_llm.QLinear(shape[-1], n_out, kind).cpu()
    actual = layer(random_tensor(shape))
    assert actual.shape == (*shape[:-1], n_out) and actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    actual.square().sum().backward()
    grad = layer.weight.grad
    assert grad is not None and grad.shape == layer.weight.shape and grad.device.type == "cpu"
    assert torch.isfinite(grad).all()
