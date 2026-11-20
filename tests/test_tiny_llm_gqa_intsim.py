"""Tiny CPU checks for the additive GQA integer path."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


@pytest.fixture
def modules(monkeypatch):
    loaded = []
    for name in ("train", "intsim", "gqa_intsim"):
        spec = importlib.util.spec_from_file_location(name, DIR / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield loaded
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("seed", [3, 17])
def test_float_model_and_mha_exact(modules, kv_heads, seed):
    T, I, G = modules
    torch.manual_seed(seed)
    model = T.TinyLM(24, 256, 2, 4, 64, 7, "binary", kv_heads=kv_heads).eval()
    idx = torch.randint(24, (2, 7))
    P = G.integerize(model, "binary")
    z, hm = G.forward(P, idx)
    # Fixed absolute error budget on logits: quantized embedding, Q12 rounding,
    # int8 Q/K/V and table softmax versus the training model's float attention.
    torch.testing.assert_close(I.float_logits(z, hm), model(idx).double(), atol=0.06, rtol=0)
    if kv_heads == 4:
        expected = I.forward(P, idx)
        assert torch.equal(z, expected[0])
        assert torch.equal(hm, expected[1])


def test_attention_grouping_causality_and_zero(modules):
    _, I, G = modules
    torch.manual_seed(9)
    q = torch.randint(-9000, 9001, (2, 5, 4 * 64))
    k, v = (torch.randint(-9000, 9001, (2, 5, 2 * 64)) for _ in range(2))
    C = torch.full((5, 32), 1 << 14)
    S = torch.zeros_like(C)
    exp = torch.tensor(I.EXP_TABLE)
    expected = I.attention(q, k.view(2, 5, 2, 64).repeat_interleave(2, 2).flatten(2),
                           v.view(2, 5, 2, 64).repeat_interleave(2, 2).flatten(2), 4, C, S, exp)
    actual = G.attention(q, k, v, 4, 2, C, S, exp)
    assert torch.equal(actual, expected)
    k[:, 3:] *= -1
    v[:, 3:] *= -1
    assert torch.equal(actual[:, :3], G.attention(q, k, v, 4, 2, C, S, exp)[:, :3])
    assert not G.attention(q * 0, k * 0, v * 0, 4, 2, C, S, exp).any()


def test_invalid_shapes(modules):
    T, _, G = modules
    with pytest.raises(ValueError, match="64-wide"):
        G.integerize(T.TinyLM(8, 32, 1, 2, 32, 3, "binary"), "binary")
    with pytest.raises(ValueError, match="multiple"):
        G.validate_heads(4, 3)
