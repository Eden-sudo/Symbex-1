import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("factorized_head", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/factorized_head.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_shapes_formula_and_gradient():
    with torch.random.fork_rng():
        torch.manual_seed(3)
        head = m.FactorizedHead(8, 16, 3).double()
        h = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
        assert head(h).shape == (2, 4, 16)
        torch.testing.assert_close(head(h), h @ (head.A @ head.B))
        assert torch.autograd.gradcheck(head, (h,), fast_mode=True)
        head(h).square().sum().backward()
        for p in (h, head.A, head.B):
            assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_toy_training_and_reload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(11)
            args = (16, 8, 1, 2, 12, 4, "fp32")
            model = m.FactorizedTinyLM(*args, rank=3)
            baseline = m.train.TinyLM(*args)
            assert not hasattr(baseline, "output_head")
            assert model.emb.weight.data_ptr() != model.output_head.B.data_ptr()
            x = torch.tensor([[1, 2, 3, 4]])
            target = torch.tensor([2, 3, 4, 5])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            losses = []
            for _ in range(3):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 16), target)
                losses.append(loss.item())
                loss.backward()
                for p in (model.output_head.A, model.output_head.B, model.emb.weight):
                    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                optimizer.step()
            assert losses[-1] < losses[0]
            restored = m.FactorizedTinyLM(*args, rank=3)
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("rank", [0, -1, 9, 1.5, True])
def test_invalid_rank(rank):
    with pytest.raises(ValueError):
        m.FactorizedHead(8, 16, rank)


def test_reject_quantized_head():
    with pytest.raises(ValueError, match="fp32"):
        m.FactorizedTinyLM(16, 8, 1, 2, 12, 4, "binary", head="int4", rank=3)
