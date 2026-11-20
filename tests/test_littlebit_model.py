import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("lbmodel", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit_model.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_tiny_model_training():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = m.lb._train.TinyLM(16, 8, 2, 2, 12, 4, "fp32")
            q, emb = model.blocks[0].q, model.emb
            m.replace_linears(model, rank=3, residual=True, itq_iters=3)
            assert model.blocks[0].q is q and model.emb is emb
            assert all(isinstance(b.w1, m.FactorAct8Linear) for b in model.blocks)
            x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
            target = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            losses = []
            for _ in range(8):
                optimizer.zero_grad()
                logits = model(x)
                assert logits.shape == (2, 4, 16)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
                losses.append(loss.item())
                loss.backward()
                for block in model.blocks:
                    for parameter in block.w1.parameters():
                        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        assert parameter.grad.abs().sum() > 0
                optimizer.step()
            assert losses[-1] < losses[0]
    finally:
        torch.set_num_threads(previous)


def test_reject_without_partial_mutation():
    model = m.lb._train.TinyLM(16, 8, 2, 2, 12, 4, "binary")
    old = model.blocks[0].w1
    model.blocks[1].w1.acts = 1
    with pytest.raises(ValueError, match="activations"):
        m.replace_linears(model, rank=3)
    assert model.blocks[0].w1 is old
