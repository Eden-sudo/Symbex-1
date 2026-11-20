import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_values_and_gradients():
    m = load("littlebit_distill")
    s = torch.tensor([[[1., -2., 0.], [0., 2., 1.]]], dtype=torch.double, requires_grad=True)
    t = torch.tensor([[[0., 1., 2.], [1., 0., -1.]]], dtype=torch.double, requires_grad=True)
    sh = torch.ones(1, 2, 4, dtype=torch.double, requires_grad=True)
    th = torch.zeros_like(sh, requires_grad=True)
    p = (t.detach() / 2).softmax(-1)
    expected = (p * (p.log() - (s / 2).log_softmax(-1))).sum(-1).mean() * 4 + .3
    loss = m.distillation_loss(s, t, temperature=2, student_hidden=sh, teacher_hidden=th, hidden_weight=.3)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert s.grad.abs().sum() > 0 and sh.grad.abs().sum() > 0
    assert t.grad is None and th.grad is None
    assert torch.autograd.gradcheck(lambda x: m.distillation_loss(x, t, temperature=2), (s,))
    torch.testing.assert_close(m.distillation_loss(s, s), torch.zeros((), dtype=torch.double), atol=1e-12, rtol=0)
    for temp in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            m.distillation_loss(s, t, temperature=temp)
    with pytest.raises(ValueError):
        m.distillation_loss(s, t, hidden_weight=1)


def test_toy_model_opt_in_hidden():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train, kd = load("train"), load("littlebit_distill")
        model = train.TinyLM(8, 4, 1, 1, 8, 2, "fp32")
        teacher = train.TinyLM(8, 4, 1, 1, 8, 2, "fp32")
        x = torch.tensor([[1, 2]])
        logits, hidden = model(x, return_hidden=True)
        tl, th = teacher(x, return_hidden=True)
        torch.testing.assert_close(model(x), logits)
        kd.distillation_loss(logits, tl, student_hidden=hidden, teacher_hidden=th, hidden_weight=.1).backward()
        assert model.blocks[0].q.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in teacher.parameters())
    finally:
        torch.set_num_threads(previous)
