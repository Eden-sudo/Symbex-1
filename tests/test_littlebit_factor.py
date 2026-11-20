"""CPU tests for the trainable binary factor layer."""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit.py"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


littlebit = load("littlebit_test")
FactorLinear = littlebit.FactorLinear


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7), (0, 7), (2, 0, 7)])
def test_forward(residual, shape, monkeypatch):
    layer = FactorLinear(7, 11, 3, residual)
    x = torch.randn(*shape)
    expected = x @ layer.dense().T
    monkeypatch.setattr(layer, "dense", lambda: (_ for _ in ()).throw(AssertionError("dense called")))
    torch.testing.assert_close(layer(x), expected, atol=1e-5, rtol=1e-5)
    x = torch.randn(7, 4).T
    assert not x.is_contiguous()
    other = FactorLinear(7, 11, 3, residual)
    torch.testing.assert_close(other(x), x @ other.dense().T, atol=1e-5, rtol=1e-5)


def test_manual_zero_and_paths():
    layer = FactorLinear(2, 2, 1, True)
    with torch.no_grad():
        for p in layer.paths:
            p.U.copy_(torch.tensor([[0.], [-1.]]))
            p.V.copy_(torch.tensor([[0.], [-1.]]))
            p.g.copy_(torch.tensor([2., 3.]))
            p.h.copy_(torch.tensor([4., 5.]))
            p.l.fill_(2.)
    expected = torch.tensor([[32., -40.], [-48., 60.]])
    torch.testing.assert_close(layer.dense(), expected)
    torch.testing.assert_close(layer(torch.tensor([1., 2.])), expected @ torch.tensor([1., 2.]))


def test_gradients_and_state():
    z = torch.tensor([-1., 0., 1.], requires_grad=True)
    littlebit._binary(z).sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))
    with torch.random.fork_rng():
        torch.manual_seed(42)
        layer = FactorLinear(7, 11, 3, True)
        x = torch.randn(4, 7, requires_grad=True)
        (layer(x) * torch.randn(4, 11)).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    assert len(list(layer.parameters())) == 10
    for path in layer.paths:
        for name in ("U", "V", "g", "h", "l"):
            grad = getattr(path, name).grad
            assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    copy = FactorLinear(7, 11, 3, True)
    copy.load_state_dict(layer.state_dict())
    torch.testing.assert_close(copy(x), layer(x))


def oracle(target, rank):
    left, s, right = torch.linalg.svd(target, full_matrices=False)
    u, v = left[:, :rank] * s[:rank].sqrt(), right[:rank].T * s[:rank].sqrt()
    l = u.abs().mean(0) * v.abs().mean(0)
    b = (torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u)) * l) @ torch.where(
        v >= 0, torch.ones_like(v), -torch.ones_like(v)).T
    denom = b.square().sum(1)
    g = torch.where(denom > 0, (target * b).sum(1) / denom.clamp_min(1e-30), 0.)
    return u, v, l, g, b


def test_initialization_oracle():
    generator = torch.Generator().manual_seed(91)
    w = torch.randn(9, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    original = w.detach().clone()
    one = FactorLinear.from_dense(w, 3)
    two = FactorLinear.from_dense(w, 3, True)
    target = original
    for path in two.paths:
        u, v, l, g, b = oracle(target, 3)
        # Singular vectors may flip signs together.
        signs = torch.where((path.U * u).sum(0) >= 0, 1., -1.)
        torch.testing.assert_close(path.U, u * signs)
        torch.testing.assert_close(path.V, v * signs)
        torch.testing.assert_close(path.l, l)
        torch.testing.assert_close(path.h, torch.ones_like(path.h))
        torch.testing.assert_close(path.g, g)
        reconstruction = g[:, None] * b
        torch.testing.assert_close(((target - reconstruction) * b).sum(1), torch.zeros(9, dtype=w.dtype), atol=1e-12, rtol=0)
        target = target - reconstruction
    torch.testing.assert_close(two.dense(), original - target)
    for a, b in zip(one.paths[0].parameters(), two.paths[0].parameters()):
        torch.testing.assert_close(a, b)
    assert torch.equal(w.detach(), original)
    assert all(p.is_leaf and p.dtype == w.dtype and p.device == w.device for p in two.parameters())


def test_approximation():
    gen = torch.Generator().manual_seed(123)
    w = torch.randn(128, 64, generator=gen) @ torch.randn(64, 160, generator=gen) + 0.01 * torch.randn(128, 160, generator=gen)
    errors = [(w - FactorLinear.from_dense(w, r).dense()).norm().item() / w.norm().item() for r in (8, 16, 32, 64)]
    residual = (w - FactorLinear.from_dense(w, 16, True).dense()).norm().item() / w.norm().item()
    print("approximation errors:", errors, "residual rank 16:", residual)
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert residual < errors[1]


def test_storage():
    layer = FactorLinear(384, 1024, 64)
    assert layer.storage_bytes() == 14208
    assert layer.bits_per_weight() == 0.2890625
    assert layer.storage_bytes(32) == 17152
    assert FactorLinear(384, 1024, 64, True).storage_bytes() == 28416
    assert FactorLinear(7, 11, 3).storage_bytes() == 60
    for value, exc in [(True, TypeError), (1.5, TypeError), (0, ValueError), (7, ValueError)]:
        with pytest.raises(exc, match="scale_bits"):
            layer.storage_bytes(value)


def test_boundaries():
    for w in (torch.zeros(4, 5), torch.ones(4, 5), torch.randn(4, 5).half()):
        for rank in (1, 4):
            layer = FactorLinear.from_dense(w, rank, True)
            assert all(torch.isfinite(p).all() for p in layer.parameters())
            assert torch.isfinite(layer.dense()).all()
            if not w.any():
                assert not layer.dense().any()
    for args, exc, name in [((0, 2, 1), ValueError, "in_features"), ((2, -1, 1), ValueError, "out_features"),
                            ((2, 2, 0), ValueError, "rank"), ((2, 2, 3), ValueError, "rank"),
                            ((True, 2, 1), TypeError, "in_features"), ((2, 2, 1.0), TypeError, "rank"),
                            ((2, 2, 1, 1), TypeError, "residual")]:
        with pytest.raises(exc, match=name):
            FactorLinear(*args)
    for w, exc in [(None, TypeError), (torch.ones(2, 2, dtype=torch.int64), TypeError),
                   (torch.ones(2, 2, dtype=torch.complex64), TypeError), (torch.ones(2), ValueError),
                   (torch.empty(0, 2), ValueError), (torch.tensor([[float('nan')]]), ValueError),
                   (torch.tensor([[float('inf')]]), ValueError), (torch.eye(2).to_sparse(), ValueError)]:
        with pytest.raises(exc, match="W"):
            FactorLinear.from_dense(w, 1)
    layer = FactorLinear(2, 3, 1)
    for x, exc in [(None, TypeError), (torch.ones(2, dtype=torch.int64), TypeError),
                   (torch.tensor(1.), ValueError), (torch.ones(3), ValueError)]:
        with pytest.raises(exc, match="x"):
            layer(x)


def test_import_from_other_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = load("littlebit_other_cwd")
    assert module.FactorLinear(2, 3, 1)(torch.ones(2)).shape == (3,)
