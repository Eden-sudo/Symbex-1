import importlib.util
from pathlib import Path

import pytest

try:
    import torch
except ImportError as exc:
    raise ImportError("test_littlebit_itq requires PyTorch") from exc


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def itq():
    path = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit_itq.py"
    spec = importlib.util.spec_from_file_location("littlebit_itq", path)
    assert spec is not None and spec.loader is not None, "Cannot load Joint-ITQ module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_rotation(rank, seed):
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(rank, rank, dtype=torch.float64, generator=generator))
    return q * torch.where(r.diag() >= 0, 1.0, -1.0)


def independent_error(u, v, rotation):
    x, y = u @ rotation, v @ rotation
    sx = torch.where(x >= 0, 1.0, -1.0)
    sy = torch.where(y >= 0, 1.0, -1.0)
    target = u @ v.T
    approx = (x.abs().mean(1, keepdim=True) * sx) @ (y.abs().mean(1, keepdim=True) * sy).T
    return (torch.linalg.norm(target - approx) / torch.linalg.norm(target)).item()


@pytest.mark.parametrize("rank", [8, 32, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_factors(itq, rank, seed):
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(2 * rank, rank, generator=generator, dtype=torch.float64)
    v = torch.randn(3 * rank, rank, generator=generator, dtype=torch.float64)
    rotation = itq.joint_itq(u, v)
    identity = torch.eye(rank, dtype=torch.float64)
    torch.testing.assert_close(rotation @ rotation.T, identity, atol=1e-5, rtol=0)
    torch.testing.assert_close((u @ rotation) @ (v @ rotation).T, u @ v.T, atol=1e-4, rtol=0)
    rotations = (rotation, identity, random_rotation(rank, 1))
    errors = [itq.binarization_error(u, v, r) for r in rotations]
    for r, error in zip(rotations, errors):
        assert error == pytest.approx(independent_error(u, v, r), abs=1e-12)
    # Algorithm 1 minimizes ||sign(ZR)-ZR||, not the scaled product error.
    z = torch.cat((u, v))
    initial = itq.joint_itq(u, v, iters=0)
    objective = lambda r: (z @ r - torch.where(z @ r >= 0, 1., -1.)).square().sum()
    assert objective(rotation) < objective(initial)


def itq_objective(u, v, rotation):
    z = torch.cat((u, v)) @ rotation
    return torch.linalg.norm(torch.where(z >= 0, 1.0, -1.0) - z).item()


@pytest.mark.parametrize("rank", [8, 32, 64])
def test_objective_never_increases(itq, rank):
    generator = torch.Generator().manual_seed(rank)
    u = torch.randn(2 * rank, rank, generator=generator, dtype=torch.float64)
    v = torch.randn(3 * rank, rank, generator=generator, dtype=torch.float64)
    values = [itq_objective(u, v, itq.joint_itq(u, v, iters=k)) for k in range(0, 21)]
    steps = [(k, b - a) for k, (a, b) in enumerate(zip(values, values[1:])) if b > a + 1e-9]
    assert not steps, f"rank {rank}: objective rose at iterations {steps}; values {values}"
    assert values[-1] < values[0], f"rank {rank}: no progress from the start rotation, {values[0]} -> {values[-1]}"


@pytest.mark.parametrize("rank", [8, 32, 64])
def test_mean_error_beats_identity(itq, rank):
    # ITQ minimizes its own objective, so a single case can lose to the identity
    # (rank 8 seed 1: 0.682 vs 0.665); over 8 seeds the mean must still be lower.
    identity = torch.eye(rank, dtype=torch.float64)
    cases = []
    for seed in range(8):
        generator = torch.Generator().manual_seed(seed)
        u = torch.randn(2 * rank, rank, generator=generator, dtype=torch.float64)
        v = torch.randn(3 * rank, rank, generator=generator, dtype=torch.float64)
        cases.append((seed, itq.binarization_error(u, v, itq.joint_itq(u, v)),
                      itq.binarization_error(u, v, identity)))
    rotated = sum(c[1] for c in cases) / len(cases)
    plain = sum(c[2] for c in cases) / len(cases)
    assert rotated < plain, f"rank {rank}: mean {rotated:.4f} vs identity {plain:.4f}; cases {cases}"


def test_one_update(itq):
    u = torch.tensor([[1., 2., 0.], [0., -1., 3.]], dtype=torch.float64)
    v = torch.tensor([[2., -1., 1.], [0., 0., 0.], [1., 3., 2.]], dtype=torch.float64)
    z = torch.cat((u, v))
    initial = random_rotation(3, 7)
    binary = torch.where(z @ initial >= 0, 1., -1.).double()
    left, _, right = torch.linalg.svd(z.T @ binary, full_matrices=False)
    expected = left @ right
    torch.testing.assert_close(itq.joint_itq(u, v, iters=0, seed=7), initial)
    actual = itq.joint_itq(u, v, iters=1, seed=7)
    torch.testing.assert_close(actual, expected)
    assert torch.linalg.norm(binary - z @ actual) <= torch.linalg.norm(binary - z @ initial) + 1e-12


def test_determinism_and_inputs(itq):
    u = torch.arange(48, dtype=torch.float32).reshape(6, 8)[:, ::2]
    v = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
    before_u, before_v = u.clone(), v.clone()
    state = torch.random.get_rng_state()
    threads = torch.get_num_threads()
    first = itq.joint_itq(u, v, seed=3)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(8)
    state = torch.random.get_rng_state()
    second = itq.joint_itq(u, v, seed=3)
    assert torch.equal(first, second) and torch.equal(state, torch.random.get_rng_state())
    assert first.dtype == u.dtype and first.device.type == "cpu"
    assert isinstance(itq.binarization_error(u, v, first), float)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(u, before_u) and torch.equal(v, before_v)
    assert torch.get_num_threads() == threads


def test_error_edges(itq):
    u = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
    v = torch.tensor([[0., 1.], [1., 2.], [-1., 0.]], dtype=torch.float64)
    r = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    assert itq.binarization_error(u, v, r) == pytest.approx(independent_error(u, v, r))
    assert itq.binarization_error(torch.tensor([[2.]]), torch.tensor([[-3.]]), torch.eye(1)) == 0
    assert itq.binarization_error(torch.zeros(2, 2), torch.zeros(3, 2), torch.eye(2)) == 0
    # Orthogonal original rows can have a nonzero binary approximation.
    assert itq.binarization_error(torch.tensor([[1., 1., 2.]]), torch.tensor([[1., 1., -1.]]), torch.eye(3)) == float("inf")
    rotation = itq.joint_itq(torch.zeros(2, 3), torch.tensor([[1., 0., 0.]]))
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(3), atol=1e-5, rtol=0)


def test_import_from_other_cwd(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/littlebit_itq.py"
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("cwd_itq", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.joint_itq)


@pytest.mark.parametrize("u,v,exception,match", [
    (None, torch.ones(2, 2), TypeError, "U"),
    (torch.ones(2), torch.ones(2, 2), ValueError, "U"),
    (torch.ones(0, 2), torch.ones(2, 2), ValueError, "U"),
    (torch.ones(2, 2), torch.ones(2, 3), ValueError, "rank"),
    (torch.ones(2, 2), torch.ones(2, 2, dtype=torch.float64), ValueError, "dtype"),
    (torch.ones(2, 2, dtype=torch.int64), torch.ones(2, 2), TypeError, "U"),
    (torch.ones(2, 2, dtype=torch.complex64), torch.ones(2, 2), TypeError, "U"),
    (torch.full((2, 2), float("nan")), torch.ones(2, 2), ValueError, "finite"),
    (torch.empty(2, 2, device="meta"), torch.ones(2, 2), ValueError, "CPU"),
])
def test_invalid_factors(itq, u, v, exception, match):
    with pytest.raises(exception, match=match):
        itq.joint_itq(u, v)


@pytest.mark.parametrize("kwargs,exception,match", [
    ({"iters": True}, TypeError, "iters"), ({"iters": 1.5}, TypeError, "iters"),
    ({"iters": -1}, ValueError, "iters"), ({"seed": False}, TypeError, "seed"),
    ({"seed": -1}, ValueError, "seed"), ({"seed": 2**63}, ValueError, "seed"),
])
def test_invalid_options(itq, kwargs, exception, match):
    with pytest.raises(exception, match=match):
        itq.joint_itq(torch.ones(2, 2), torch.ones(2, 2), **kwargs)


@pytest.mark.parametrize("rotation,exception,match", [
    (None, TypeError, "R"), (torch.ones(2), ValueError, "shape"),
    (torch.ones(2, 2), ValueError, "orthogonal"),
    (torch.eye(2, dtype=torch.float64), ValueError, "dtype"),
    (torch.full((2, 2), float("inf")), ValueError, "finite"),
    (torch.empty(2, 2, device="meta"), ValueError, "CPU"),
])
def test_invalid_rotation(itq, rotation, exception, match):
    with pytest.raises(exception, match=match):
        itq.binarization_error(torch.ones(2, 2), torch.ones(2, 2), rotation)
