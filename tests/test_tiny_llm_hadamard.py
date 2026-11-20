"""CPU contract tests for tiny LLM Hadamard rotations."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


TINY_LLM_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "experiments" / "tiny_llm"


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_llm_modules(monkeypatch):
    train = _load("_tiny_llm_hadamard_train", TINY_LLM_DIR / "train.py", monkeypatch)
    monkeypatch.setitem(sys.modules, "train", train)
    intsim = _load("_tiny_llm_hadamard_intsim", TINY_LLM_DIR / "intsim.py", monkeypatch)
    assert intsim.T is train
    return train, intsim


def _input(n, layout):
    shape = ((n,), (2, 3 * n), (2, 3, 2 * n))[layout]
    generator = torch.Generator(device="cpu").manual_seed(1729)
    return torch.randint(-8, 9, shape, dtype=torch.int64, device="cpu", generator=generator)


@pytest.mark.parametrize("n", (1, 2, 4, 64))
def test_hadamard_is_orthonormal_and_symmetric(tiny_llm_modules, n):
    train, _ = tiny_llm_modules
    h = train.hadamard(n)
    assert h.shape == (n, n)
    assert h.device.type == "cpu"
    torch.testing.assert_close(h @ h.T, torch.eye(n, dtype=h.dtype, device="cpu"),
                               rtol=0, atol=1e-6)
    torch.testing.assert_close(h, h.T, rtol=0, atol=1e-6)


@pytest.mark.parametrize("n", (1, 2, 4, 64))
@pytest.mark.parametrize("layout", (0, 1, 2))
def test_hadamard_int_matches_block_reference(tiny_llm_modules, n, layout):
    train, intsim = tiny_llm_modules
    x = _input(n, layout)
    original = x.clone()
    h_int = torch.round(train.hadamard(n).double() * math.sqrt(n)).to(torch.int64)
    blocks = original.reshape(*original.shape[:-1], -1, n)
    expected = (blocks @ h_int).reshape(original.shape)
    actual = intsim.hadamard_int(x, n)
    assert actual.shape == original.shape
    assert actual.dtype == torch.int64
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)
    assert torch.equal(x, original)


@pytest.mark.parametrize("n", (1, 2, 4, 64))
@pytest.mark.parametrize("layout", (0, 1, 2))
def test_hadamard_int_twice_scales_input(tiny_llm_modules, n, layout):
    _, intsim = tiny_llm_modules
    x = _input(n, layout)
    original = x.clone()
    actual = intsim.hadamard_int(intsim.hadamard_int(x, n), n)
    assert actual.shape == original.shape
    assert actual.dtype == torch.int64
    assert actual.device.type == "cpu"
    assert torch.equal(actual, n * original)
    assert torch.equal(x, original)


def test_hadamard_int_zero_returns_input(tiny_llm_modules):
    _, intsim = tiny_llm_modules
    x = torch.arange(-15, 15, dtype=torch.int64, device="cpu").reshape(2, 3, 5)
    original = x.clone()
    assert intsim.hadamard_int(x, 0) is x
    assert torch.equal(x, original)
