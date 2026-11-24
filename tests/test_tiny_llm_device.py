"""Device policy without a GPU; path imports must also work outside the experiment directory."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(scope="module")
def train():
    path = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/train.py"
    spec = importlib.util.spec_from_file_location("tiny_device_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("mps", [False, True])
@pytest.mark.parametrize("allow_mps", [False, True])
def test_pick_device(train, monkeypatch, cuda, mps, allow_mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    mps_available = Mock(return_value=mps)
    monkeypatch.setattr(torch.backends.mps, "is_available", mps_available)
    expected = "cuda" if cuda else "mps" if allow_mps and mps else "cpu"
    assert train.pick_device(allow_mps=allow_mps) == expected
    if cuda or not allow_mps:
        mps_available.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "mps", torch.device("cpu"), torch.device("mps")])
def test_non_cuda_autocast_is_noop(train, monkeypatch, device):
    autocast = Mock(side_effect=AssertionError("non-CUDA must not enter torch.autocast"))
    monkeypatch.setattr(torch, "autocast", autocast)
    x = torch.ones(2, 2, dtype=torch.float32)
    with train.autocast(device):
        assert (x @ x).dtype == torch.float32
    autocast.assert_not_called()


@pytest.mark.parametrize("device", ["cuda", torch.device("cuda:1")])
def test_cuda_autocast_unchanged(train, monkeypatch, device):
    autocast = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch, "autocast", autocast)
    with train.autocast(device):
        pass
    autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("device, expected", [("cuda", "cuda"), ("cuda:1", "cuda:1"), ("mps", "cpu"), ("cpu", "cpu")])
def test_data_device_keeps_splits_off_mps(train, device, expected):
    assert train.data_device(device) == expected


def test_split_from_disk_matches_in_memory_and_batches_move_to_device(train, tmp_path, monkeypatch):
    tokens = (torch.arange(5000) * 7 % 4096).numpy().astype("uint16")
    tokens.tofile(tmp_path / "val.bin")
    monkeypatch.setattr(train, "DATA", str(tmp_path))
    in_memory = train.load_split("val", "cpu")
    from_disk = train.load_split("val", "cpu", from_disk=True)
    assert from_disk.dtype == in_memory.dtype == torch.int16
    assert torch.equal(from_disk, in_memory)
    gen_a, gen_b = (torch.Generator().manual_seed(0) for _ in range(2))
    xa, ya = train.batch(in_memory, 4, 16, gen_a)
    xb, yb = train.batch(from_disk, 4, 16, gen_b, torch.device("cpu"))
    assert torch.equal(xa, xb) and torch.equal(ya, yb) and xb.dtype == torch.int64
    assert torch.equal(xb[:, 1:], yb[:, :-1])
