"""Check the estimator against actual integer exports and S3 allocations."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "benchmarks" / "experiments" / "tiny_llm"
SCRIPT = DIR / "memory_budget.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sizing():
    return load("memory_budget", SCRIPT)


@pytest.fixture(scope="module")
def exporter():
    try:
        import numpy  # noqa: F401
        import torch
    except ImportError as exc:
        pytest.fail(f"memory budget export tests require torch and numpy: {exc}")
    train = load("tiny_train", DIR / "train.py")
    old = sys.modules.get("train")
    sys.modules["train"] = train
    try:
        intsim = load("tiny_intsim", DIR / "intsim.py")
    finally:
        if old is None:
            del sys.modules["train"]
        else:
            sys.modules["train"] = old
    return torch, train, intsim


CASES = [
    (64, 1, 1, 128, 8, 17, "binary", "fp32"),
    (64, 2, 1, 72, 7, 19, "ternary", "fp32"),
    (64, 1, 1, 96, 8, 17, "binary", "int4"),
    (128, 2, 2, 160, 9, 23, "ternary", "int4"),
    (64, 2, 1, 128, 8, 17, "binary", "int2"),
    (128, 1, 2, 136, 7, 19, "ternary", "int2"),
]
DEFAULT = (384, 8, 6, 1024, 256, 4096, "binary", "fp32")


@pytest.mark.parametrize("case", CASES)
def test_export_size(sizing, exporter, tmp_path, case):
    torch, train, intsim = exporter
    dim, layers, heads, hidden, seq, vocab, weights, head = case
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(42)
            model = train.TinyLM(vocab, dim, layers, heads, hidden, seq, weights, head).cpu().eval()
            assert all(t.device.type == "cpu" for t in list(model.parameters()) + list(model.buffers()))
            params = intsim.integerize(model, weights)
            path = tmp_path / "model.bin"
            intsim.write_model(params, {}, path)
        assert os.path.getsize(path) == sizing.budget(*case)["model_bin_bytes"]
    finally:
        torch.set_num_threads(threads)


def test_default(sizing):
    assert sizing.budget(*DEFAULT) == {
        "model_bin_bytes": 3550752, "linear_weight_bytes": 1769472,
        "embedding_head_bytes": 1589248, "kv_cache_bytes": 1671184,
        "per_token_internal_ram_bytes": 132228, "bits_per_linear_weight": 1,
    }


def test_padding_and_metadata(sizing):
    args = (64, 2, 1, 72, 7, 19)
    binary = sizing.budget(*args, "binary", "fp32")
    assert binary["linear_weight_bytes"] == 2 * (4 * 64 * 8 + 2 * 72 * 8 + 64 * 12)
    assert binary == sizing.budget(*args, "ternary", "fp32")
    packed = sizing.budget(*args, "binary", "int4")
    assert packed["model_bin_bytes"] - binary["model_bin_bytes"] == 2 * 7 * 16 - 19 * 64 // 2
    assert packed["bits_per_linear_weight"] == 1


def test_ram(sizing):
    a = sizing.budget(64, 2, 1, 72, 7, 19, "binary", "fp32")
    assert a["kv_cache_bytes"] == 2 * 2 * 7 * 64 + 8 * 2 * 7 + 16
    assert a["per_token_internal_ram_bytes"] == 20 * 64 + 21 * 72 + 14 + 512 * 9 + 12 + 8 * 19 + 4096
    b = sizing.budget(128, 3, 2, 64, 11, 23, "binary", "fp32")
    assert b["kv_cache_bytes"] == 2 * 3 * 11 * 128 + 8 * 3 * 11 * 2 + 16
    assert b["per_token_internal_ram_bytes"] == 20 * 128 + 21 * 128 + 22 + 512 * 16 + 20 + 8 * 23 + 4096


@pytest.mark.parametrize("index,value", [(i, v) for i in range(6) for v in (0, -1, True, 1.5, 2**31)])
def test_invalid_numbers(sizing, index, value):
    args = list(DEFAULT)
    args[index] = value
    with pytest.raises(ValueError):
        sizing.budget(*args)


@pytest.mark.parametrize("args", [(65, 1, 2, 8, 1, 1, "binary", "fp32"),
                                  (6, 1, 2, 8, 1, 1, "binary", "fp32"),
                                  (30, 1, 1, 8, 1, 1, "binary", "int4"),
                                  (32, 1, 1, 8, 1, 1, "binary", "int2"),
                                  (64, 1, 1, 8, 1, 1, "bad", "fp32"),
                                  (64, 1, 1, 8, 1, 1, "binary", "bad")])
def test_invalid_shapes(sizing, args):
    with pytest.raises(ValueError):
        sizing.budget(*args)


def test_small_and_pure(sizing, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = (2, 1, 1, 1, 1, 1, "binary", "fp32")
    assert sizing.budget(*args) == sizing.budget(*args)
    assert not list(tmp_path.iterdir())


def test_cli(sizing, tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, "-S", str(SCRIPT), *args], cwd=tmp_path,
                              capture_output=True, text=True)
    default = run()
    assert default.returncode == 0 and json.loads(default.stdout) == sizing.budget(*DEFAULT)
    flags = ["--dim", "64", "--layers", "1", "--heads", "1", "--hidden", "96", "--seq", "8",
             "--vocab", "17", "--weights", "binary", "--head", "int4"]
    explicit = run(*flags)
    assert explicit.returncode == 0 and json.loads(explicit.stdout) == sizing.budget(*CASES[2])
    for flags in (["--dim", "0"], ["--dim", "oops"], ["--head", "bad"], ["--dim", "3"], ["--unknown"]):
        result = run(*flags)
        assert result.returncode == 2 and result.stderr and not result.stdout
    assert run("--help").returncode == 0
