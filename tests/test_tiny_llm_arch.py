"""Small CPU-only checks for the analytical architecture planner; no forward/train."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "benchmarks/experiments/tiny_llm"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def planner():
    return load("tiny_arch_planner_test", TINY / "arch/planner.py")


def test_real_default_model_parameter_count(planner):
    import torch

    train = load("tiny_arch_train_test", TINY / "train.py")
    c = planner.Config()
    # Explicit CPU construction only: ~63 MB, no forward/backward or data.
    with torch.device("cpu"):
        model = train.TinyLM(c.vocab, c.dim, c.layers, c.heads, c.hidden, c.ctx, "binary")
    measured = json.loads((TINY / "results/train_binary_d384l8_s42.json").read_text())
    counts = planner.size_model(c)
    assert counts["parameters"] == sum(p.numel() for p in model.parameters()) == measured["parameters"]
    assert counts["linear_parameters"] == measured["quantized_linear_parameters"]


@pytest.mark.parametrize("bits,expected", [(8, 3550752), (4, 2765216), (2, 2372000)])
def test_recorded_export_bytes(planner, bits, expected):
    assert planner.size_model(planner.Config(head_bits=bits))["flash_bytes"] == expected


def test_gqa_and_tied_embedding(planner):
    c = planner.Config(dim=64, layers=2, heads=4, kv_heads=2, hidden=128, vocab=64, ctx=16)
    linear = 2 * (2 * 64 * 64 + 2 * 32 * 64 + 3 * 64 * 128)
    s = planner.size_model(c)
    assert s["parameters"] == linear + 64 * 64 + 5 * 64
    assert s["flash_parts"]["embedding"] == 4096  # not counted twice


def test_factor_budget_counts_scales_and_row_padding(planner):
    p = planner.projection_layout(1024, 384, 0.3)
    r = p["rank"]
    assert r > 0
    assert p["parameters"] == r * (1024 + 384) + 1024 + 384 + r
    assert p["achieved_bpw"] <= 0.3
    next_bytes = (planner.packed_rows(1024, r + 1, 1) + planner.packed_rows(r + 1, 384, 1)
                  + ((2 * (1024 + 384 + r + 1) + 3) // 4) * 4)
    assert next_bytes * 8 > 0.3 * 1024 * 384
    assert planner.packed_rows(3, 33, 1) == 24


@pytest.mark.parametrize("kwargs", [dict(dim=0), dict(dim=383), dict(kv_heads=4),
                                   dict(head_bits=1), dict(kv_bits=3), dict(ctx=True),
                                   dict(weight_bits=(float("nan"),) * 7),
                                   dict(weight_bits=(0.00001,) * 7)])
def test_invalid_configs(planner, kwargs):
    with pytest.raises(ValueError):
        planner.Config(**kwargs)
