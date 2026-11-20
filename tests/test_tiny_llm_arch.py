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


@pytest.mark.parametrize("bits,suffix", [(8, ""), (4, "_head4"), (2, "_head2")])
def test_latency_against_measured_totals(planner, bits, suffix):
    measured = json.loads((TINY / f"results/chip_s3_binary_d384l8{suffix}_s42_symbex.json").read_text())
    c = planner.Config(head_bits=bits)
    for workload, key in (("mean", "ms_per_token_check_mean"), ("story", "ms_per_token_story_mean")):
        estimate = planner.latency(c, workload)
        assert estimate["ms_per_token"] == pytest.approx(measured[key], rel=0.05)
        assert estimate["tokens_per_second"] * estimate["ms_per_token"] == pytest.approx(1000)
    assert planner.latency(c, "full")["ms_per_token"] == pytest.approx(
        measured["ms_per_token_at_position"]["255"], rel=0.05)


def test_latency_scaling_and_workloads(planner):
    from dataclasses import replace

    c = planner.Config()
    base = planner.latency(c)
    assert planner.latency(replace(c, layers=16))["parts_ms"]["linear"] == 2 * base["parts_ms"]["linear"]
    assert planner.latency(replace(c, head_bits=4))["parts_ms"]["head"] == base["parts_ms"]["head"] / 2
    assert planner.latency(replace(c, ctx=512))["parts_ms"]["attention"] > base["parts_ms"]["attention"]
    assert planner.latency(c, "story")["tokens_per_second"] == pytest.approx(8.481, rel=0.001)
    assert planner.latency(c)["tokens_per_second"] < 8
    with pytest.raises(ValueError):
        planner.latency(c, "unknown")


def test_memory_default_and_placement(planner):
    m = planner.memory(planner.Config())
    assert m["fits"] and not m["firmware_gaps"]
    assert m["psram"]["kv_payload"] == 1572864
    assert m["psram"]["kv_scales"] == 98304
    assert m["sram"]["linear_lut"] == 65536
    assert m["sram"]["logits"] == 32768
    # Recorded ~5.35 MB PSRAM consumption includes model, KV/scales and row counts.
    used = m["totals"]["psram"] - m["psram"]["reserve"]
    assert used == pytest.approx(8386295 - 3036407, abs=4096)


def test_gqa_low_bit_kv_saves_payload_not_scales(planner):
    base = planner.memory(planner.Config())
    proposed = planner.memory(planner.Config(kv_heads=2, kv_bits=4))
    assert proposed["psram"]["kv_payload"] == base["psram"]["kv_payload"] // 6
    assert proposed["psram"]["kv_scales"] == base["psram"]["kv_scales"] // 3
    assert len(proposed["firmware_gaps"]) == 2


@pytest.mark.parametrize("budget,part", [("sram_bytes", "sram"), ("psram_bytes", "psram"),
                                       ("flash_bytes", "flash"), ("model_partition_bytes", "partition")])
def test_memory_rejects_each_overflow(planner, budget, part):
    result = planner.memory(planner.Config(), **{budget: 1})
    assert not result["fits"]
    assert any(part in message for message in result["violations"])


def test_memory_exact_boundary_and_firmware_limits(planner):
    c = planner.Config()
    total = planner.memory(c)["totals"]["sram"]
    assert planner.memory(c, sram_bytes=total)["fits"]
    assert not planner.memory(c, sram_bytes=total - 1)["fits"]
    assert planner.memory(planner.Config(ctx=512))["firmware_gaps"]
    with pytest.raises(ValueError):
        planner.memory(c, sram_reserve=-1)


def test_mac_training_anchor_and_arithmetic(planner):
    t = planner.training(planner.Config())
    assert t["seconds_per_step"] == pytest.approx(1.59)
    assert t["tokens_per_second"] == pytest.approx(32768 / 1.59)
    assert t["hours"] == pytest.approx(10000 * 1.59 / 3600)
    assert t["memory_bytes"] == sum(t["memory_parts"].values())
    assert t["fits_24gib"]
    assert planner.training(planner.Config(), tokens=0)["hours"] == 0


def test_training_scales_flops_not_quantized_bytes(planner):
    from dataclasses import replace

    c = planner.Config()
    base = planner.training(c)
    assert planner.training(replace(c, head_bits=2, kv_bits=2))["seconds_per_step"] == base["seconds_per_step"]
    assert planner.training(c, batch=64)["seconds_per_step"] == base["seconds_per_step"] / 2
    bigger = planner.training(replace(c, layers=16))
    expected_flops = 6 * (2 * 14155776 + 4096 * 384) + 12 * 16 * 256 * 384
    assert bigger["flops_per_token"] == expected_flops
    assert bigger["hours"] / base["hours"] == pytest.approx(expected_flops / base["flops_per_token"])
    assert not planner.training(c, batch=4096)["fits_24gib"]


@pytest.mark.parametrize("kwargs", [dict(batch=0), dict(seq=257), dict(tokens=-1),
                                   dict(tokens=float("inf")), dict(seq=1.5)])
def test_training_rejects_invalid_inputs(planner, kwargs):
    with pytest.raises(ValueError):
        planner.training(planner.Config(), **kwargs)
