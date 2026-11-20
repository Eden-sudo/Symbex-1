"""Arithmetic and CLI tests; lower-bit sizing makes no kernel claims."""
import importlib.util
import json
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kv", [6, 3, 2, 1])
@pytest.mark.parametrize("bits", [8, 4, 2, 1])
def test_default_shape_arithmetic(kv, bits):
    M = load("attention_memory")
    r = M.kv_bytes(kv_heads=kv, bits=bits)
    assert r["payload_bytes_per_token"] == 2 * 8 * kv * 64 * bits // 8
    assert r["scale_bytes_per_token"] == 2 * 8 * kv * 4
    assert r["kv_total_bytes"] == (2 * 8 * kv * (64 * bits // 8 + 4)) * 256
    mha = M.kv_bytes(bits=bits)
    assert r["kv_total_bytes"] * (6 // kv) == mha["kv_total_bytes"]


def test_existing_budget_and_row_padding():
    M, B = load("attention_memory"), load("memory_budget")
    old = B.budget(384, 8, 6, 1024, 256, 4096, "binary", "fp32")
    assert M.kv_bytes()["kv_total_bytes"] + 16 == old["kv_cache_bytes"]
    assert M.kv_bytes()["payload_total_bytes"] == 1572864
    assert M.kv_bytes(kv_heads=1)["kv_total_bytes"] == 278528
    # Two separately packed 3-element rows require 2 bytes, not ceil(6/8)=1.
    assert M.kv_bytes(1, 1, 1, 1, 3, 1)["payload_bytes_per_token"] == 2


@pytest.mark.parametrize("kwargs", [{"heads": 0}, {"kv_heads": 4}, {"bits": 3},
                                    {"ctx": -1}, {"layers": True}, {"dim": 383}])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        load("attention_memory").kv_bytes(**kwargs)


def test_cli(capsys):
    M = load("attention_memory")
    M.main(["--kv-heads", "1", "--ctx", "256", "--bits", "8", "--json"])
    assert json.loads(capsys.readouterr().out) == [M.kv_bytes(kv_heads=1)]
    M.main([])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 25 and lines[0].endswith("KV_total_B")
    assert lines[1] == "6 6 256 8 6144 384 6528 1671168"
    with pytest.raises(SystemExit) as exc:
        M.main(["--kv-heads", "4"])
    assert exc.value.code == 2
