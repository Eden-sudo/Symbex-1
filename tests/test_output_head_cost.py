import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/head_cost.py"
spec = importlib.util.spec_from_file_location("head_cost", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_anchor_and_dense_scaling():
    anchor = m.head_cost(4096)
    assert anchor["head_payload_bytes"] == 1572864
    assert anchor["estimated_ms_payload"] == anchor["estimated_ms_macs"] == 28
    assert anchor["head_scale_bytes"] == 16384
    assert anchor["separate_embedding_bytes"] == 0
    assert m.head_cost(2048, bits=4)["estimated_ms_payload"] == 7
    assert m.head_cost(2048, bits=4)["estimated_ms_macs"] == 14


def test_factor_sizes_padding_and_embedding():
    row = m.head_cost(1024, rank=32, bits=2)
    assert row["macs"] == 32 * (384 + 1024)
    assert row["head_payload_bytes"] == 11264
    assert row["head_scale_bytes"] == 4 * (32 + 1024)
    assert row["separate_embedding_bytes"] == 1024 * 384 + 4096
    assert row["model_table_bytes"] == 412800
    assert row["intermediate_int32_bytes"] == 128
    assert row["estimated_ms_payload"] == pytest.approx(28 * 11264 / 1572864)
    odd = m.head_cost(7, rank=3, bits=2, dim=5)
    assert odd["head_payload_bytes"] == 3 * 2 + 7 * 1


def test_table_and_cli():
    table = m.cost_table()
    assert len(table) == 36
    assert all(row["estimate_only"] for row in table)
    result = subprocess.run([sys.executable, str(PATH), "--format", "json"],
                            check=True, capture_output=True, text=True, timeout=5)
    assert json.loads(result.stdout)["rows"] == table
    result = subprocess.run([sys.executable, str(PATH)], check=True,
                            capture_output=True, text=True, timeout=5)
    assert "Estimates only" in result.stdout and "28.000" in result.stdout


@pytest.mark.parametrize("kwargs", [{"vocab": 0}, {"vocab": 8, "rank": 9},
                                   {"vocab": 8, "bits": 3}, {"vocab": 8, "rank": True}])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        m.head_cost(**kwargs)
