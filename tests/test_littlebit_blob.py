import importlib.util
from pathlib import Path
import struct

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lb, blob, budget = load("littlebit"), load("littlebit_blob"), load("littlebit_budget")


@pytest.mark.parametrize("residual", [False, True])
def test_roundtrip(residual):
    w = torch.randn(7, 9, generator=torch.Generator().manual_seed(51)) * 0.1
    params = blob.integer.integerize(lb.FactorLinear.from_dense(w, 3, residual))
    data = blob.dumps(params)
    restored = blob.loads(data)
    assert restored == params
    assert blob.dumps(restored) == data
    assert len(data) == 40 + budget.storage_bytes(3, 9, 7, residual, 32)
    q = [-128, -90, -1, 0, 1, 5, 72, 126, 127]
    assert blob.integer.forward_int(params, q) == blob.integer.forward_int(restored, q)


def test_malformed():
    params = blob.integer.integerize(lb.FactorLinear.from_dense(torch.ones(3, 5), 2))
    data = blob.dumps(params)
    bad = [data[:4], data[:-1], data + b"x", b"BAD!" + data[4:]]
    for offset, value in ((4, 2), (8, 0), (12, 0), (20, 3), (24, 19), (28, 11)):
        changed = bytearray(data)
        struct.pack_into("<I", changed, offset, value)
        bad.append(changed)
    changed = bytearray(data)
    changed[40] |= 128
    bad.append(changed)
    changed = bytearray(data)
    struct.pack_into("<d", changed, 32, float("nan"))
    bad.append(changed)
    for value in bad:
        with pytest.raises(ValueError):
            blob.loads(value)
