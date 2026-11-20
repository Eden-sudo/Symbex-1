import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_whole_toy_roundtrip(tmp_path):
    m, c = load("littlebit_model"), load("littlebit_container")
    model = m.lb._train.TinyLM(8, 4, 1, 1, 8, 2, "binary")
    m.replace_linears(model, rank=2, names=("w1",), residual=True)
    path = tmp_path / "model.lbm"
    c.export(model, path)
    data = path.read_bytes()
    result = c.read(path)
    assert sum(e["kind"] == "factor" for e in result.values()) == 1
    assert sum(e["kind"] == "binary" for e in result.values()) == 6
    actual = result["blocks.0.w1"]["value"]
    expected = c.blob.integer.integerize(model.blocks[0].w1)
    assert actual == expected
    assert c.blob.integer.forward_int(actual, [1, -2, 3, -4]) == c.blob.integer.forward_int(expected, [1, -2, 3, -4])
    binary = result["blocks.0.q"]["value"]
    signs = np.unpackbits(np.frombuffer(binary["signs"], dtype=np.uint8), bitorder="little").reshape(4, 8)[:, :4]
    weight = (signs.astype(np.float32) * 2 - 1) * binary["scales"][:, None]
    torch.testing.assert_close(torch.from_numpy(weight), m.lb._train.quant_weight(model.blocks[0].q.weight, "binary"))
    np.testing.assert_array_equal(result["emb.weight"]["value"], model.emb.weight.detach().numpy())
    np.testing.assert_array_equal(result["cos"]["value"], model.cos.numpy())
    for offset in (4, 32, 70, len(data) - 1):
        corrupt = bytearray(data)
        corrupt[offset] ^= 1
        with pytest.raises(ValueError):
            c.loads(corrupt)
    for corrupt in (data[:12], data[:-1], data + b"x"):
        with pytest.raises(ValueError):
            c.loads(corrupt)


def test_reject_unsupported_projection():
    m, c = load("littlebit_model"), load("littlebit_container")
    model = m.lb._train.TinyLM(8, 4, 1, 1, 8, 2, "fp32")
    with pytest.raises(ValueError, match="unsupported projection"):
        c.dumps(model)
