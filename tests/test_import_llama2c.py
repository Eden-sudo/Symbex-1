import struct

import numpy as np
import pytest

from tiny_import_helpers import checkpoint, load

reader = load("importers/llama2c")


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("kv", [1, 2])
def test_every_tensor(tmp_path, shared, kv):
    path = tmp_path / "toy.bin"
    expected = checkpoint(path, shared=shared, kv=kv)
    cfg, actual = reader.read_checkpoint(path)
    assert cfg == dict(dim=8, hidden_dim=12, n_layers=2, n_heads=2,
                       n_kv_heads=kv, vocab_size=7, seq_len=4, shared_classifier=shared)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert (actual["wcls"] is actual["token_embedding_table"]) == shared


@pytest.mark.parametrize("data,match", [
    (b"short", "header"),
    (struct.pack("<7i", 8, 12, 2, 3, 1, 7, 4), "attention"),
    (struct.pack("<7i", 8, 12, 0, 2, 1, 7, 4), "positive"),
    (struct.pack("<7i", 8, 12, 2, 2, 1, 7, 4), "size mismatch"),
])
def test_bad_header(tmp_path, data, match):
    path = tmp_path / "bad.bin"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=match):
        reader.read_checkpoint(path)


@pytest.mark.parametrize("delta", [-1, 1])
def test_exact_length(tmp_path, delta):
    path = tmp_path / "toy.bin"
    checkpoint(path)
    data = path.read_bytes()
    path.write_bytes(data[:-1] if delta < 0 else data + b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        reader.read_checkpoint(path)
