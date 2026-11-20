import numpy as np
import pytest
import torch

from tiny_import_helpers import checkpoint, load

reader = load("importers/llama2c")
mapping = load("importers/mapping")


def reference(c, w, tokens):
    """NumPy causal attention with explicit per-query GQA selection."""
    x = w["token_embedding_table"][tokens].astype(np.float64)
    heads, kv = c["n_heads"], c["n_kv_heads"]
    hd = c["dim"] // heads
    length = len(tokens)
    angle = np.outer(np.arange(length), 10000.0 ** (-np.arange(0, hd, 2) / hd))

    def norm(x, g):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * g

    def rotate(x):
        y = x.copy()
        a, b = x[..., ::2], x[..., 1::2]
        y[..., ::2] = a * np.cos(angle)[:, None] - b * np.sin(angle)[:, None]
        y[..., 1::2] = a * np.sin(angle)[:, None] + b * np.cos(angle)[:, None]
        return y

    for layer in range(c["n_layers"]):
        h = norm(x, w["rms_att_weight"][layer])
        q = rotate((h @ w["wq"][layer].T).reshape(length, heads, hd))
        k = rotate((h @ w["wk"][layer].T).reshape(length, kv, hd))
        v = (h @ w["wv"][layer].T).reshape(length, kv, hd)
        out = np.empty_like(q)
        for t in range(length):
            for head in range(heads):
                kh = head // (heads // kv)
                scores = k[:t + 1, kh] @ q[t, head] / np.sqrt(hd)
                p = np.exp(scores - scores.max())
                out[t, head] = (p / p.sum()) @ v[:t + 1, kh]
        x = x + out.reshape(length, -1) @ w["wo"][layer].T
        h = norm(x, w["rms_ffn_weight"][layer])
        gate = h @ w["w1"][layer].T
        x = x + ((gate / (1 + np.exp(-gate))) * (h @ w["w3"][layer].T)) @ w["w2"][layer].T
    return norm(x, w["rms_final_weight"]) @ w["wcls"].T


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("kv", [1, 2])
def test_forward_matches_numpy(tmp_path, shared, kv):
    torch.set_num_threads(1)
    path = tmp_path / "toy.bin"
    weights = checkpoint(path, shared=shared, kv=kv)
    cfg, tensors = reader.read_checkpoint(path)
    model = mapping.to_model(cfg, tensors)
    tokens = np.array([1, 3, 2, 6])
    with torch.no_grad():
        actual = model(torch.tensor(tokens)[None])[0].numpy()
    np.testing.assert_allclose(actual, reference(cfg, weights, tokens), rtol=2e-5, atol=3e-7)
    assert model.blocks[0].k.weight.shape == (8, 8)
    if shared:
        assert type(model) is mapping.T.TinyLM
    else:
        np.testing.assert_array_equal(model.classifier.weight.detach(), weights["wcls"])


def test_reject_nonfinite(tmp_path):
    path = tmp_path / "toy.bin"
    weights = checkpoint(path)
    cfg, _ = reader.read_checkpoint(path)
    weights["wq"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mapping.to_model(cfg, weights)
