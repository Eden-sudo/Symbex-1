"""Rolling attention checked against independent chronological recomputation."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


@pytest.fixture
def modules(monkeypatch):
    loaded = []
    for name in ("train", "intsim", "gqa_intsim", "gqa_rolling"):
        spec = importlib.util.spec_from_file_location(name, DIR / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield loaded
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("window", [1, 4])
def test_ring_absolute_rope_oracle(modules, kv, window):
    _, I, G, R = modules
    torch.manual_seed(31)
    C, S = R.rope_tables(64, 13)
    ring = R.RollingAttention(4, kv, window, C, S)
    keys, values = [], []
    for pos in range(13):
        q = torch.randint(-12000, 12001, (4, 64))
        k, v = (torch.randint(-12000, 12001, (kv, 64)) for _ in range(2))
        keys.append(I.rope(k, C[pos], S[pos]))
        values.append(v)
        qq, qm = I.quant8(I.rope(q, C[pos], S[pos]))
        kq, km = I.quant8(torch.stack(keys[-window:], dim=1))
        vq, vm = I.quant8(torch.stack(values[-window:], dim=1))
        expected = G.quantized_attention(qq[None, :, None], qm[None, :, None],
                                        kq[None], km[None], vq[None], vm[None], torch.tensor(I.EXP_TABLE))
        assert torch.equal(ring.step(q, k, v), expected[0, :, 0])
    assert ring.k.numel() == kv * window * 64 and ring.k.dtype == torch.int8
    snapshot = ring.k.clone()
    with pytest.raises(ValueError, match="position"):
        ring.step(q, k, v)
    assert ring.position == 13 and torch.equal(snapshot, ring.k)


@pytest.mark.parametrize("kv", [1, 2, 4])
def test_generation_matches_prefix_then_deterministic(modules, kv):
    T, I, G, R = modules
    torch.manual_seed(4)
    model = T.TinyLM(16, 256, 2, 4, 64, 4, "binary", kv_heads=kv).eval()
    P = G.integerize(model, "binary")
    runs = [R.RollingModel(P, 4, 12) for _ in range(2)]
    tokens = [2, 7, 1, 5]
    for pos in range(12):
        token = tokens[pos]
        outputs = [run.step(token) for run in runs]
        assert all(torch.equal(a, b) for a, b in zip(*outputs))
        if pos < 4:
            z, hm = G.forward(P, torch.tensor([tokens[:pos + 1]]))
            assert torch.equal(z[0, -1], outputs[0][0])
            assert torch.equal(hm[0, -1], outputs[0][1])
        if pos >= 3:
            tokens.append(int(outputs[0][0].argmax()))
    assert len(tokens) > model.cos.shape[0]


def test_invalid_step_preserves_ring(modules):
    _, _, _, R = modules
    C, S = R.rope_tables(64, 3)
    ring = R.RollingAttention(2, 1, 2, C, S)
    with pytest.raises(ValueError, match="Q/K/V"):
        ring.step(torch.zeros(2, 64), torch.zeros(1, 64), torch.zeros(1, 64))
    assert ring.position == 0 and not ring.k.any()
