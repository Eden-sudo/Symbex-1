"""CPU regression tests for training-only grouped-query attention."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F


DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "experiments" / "tiny_llm"


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def modules(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    train = load("train", DIR / "train.py")
    return train, load("tiny_intsim", DIR / "intsim.py")


class LegacyBlock(nn.Module):
    def __init__(self, train, heads, kind):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(128, eps=1e-5), nn.RMSNorm(128, eps=1e-5)
        self.q, self.k, self.v, self.o = (train.QLinear(128, 128, kind) for _ in range(4))
        self.w1, self.w3 = train.QLinear(128, 128, kind), train.QLinear(128, 128, kind)
        self.w2 = train.QLinear(128, 128, kind)
        self.rope = train.rope

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = (m(h).view(b, t, self.heads, -1).transpose(1, 2) for m in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(self.rope(q, cos, sin), self.rope(k, cos, sin), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class LegacyModel(nn.Module):
    def __init__(self, train, heads, kind):
        super().__init__()
        self.emb = nn.Embedding(32, 128)
        self.blocks = nn.ModuleList(LegacyBlock(train, heads, kind) for _ in range(2))
        self.norm = nn.RMSNorm(128, eps=1e-5)
        hd = 128 // heads
        freq = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(8).float(), freq)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for blk in self.blocks:
            for m in (blk.o, blk.w2):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(4))

    def forward(self, idx):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, self.cos[:idx.shape[1]], self.sin[:idx.shape[1]])
        return F.linear(self.norm(x), self.emb.weight)


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("kind", ["fp32", "binary", "ternary"])
def test_legacy_equality(modules, heads, kind):
    train, _ = modules
    models = []
    for make in (lambda: LegacyModel(train, heads, kind),
                 lambda: train.TinyLM(32, 128, 2, heads, 128, 8, kind),
                 lambda: train.TinyLM(32, 128, 2, heads, 128, 8, kind, kv_heads=heads)):
        torch.manual_seed(123)
        models.append(make().cpu())
    states = [m.state_dict() for m in models]
    assert all({k: v.shape for k, v in s.items()} == {k: v.shape for k, v in states[0].items()} for s in states)
    assert all(torch.equal(states[0][k], s[k]) for s in states[1:] for k in states[0])
    x = torch.randint(32, (2, 8), device="cpu")
    assert all(torch.equal(models[0](x), m(x)) for m in models[1:])


@pytest.mark.parametrize("kv_heads", [1, 2, 3])
@pytest.mark.parametrize("kind", ["fp32", "binary", "ternary"])
def test_gqa_gradients(modules, kv_heads, kind):
    train, _ = modules
    torch.manual_seed(123)
    model = train.TinyLM(32, 128, 2, 6, 128, 8, kind, kv_heads=kv_heads).cpu()
    for block in model.blocks:
        assert block.q.weight.shape == (384, 128)
        assert block.k.weight.shape == block.v.weight.shape == (kv_heads * 64, 128)
        assert block.o.weight.shape == (128, 384)
    x = torch.randint(32, (2, 8), device="cpu")
    logits = model(x)
    assert logits.shape == (2, 8, 32) and torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.reshape(-1, 32), x.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    for block in model.blocks:
        for layer in (block.q, block.k, block.v):
            assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0


def test_grouping_and_causality(modules):
    train, _ = modules
    torch.manual_seed(2)
    model = train.TinyLM(32, 128, 2, 6, 128, 8, "fp32", kv_heads=2).cpu()
    block = model.blocks[0]
    x = torch.randn(2, 8, 128, device="cpu")
    h = block.n1(x)
    q = train.rope(block.q(h).view(2, 8, 6, 64).transpose(1, 2), model.cos, model.sin)
    k = train.rope(block.k(h).view(2, 8, 2, 64).transpose(1, 2), model.cos, model.sin)
    v = block.v(h).view(2, 8, 2, 64).transpose(1, 2)
    indices = torch.tensor([0, 0, 0, 1, 1, 1])
    scores = q @ k[:, indices].transpose(-2, -1) / 8
    scores = scores.masked_fill(torch.ones(8, 8, dtype=torch.bool).triu(1), float("-inf"))
    a = (scores.softmax(-1) @ v[:, indices]).transpose(1, 2).reshape(2, 8, 384)
    residual = x + block.o(a)
    h = block.n2(residual)
    expected = residual + block.w2(F.silu(block.w1(h)) * block.w3(h))
    actual = block(x, model.cos, model.sin)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    changed = x.clone()
    changed[:, 4:] += 10
    torch.testing.assert_close(actual[:, :4], block(changed, model.cos, model.sin)[:, :4])


@pytest.mark.parametrize("extra,expected", [([], 6), (["--heads", "4"], 4),
                                          (["--heads", "6", "--kv-heads", "6"], 6),
                                          (["--heads", "6", "--kv-heads", "2"], 2)])
def test_parser_valid(modules, monkeypatch, extra, expected):
    train, _ = modules
    seen = []
    monkeypatch.setattr(train, "run", lambda args, seed: seen.append(vars(args).copy()))
    monkeypatch.setattr(sys, "argv", ["train.py", "--weights", "fp32", *extra])
    train.main()
    assert seen[0]["kv_heads"] == expected


@pytest.mark.parametrize("extra", [["--kv-heads", str(n)] for n in (4, 0, -1, 7)] +
                         [["--kv-heads", "no"], ["--heads", "0"], ["--heads", "-1"]])
def test_parser_invalid(modules, monkeypatch, capsys, extra):
    train, _ = modules
    seen = []
    monkeypatch.setattr(train, "run", lambda *args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["train.py", "--weights", "fp32", *extra])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 2 and "error:" in capsys.readouterr().err and not seen


@pytest.mark.parametrize("heads,kv", [(6, 4), (6, 0), (0, None), (-1, 1), (6, 7)])
def test_invalid_constructor(modules, heads, kv):
    train, _ = modules
    with pytest.raises(ValueError, match="heads"):
        train.TinyLM(32, 128, 2, heads, 128, 8, "fp32", kv_heads=kv)
    with pytest.raises(ValueError, match="heads"):
        train.Block(128, heads, 128, "fp32", kv_heads=kv)


@pytest.mark.parametrize("kv", [None, 2])
def test_checkpoint_legacy(modules, tmp_path, kv):
    train, intsim = modules
    args = dict(dim=128, layers=2, heads=2, hidden=128, seq=8, weights="fp32")
    if kv is not None:
        args["kv_heads"] = kv
    model = train.TinyLM(32, 128, 2, 2, 128, 8, "fp32").cpu()
    torch.save({"args": args, "model": model.state_dict()}, tmp_path / "model.pt")
    loaded, saved = intsim.load_model(str(tmp_path), 32, "cpu")
    assert saved == args and not loaded.training
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())


def test_checkpoint_rejected_before_construction(modules, tmp_path, monkeypatch):
    train, intsim = modules
    model = train.TinyLM(32, 128, 2, 6, 128, 8, "fp32", kv_heads=2).cpu()
    args = dict(dim=128, layers=2, heads=6, kv_heads=2, hidden=128, seq=8, weights="fp32")
    torch.save({"args": args, "model": model.state_dict()}, tmp_path / "model.pt")
    def fail(*args, **kwargs):
        raise RuntimeError("model construction must not occur")
    monkeypatch.setattr(train, "TinyLM", fail)
    with pytest.raises(AssertionError, match="no integer model or C support yet"):
        intsim.load_model(str(tmp_path), 32, "cpu")
