"""Training tester: does a binary network keep learning? The exactness tests check that the C equals the integer
simulation; they cannot see a training that stopped moving its bits (the frozen network is still exact). Here each
benchmark's real model trains for a few CPU steps on random data, with a large Adam learning rate on the 1-bit
weights so that latent weights cross 1 within the test, and we check compiler_core/train/health.py:
    - without clipping the defect is visible (some latent 1-bit weights end beyond |w| > 1 and get zero gradient);
    - with clip_latent after every step no weight is beyond 1 and every 1-bit weight still gets a gradient (except
      whole output channels whose activation saturated: a different effect, not the weights');
    - the CDT's boolean.train honors [training] clip_latent and writes binary_health into train.json.
Found on November 13, 2026: ResNet-8 trained with 52 % of its bits frozen, HAR / ECG / FOMO 5-45 %.
"""
import importlib.util
import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from compiler_core.train import boolean
from compiler_core.train.health import binary_health, clip_latent

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "benchmarks")
STEPS, LR = 60, 0.1


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def resnet8():
    return _load("resnet8_model_for_health", os.path.join(BENCH, "conv2d", "cnn_cifar_resnet8", "model.py"))


@pytest.fixture(scope="module")
def har():
    return _load("har_model_for_health", os.path.join(BENCH, "conv1d", "har_ultracompact", "model.py"))


@pytest.fixture(scope="module")
def fomo():
    return _load("fomo_model_for_health", os.path.join(BENCH, "conv2d", "fomo_voc_person", "model.py"))


def _cases(resnet8, har, fomo):
    """(name, model, its 1-bit weights, a batch maker, a loss): every binary recipe that trains with Adam"""
    def images(n, size):
        return torch.randint(0, 256, (n, 3, size, size)).float() / 255

    def windows(n, length, channels):
        return torch.randint(-128, 128, (n, length, channels)).to(torch.int8)

    r8 = resnet8.BoolResNet8(1.0, 2)
    fm = fomo.BoolFOMO((8, 8, 8), (1, 1, 1), 2)
    hb = har.BinaryHAR("min", "binary")
    hi = har.BinaryHAR("min", "int8")
    ecg = har.BinaryHAR("s", "binary", layers=[("conv", 8, 9), ("pool", 4), ("conv", 16, 9), ("pool", 4)],
                        length=64, channels=1, classes=4)
    ce = lambda out, y: F.cross_entropy(out, y)                                           # noqa: E731
    return [
        ("resnet8", r8, boolean.binary_weights(r8), lambda: (images(16, 32), torch.randint(0, 10, (16,))), ce),
        ("fomo", fm, boolean.binary_weights(fm),
         lambda: (images(8, 96), torch.randint(0, 2, (8, 12, 12))), ce),
        ("har_binary", hb, hb.one_bit_weights(),
         lambda: (windows(32, 151, 3), torch.randint(0, 17, (32,))), ce),
        ("har_int8_first", hi, hi.one_bit_weights(),
         lambda: (windows(32, 151, 3), torch.randint(0, 17, (32,))), ce),
        ("ecg", ecg, ecg.one_bit_weights(), lambda: (windows(32, 64, 1), torch.randint(0, 4, (32,))), ce),
    ]


def _train(model, bits, batch, loss_fn, clip):
    torch.manual_seed(0)
    ids = {id(w) for w in bits}                      # the large rate only for the 1-bit weights, BatchNorms as usual
    opt = torch.optim.Adam([{"params": bits, "lr": LR},
                            {"params": [p for p in model.parameters() if id(p) not in ids], "lr": 1e-3}])
    model.train()
    for _ in range(STEPS):
        x, y = batch()
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if clip:
            clip_latent(bits)
    x, y = batch()                                   # one more backward: which 1-bit weights still get a gradient
    opt.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    return [w.grad if w.grad is not None else torch.zeros_like(w) for w in bits]


def test_binary_health_counts():
    w = [torch.tensor([0.5, -1.0, 1.2, -3.0]), torch.tensor([[0.1, -0.1]])]
    h = binary_health(w, [torch.tensor([0.5, 1.0, 1.0, -1.0]), torch.tensor([[-0.1, -0.1]])])
    assert h["n"] == 6 and h["frozen"] == pytest.approx(100 * 2 / 6, abs=1e-3) and h["max_abs"] == 3.0
    assert h["flipped"] == pytest.approx(100 * 2 / 6, abs=1e-3)                          # -1.0 vs 1.0, 0.1 vs -0.1
    clip_latent(w)
    assert binary_health(w)["frozen"] == 0 and binary_health(w)["max_abs"] == 1.0


def test_one_bit_weights_of_har(har):
    hb, hi = har.BinaryHAR("max", "binary"), har.BinaryHAR("max", "int8")
    assert len(hb.one_bit_weights()) == len(hb.convs) + 1
    assert id(hi.convs[0].weight) not in {id(w) for w in hi.one_bit_weights()}            # int8 first: not 1-bit
    assert len(hi.one_bit_weights()) == len(hi.convs)


@pytest.mark.parametrize("clip", [False, True], ids=["adam", "adam_clip"])
def test_every_recipe_keeps_its_bits_alive(resnet8, har, fomo, clip):
    for name, model, bits, batch, loss_fn in _cases(resnet8, har, fomo):
        bits0 = [w.detach().clone() for w in bits]
        grads = _train(model, bits, batch, loss_fn, clip)
        h = binary_health(bits, bits0)
        beyond = torch.cat([(w.detach().abs() > 1).flatten() for w in bits])
        dead = torch.cat([(g == 0).flatten() for g in grads])
        # a whole output channel without gradient is its ACTIVATION saturated (|BN| > 1 over the whole batch at this
        # large learning rate), not its weights: the weight-level check counts only the rest
        lone = torch.cat([((g == 0) & ~(g == 0).flatten(1).all(1).view(-1, *[1] * (g.dim() - 1))).flatten()
                          for g in grads])
        assert h["flipped"] > 0, f"{name}: no bit moved in {STEPS} steps"
        if clip:
            assert h["frozen"] == 0 and h["max_abs"] <= 1.0, f"{name}: {h}"
            # (a sum of +-1 products can cancel exactly: a few lone zeros are allowed, frozen weights were 5-50 %)
            assert lone.float().mean() < 0.005, f"{name}: {int(lone.sum())} 1-bit weights get no gradient although clipped"
        else:                                        # the defect the clipping removes, visible to the tester
            assert h["frozen"] > 0, f"{name}: {LR} x {STEPS} Adam steps should push some weights past 1"
            assert dead[beyond].all(), f"{name}: a weight beyond 1 still gets a gradient (straight-through changed?)"


def test_boolean_train_clip_latent(resnet8, tmp_path):
    torch.manual_seed(0)
    x = torch.randint(0, 256, (64, 3, 32, 32)).float() / 255
    y = torch.randint(0, 10, (64,))
    ds = TensorDataset(x, y)
    cfg = {"training": dict(seed=0, epochs=2, learning_rate=0.2, batch_size=16, clip_latent=True)}
    trained = boolean.train(resnet8.BoolResNet8(1.0, 2), ds, ds, cfg, str(tmp_path))["model"]
    assert max(w.detach().abs().max().item() for w in boolean.binary_weights(trained)) <= 1.0
    info = json.load(open(tmp_path / "train.json"))
    assert info["clip_latent"] is True and info["binary_health"]["frozen"] == 0
    assert info["binary_health"]["flipped"] > 0


def test_boolean_train_records_frozen_weights_without_clip(resnet8, tmp_path):
    torch.manual_seed(0)
    x = torch.randint(0, 256, (64, 3, 32, 32)).float() / 255
    ds = TensorDataset(x, torch.randint(0, 10, (64,)))
    cfg = {"training": dict(seed=0, epochs=2, learning_rate=0.2, batch_size=16)}
    boolean.train(resnet8.BoolResNet8(1.0, 2), ds, ds, cfg, str(tmp_path))
    info = json.load(open(tmp_path / "train.json"))
    assert info["clip_latent"] is False and info["binary_health"]["frozen"] > 0
