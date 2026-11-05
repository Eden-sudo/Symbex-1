"""Bop, the optimizer for the 1-bit weights of method "boolean" (compiler_core/train/bop.py, [training]
binary_optimizer = "bop"): when a weight flips, which weights it owns, and that boolean.train uses it end to end
(a few CPU steps on random images; the accuracy is measured by benchmarks/experiments/training_gap/train_gap_study.py, not here).
"""
import importlib.util
import os

import pytest
import torch
from torch.utils.data import TensorDataset

from compiler_core.train import boolean
from compiler_core.train.bop import MAGNITUDE, Bop

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESNET8 = os.path.join(REPO, "benchmarks", "conv2d", "cnn_cifar_resnet8", "model.py")


@pytest.fixture(scope="module")
def resnet8():
    spec = importlib.util.spec_from_file_location("resnet8_model_for_bop", RESNET8)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _one_step(w0, grad, gamma=1.0, tau=0.5):
    w = torch.nn.Parameter(torch.tensor(w0))
    bop = Bop([w], gamma=gamma, tau=tau)
    w.grad = torch.tensor(grad)
    bop.step()
    return w.detach()


def test_weights_become_plus_minus_magnitude():
    assert torch.equal(_one_step([0.3, -2.0, 0.0], [0.0, 0.0, 0.0]), torch.tensor([MAGNITUDE, -MAGNITUDE, MAGNITUDE]))


def test_flips_only_when_the_gradient_pushes_against_the_sign_beyond_tau():
    # w > 0 and grad > tau: descent wants it negative -> flips; grad < 0 pushes it further positive -> stays;
    # |grad| <= tau -> stays
    got = _one_step([1.0, 1.0, 1.0, -1.0, -1.0], [0.9, -0.9, 0.4, -0.9, 0.9])
    assert torch.equal(got, MAGNITUDE * torch.tensor([-1.0, 1.0, 1.0, 1.0, -1.0]))


def test_the_gradient_is_averaged():
    w = torch.nn.Parameter(torch.tensor([1.0]))
    bop = Bop([w], gamma=0.5, tau=0.6)
    for step in range(2):                                  # m = 0.5, then 0.75: flips at the second step only
        w.grad = torch.tensor([1.0])
        bop.step()
        if step == 0:
            assert w.item() == MAGNITUDE
    assert w.item() == -MAGNITUDE
    with pytest.raises(ValueError):
        Bop([w], gamma=0.0)


def test_binary_weights_follow_the_boolean_contract(resnet8):
    net = resnet8.BoolResNet8(1.0, 8)
    convs = [m.weight for m in net.modules() if isinstance(m, (resnet8.BinaryConv, resnet8.BinaryLinear))]
    assert {id(w) for w in boolean.binary_weights(net)} == {id(w) for w in convs}
    i8 = resnet8.BoolResNet8(1.0, 8, stem="int8")
    ws = {id(w) for w in boolean.binary_weights(i8)}
    assert id(i8.stem.weight) not in ws and len(ws) == len(convs) - 1


def test_boolean_train_with_bop(resnet8, tmp_path):
    torch.manual_seed(0)
    x = torch.randint(0, 256, (64, 3, 32, 32)).float() / 255
    y = torch.randint(0, 10, (64,))
    ds = TensorDataset(x, y)
    cfg = {"training": dict(seed=0, epochs=1, learning_rate=5e-3, batch_size=16, binary_optimizer="bop",
                            bop_gamma=1e-2, bop_tau=1e-8)}
    model = resnet8.BoolResNet8(1.0, 2)
    before = [w.detach().clone() for w in boolean.binary_weights(model)]
    bn_before = model.stem_bn.weight.detach().clone()
    trained = boolean.train(model, ds, ds, cfg, str(tmp_path))["model"]
    after = boolean.binary_weights(trained)
    assert all(torch.equal(w.abs(), torch.full_like(w, MAGNITUDE)) for w in after)        # no latent magnitudes
    assert any(not torch.equal(torch.sign(a), torch.where(b >= 0, 1.0, -1.0)) for a, b in zip(after, before))
    assert not torch.equal(trained.stem_bn.weight, bn_before)                            # Adam trains the rest
    assert (tmp_path / "train.json").read_text().count('"binary_optimizer": "bop"') == 1
