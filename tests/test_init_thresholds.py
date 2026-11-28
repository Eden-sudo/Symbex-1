"""init_thresholds sets the thresholds one layer at a time (bug found Nov 25, 2026): each RSign is calibrated on the
input it gets with the EARLIER thresholds in place. Calibrating them all in one pass, with every beta still 0, made
the later layers' outputs constant per channel (all +1 after a ReLU) and left those channels dead."""
import torch
import torch.nn as nn

from compiler_core.methods import apply_plan, init_thresholds
from compiler_core.methods.base import RSign


def _mlp(seed, depth=3):
    torch.manual_seed(seed)
    sizes = [20] + [32] * depth + [4]
    mods = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        mods += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
    model = nn.Sequential(*mods[:-2])                    # no BN / ReLU after the classifier
    plan = {str(3 * i): "bin" for i in range(depth + 1)}
    apply_plan(model, plan)
    return model


def _rsigns(model):
    return [m for m in model.modules() if isinstance(m, RSign)]


def _bits(model, x):
    out = []
    hooks = [m.register_forward_hook(lambda m, i, o: out.append(o.detach())) for m in _rsigns(model)]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return out


def test_no_constant_channel_after_init():
    model = _mlp(0)
    x = torch.randn(256, 20)
    assert init_thresholds(model, x) == len(_rsigns(model)) == 3
    for i, b in enumerate(_bits(model, x)):
        moving = (b.max(0).values != b.min(0).values).float().mean().item()
        assert moving > 0.9, f"RSign {i}: only {moving:.0%} of its channels move"


def test_each_threshold_is_measured_with_the_earlier_ones_in_place():
    model = _mlp(1)
    x = torch.randn(256, 20)
    from compiler_core.methods import _set_beta
    init_thresholds(model, x)
    model.eval()                                         # init_thresholds measures in eval mode (BN running stats)
    got = [m.beta.detach().clone() for m in _rsigns(model)]
    for i, mod in enumerate(_rsigns(model)):
        seen = {}
        h = mod.register_forward_pre_hook(lambda m, inp: seen.setdefault(m, inp[0].detach()))
        with torch.no_grad():
            model(x)                                     # every threshold already set: same input as at step i
        h.remove()
        ref = RSign(mod.beta.shape[1], dims=mod.beta.dim())
        with torch.no_grad():
            _set_beta(ref, seen[mod])
        assert torch.equal(ref.beta.detach(), got[i]), f"RSign {i}"
