"""The pooled int8 classifier of method "boolean" (export/boolean.py, _fold_pool_head): its BatchNorm and per-class
scale folded into one int32 scale and bias per class, logit = scale * (q . counts) + bias, must give the same class
as the float head (BatchNorm1d of s * (q . counts)), and the fold must refuse what lib_symbex cannot run.
(The C side, symbex_bool_pool_head, is tested by lib_symbex's run_tests.py; whole networks by the export's C check.)
"""
import copy

import numpy as np
import pytest
import torch

from compiler_core.export.boolean import _fold_pool_head


class _Head(torch.nn.Module):
    """the interface the exporter reads: window, side, int8_weights() -> (q, s)"""

    def __init__(self, q, s, window, side, bias=False):
        super().__init__()
        self.window, self.side = window, side
        self.q, self.s = q, s
        self.bias = torch.nn.Parameter(torch.zeros(len(s))) if bias else None

    def int8_weights(self):
        return self.q, self.s


def _case(rng, n_cls=10, channels=16, side=8, window=2):
    n_win = (side // (window or side)) ** 2                                       # window 0: the whole map
    q = torch.as_tensor(rng.integers(-128, 128, (n_cls, channels * n_win)), dtype=torch.float32)
    s = torch.as_tensor(rng.uniform(1e-4, 1e-2, n_cls), dtype=torch.float32)
    bn = torch.nn.BatchNorm1d(n_cls)
    with torch.no_grad():
        bn.running_mean.copy_(torch.as_tensor(rng.normal(0, 2, n_cls)))
        bn.running_var.copy_(torch.as_tensor(rng.uniform(0.5, 4, n_cls)))
        bn.weight.copy_(torch.as_tensor(rng.uniform(0.2, 2, n_cls)))
        bn.bias.copy_(torch.as_tensor(rng.normal(0, 1, n_cls)))
    return _Head(q, s, window, side), bn.eval()


@pytest.mark.parametrize("window", [1, 2, 4, 0])
def test_folded_class_matches_float(rng, window):
    channels, side = 16, 8
    head, bn = _case(rng, channels=channels, side=side, window=window)
    k = window or side
    ph = _fold_pool_head(head, bn, channels)
    assert ph["window"] == k and ph["q"].shape == (10, channels, side // k, side // k)
    assert np.abs(ph["scale"]).max() < 2 ** 30 and np.abs(ph["bias"]).max() < 2 ** 30
    counts = rng.integers(0, k * k + 1, (2000, channels * (side // k) ** 2))          # CHW order, as in PyTorch
    dot = counts @ ph["q"].reshape(10, -1).T
    logit_int = dot * ph["scale"][None] + ph["bias"][None]
    with torch.no_grad():
        ref = copy.deepcopy(bn).double()(torch.as_tensor(counts, dtype=torch.float64) @ head.q.double().T
                                         * head.s.double()).numpy()
    bound = (np.abs(dot) + 1) * 0.5 / 2.0 ** ph["shift"] + 1e-9    # scale and bias each rounded by <= 1/2 step
    assert (np.abs(logit_int / 2.0 ** ph["shift"] - ref) <= bound).all()
    top2 = np.sort(ref, 1)[:, -2:]
    clear = top2[:, 1] - top2[:, 0] > 2 * bound.max()                # the rounding cannot swap these classes
    assert clear.mean() > 0.95
    assert np.array_equal(logit_int.argmax(1)[clear], ref.argmax(1)[clear])


def test_refuses_what_the_chip_cannot_run(rng):
    head, bn = _case(rng, side=8, window=2)
    with pytest.raises(ValueError, match="window 3 does not divide"):
        _fold_pool_head(_Head(head.q, head.s, 3, 8), bn, 16)
    with pytest.raises(ValueError, match="weights per class"):
        _fold_pool_head(head, bn, 32)
    with pytest.raises(ValueError, match="bias"):
        _fold_pool_head(_Head(head.q, head.s, 2, 8, bias=True), bn, 16)
    with pytest.raises(ValueError, match="-128..127"):
        _fold_pool_head(_Head(head.q * 2, head.s, 2, 8), bn, 16)
