"""LittleBit Dual-SVID (2506.13771 §3.2), optionally Joint-ITQ (2603.00042 Alg. 1)."""
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location("tiny_llm_train_for_littlebit", Path(__file__).with_name("train.py"))
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load tiny_llm train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)

_itq_spec = importlib.util.spec_from_file_location("littlebit_itq", Path(__file__).with_name("littlebit_itq.py"))
_itq = importlib.util.module_from_spec(_itq_spec)
_itq_spec.loader.exec_module(_itq)

_budget_spec = importlib.util.spec_from_file_location("littlebit_budget", Path(__file__).with_name("littlebit_budget.py"))
_budget = importlib.util.module_from_spec(_budget_spec)
_budget_spec.loader.exec_module(_budget)
bits_per_weight = _budget.bits_per_weight
rank_for_bpw = _budget.rank_for_bpw


def _binary(z):
    return _train.ste(z, torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z)))


def _positive_int(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class _FactorPath(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.V = nn.Parameter(torch.randn(in_features, rank))
        self.h = nn.Parameter(torch.ones(out_features))
        self.g = nn.Parameter(torch.ones(in_features))
        self.l = nn.Parameter(torch.full((rank,), 1 / math.sqrt(in_features * rank)))


def _magnitude_scales(factor):
    """Nonnegative best rank-one magnitude envelope, split singular value evenly."""
    left, singular, right = torch.linalg.svd(factor.abs(), full_matrices=False)
    root = singular[0].sqrt()
    return left[:, 0].abs() * root, right[0].abs() * root


def _initialize(path, target, rank, itq_iters=None, seed=0):
    work = target.float() if target.dtype in (torch.float16, torch.bfloat16) else target
    left, singular, right = torch.linalg.svd(work, full_matrices=False)
    root = singular[:rank].sqrt()
    u = left[:, :rank] * root
    v = right[:rank, :].T * root
    if itq_iters is not None:
        rotation = _itq.joint_itq(u.cpu(), v.cpu(), iters=itq_iters, seed=seed).to(u.device)
        u, v = u @ rotation, v @ rotation
    h, lu = _magnitude_scales(u)
    g, lv = _magnitude_scales(v)
    path.U.copy_(u)
    path.V.copy_(v)
    path.l.copy_(lu * lv)
    path.h.copy_(h)
    path.g.copy_(g)


class FactorLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, residual=False):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features), ("rank", rank)):
            _positive_int(name, value)
        if rank > min(in_features, out_features):
            raise ValueError("rank exceeds matrix dimensions")
        if type(residual) is not bool:
            raise TypeError("residual must be bool")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.residual = residual
        self.paths = nn.ModuleList(_FactorPath(in_features, out_features, rank) for _ in range(1 + residual))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("x must be a floating tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x must have final dimension in_features")
        result = None
        for path in self.paths:
            y = ((((x * path.g) @ _binary(path.V)) * path.l) @ _binary(path.U).T) * path.h
            result = y if result is None else result + y
        return result

    def dense(self):
        result = None
        for path in self.paths:
            weight = path.h[:, None] * ((_binary(path.U) * path.l) @ _binary(path.V).T) * path.g[None, :]
            result = weight if result is None else result + weight
        return result

    @classmethod
    def from_dense(cls, W, rank, residual=False, *, itq_iters=None, seed=0):
        if not isinstance(W, torch.Tensor) or not W.is_floating_point():
            raise TypeError("W must be a floating tensor")
        if W.layout != torch.strided or W.ndim != 2 or not all(W.shape):
            raise ValueError("W must be a nonempty strided matrix")
        if not torch.isfinite(W).all():
            raise ValueError("W must be finite")
        layer = cls(W.shape[1], W.shape[0], rank, residual).to(device=W.device, dtype=W.dtype)
        with torch.no_grad():
            target = W.detach()
            _initialize(layer.paths[0], target, rank, itq_iters, seed)
            if residual:
                layer.fit_residual_(target, itq_iters=itq_iters, seed=seed)
        return layer

    @torch.no_grad()
    def fit_residual_(self, W, *, itq_iters=None, seed=0):
        """Refit only the second pair to W minus the current primary pair."""
        if not self.residual:
            raise ValueError("layer has no residual path")
        if W.shape != (self.out_features, self.in_features) or not torch.isfinite(W).all():
            raise ValueError("W must be a finite matrix with the layer shape")
        path = self.paths[0]
        primary = path.h[:, None] * ((_binary(path.U) * path.l) @ _binary(path.V).T) * path.g[None, :]
        _initialize(self.paths[1], W.detach().to(primary) - primary, self.rank, itq_iters, seed)
        return self

    def storage_bytes(self, scale_bits=16):
        _positive_int("scale_bits", scale_bits)
        if scale_bits % 8:
            raise ValueError("scale_bits must be a multiple of 8")
        return _budget.storage_bytes(self.rank, self.in_features, self.out_features, self.residual, scale_bits)

    def bits_per_weight(self):
        return 8 * self.storage_bytes() / (self.in_features * self.out_features)
