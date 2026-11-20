"""CPU Joint-ITQ based on https://arxiv.org/abs/2603.00042.

Section 4.4, equation 9 and Algorithm 1 concatenate the factors without
row normalization. The Procrustes update below uses the equivalent SVD of
Z.T @ B rather than B.T @ Z. Row scales follow Appendix A.1, equation 12.
"""

import torch


def _tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.layout != torch.strided or value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a dense float32 or float64 tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


def _factors(U, V):
    for value, name in ((U, "U"), (V, "V")):
        _tensor(value, name)
        if value.ndim != 2 or 0 in value.shape:
            raise ValueError(f"{name} must be two-dimensional with positive dimensions")
    if U.dtype != V.dtype:
        raise ValueError("U and V must have matching dtype")
    if U.shape[1] != V.shape[1]:
        raise ValueError("U and V must have matching rank")


def _sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


@torch.no_grad()
def joint_itq(U, V, iters=50, seed=0):
    """Find a shared orthogonal rotation for two CPU factor matrices."""
    _factors(U, V)
    for value, name in ((iters, "iters"), (seed, "seed")):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if iters < 0:
        raise ValueError("iters must be nonnegative")
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2**63 - 1")
    rank = U.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, triangular = torch.linalg.qr(
        torch.randn(rank, rank, dtype=torch.float64, generator=generator)
    )
    rotation = q * _sign(triangular.diagonal())
    z = torch.cat((U.to(torch.float64), V.to(torch.float64)))
    for _ in range(iters):
        binary = _sign(z @ rotation)
        cross = z.T @ binary
        if not torch.isfinite(cross).all():
            raise ArithmeticError("nonfinite Joint-ITQ intermediate")
        left, _, right_t = torch.linalg.svd(cross, full_matrices=False)
        rotation = left @ right_t
    return rotation.to(U.dtype)


@torch.no_grad()
def binarization_error(U, V, R):
    """Return relative product error with least-squares factor-row scales."""
    _factors(U, V)
    _tensor(R, "R")
    rank = U.shape[1]
    if R.ndim != 2 or R.shape != (rank, rank):
        raise ValueError("R must have shape (rank, rank)")
    if R.dtype != U.dtype:
        raise ValueError("R must have matching dtype")
    rotation = R.to(torch.float64)
    if not torch.allclose(rotation @ rotation.T, torch.eye(rank, dtype=torch.float64), atol=1e-5, rtol=0):
        raise ValueError("R must be orthogonal")
    u, v = U.to(torch.float64), V.to(torch.float64)
    x, y = u @ rotation, v @ rotation
    target = u @ v.T
    approximation = (x.abs().mean(1, keepdim=True) * _sign(x)) @ (
        y.abs().mean(1, keepdim=True) * _sign(y)
    ).T
    if not all(torch.isfinite(t).all() for t in (x, y, target, approximation)):
        raise ArithmeticError("nonfinite binarization intermediate")
    denominator = torch.linalg.vector_norm(target)
    numerator = torch.linalg.vector_norm(target - approximation)
    if not torch.isfinite(denominator) or not torch.isfinite(numerator):
        raise ArithmeticError("nonfinite binarization norm")
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    result = (numerator / denominator).item()
    if not torch.isfinite(torch.tensor(result, dtype=torch.float64)):
        raise ArithmeticError("nonfinite binarization error")
    return result
