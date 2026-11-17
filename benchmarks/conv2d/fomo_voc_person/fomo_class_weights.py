"""Class weights from the cell frequencies of multiclass FOMO grids."""

import numbers

import numpy as np


def class_weights(y, n_classes, max_weight=50.0):
    """Return float32 weights for an (N, 12, 12) integer class grid.

    Labels are 0 for background and 1 through n_classes for foreground. Each
    present foreground weight is the background count divided by its count,
    clipped to [1, max_weight]. Absent classes receive max_weight; background
    is always 1, including when there is no background or the batch is empty.

    Raise ValueError for invalid grids, class counts or maximum weights.
    """
    if isinstance(n_classes, (bool, np.bool_)) or not isinstance(n_classes, numbers.Integral) or n_classes < 1:
        raise ValueError("invalid n_classes")
    if (isinstance(max_weight, (bool, np.bool_)) or not isinstance(max_weight, numbers.Real)
            or max_weight < 1 or max_weight > float(np.finfo(np.float32).max)):
        raise ValueError("invalid max_weight")
    max_weight = float(max_weight)
    if not np.isfinite(max_weight):
        raise ValueError("invalid max_weight")
    try:
        y = np.asarray(y)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid y") from exc
    if y.ndim != 3 or y.shape[1:] != (12, 12) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("invalid y")
    if y.size and (np.any(y < 0) or np.any(y > n_classes)):
        raise ValueError("invalid y labels")

    counts = np.bincount(y.ravel().astype(np.intp), minlength=int(n_classes) + 1)
    weights = np.full(int(n_classes) + 1, max_weight, dtype=np.float64)
    np.divide(counts[0], counts, out=weights, where=counts > 0)
    np.clip(weights, 1.0, max_weight, out=weights)
    weights[0] = 1.0
    return weights.astype(np.float32)
