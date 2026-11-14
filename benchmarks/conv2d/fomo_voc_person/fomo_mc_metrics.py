"""Per-class FOMO detection scores using the binary evaluator."""
import numbers
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import f1_score  # noqa: E402


def f1_per_class(grids, centers, owner, center_class, n_classes):
    """Score classes 1..K; empty classes score zero and count in the macro mean."""
    if isinstance(n_classes, (bool, np.bool_)) or not isinstance(n_classes, numbers.Integral):
        raise TypeError("n_classes must be a positive integer")
    if n_classes < 1:
        raise ValueError("n_classes must be a positive integer")

    per_class = []
    for k in range(1, n_classes + 1):
        keep = center_class == k
        per_class.append(f1_score(grids == k, centers[keep], owner[keep]))
    return {
        "per_class": per_class,
        "macro_f1": float(sum(s["f1"] for s in per_class) / n_classes),
    }
