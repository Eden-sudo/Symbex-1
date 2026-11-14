"""Independent validation thresholds for the FOMO integer decision head."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import f1_score  # noqa: E402


def _real_array(value, name):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError
        return array
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"invalid {name}") from None


def _logits(value):
    array = _real_array(value, "logits")
    if array.ndim != 4 or array.shape[1] < 2 or array.shape[2:] != (12, 12):
        raise ValueError("invalid logits shape")
    return array


def _indices(value, name, count, low, high):
    array = _real_array(value, name)
    if (array.shape != (count,) or np.any(array != np.floor(array))
            or np.any(array < low) or np.any(array >= high)):
        raise ValueError(f"invalid {name}")
    return array.astype(np.int64)


def decide(logits, thresholds):
    """Choose the first maximum after subtracting each foreground threshold."""
    scores = _logits(logits).copy()
    thresholds = _real_array(thresholds, "thresholds")
    if thresholds.shape != (scores.shape[1] - 1,):
        raise ValueError("invalid thresholds shape")
    scores[:, 1:] -= thresholds[None, :, None, None]
    return scores.argmax(axis=1)


def fit(logits, centers, owner, center_class, candidates=np.linspace(-4, 4, 33)):
    """Fit each class independently using detection F1 and zero other thresholds."""
    logits = _logits(logits)
    centers = _real_array(centers, "centers")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("invalid centers shape")
    owner = _indices(owner, "owner", len(centers), 0, len(logits))
    classes = logits.shape[1] - 1
    center_class = _indices(center_class, "center_class", len(centers), 1, classes + 1)
    candidates = _real_array(candidates, "candidates")
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("invalid candidates shape")

    result = []
    for k in range(1, classes + 1):
        mask = center_class == k
        trial = np.zeros(classes)
        best_key = None
        for candidate in candidates:
            trial[k - 1] = candidate
            score = f1_score(decide(logits, trial) == k, centers[mask], owner[mask])["f1"]
            key = (score, -abs(float(candidate)))
            if best_key is None or key > best_key:
                best_key, best_t = key, float(candidate)
        result.append(best_t)
    return result
