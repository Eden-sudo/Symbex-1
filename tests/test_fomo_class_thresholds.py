"""Per-class FOMO decisions and independent validation threshold fitting."""
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOMO_DIR = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person")


@pytest.fixture(scope="module")
def thresholds():
    old_path = sys.path[:]
    old_model = sys.modules.pop("model", None)
    try:
        spec = importlib.util.spec_from_file_location("fomo_thresholds", os.path.join(FOMO_DIR, "fomo_thresholds.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.modules.pop("model", None)
        if old_model is not None:
            sys.modules["model"] = old_model


def empty_truth():
    return np.empty((0, 2)), np.array([], dtype=int), np.array([], dtype=int)


def test_binary_rule_and_boundaries(thresholds):
    logits = np.random.default_rng(17).integers(-5, 6, (2, 2, 12, 12))
    for t in [-2, 0, 1, 2.5]:
        x = logits.astype(float)
        x[0, 0, 0, :3] = 0
        x[0, 1, 0, :3] = [t - 0.5, t, t + 0.5]
        result = thresholds.decide(x, [t])
        assert result.shape == (2, 12, 12)
        assert np.issubdtype(result.dtype, np.integer)
        np.testing.assert_array_equal(result, x[:, 1] - x[:, 0] > t)
        assert result[0, 0, :3].tolist() == [0, 0, 1]


def test_multiclass_ties_and_fractional_integer_inputs(thresholds):
    logits = np.zeros((2, 3, 12, 12), dtype=int)
    logits[:, :, 0, :4] = np.array([[2, 0, 0, 4], [3, 2, 0, 5], [2, 2, 0, 5]])
    result = thresholds.decide(logits, [1, 0])
    assert result[:, 0, :4].tolist() == [[0, 2, 0, 2]] * 2
    assert thresholds.decide(logits, [0.5, -0.5])[0, 0, :4].tolist() == [1, 2, 2, 2]


def test_known_optimum_and_stability(thresholds):
    logits = np.full((2, 3, 12, 12), -10.0)
    logits[:, 0] = 0
    logits[0, 1, 2, 2], logits[0, 1, 9, 9] = 2, 0.5
    logits[1, 2, 2, 2], logits[1, 2, 9, 9] = -0.5, -1.5
    centers = np.array([[20, 20], [20, 20]])
    owner, classes = np.array([0, 1]), np.array([1, 2])
    candidates = np.array([-2, -1, 0, 1, 2])
    before = [a.copy() for a in (logits, centers, owner, classes, candidates)]
    for _ in range(2):
        assert thresholds.fit(logits, centers, owner, classes, candidates) == [1.0, -1.0]
    for a, b in zip((logits, centers, owner, classes, candidates), before):
        np.testing.assert_array_equal(a, b)


def test_candidate_ties_and_empty_batch(thresholds):
    logits = np.zeros((0, 3, 12, 12))
    assert thresholds.decide(logits, [0, 0]).shape == (0, 12, 12)
    for candidates, expected in [([2, -1, 1, -1], -1), ([1, -1], 1), ([-1, 1], -1), ([3], 3)]:
        assert thresholds.fit(logits, *empty_truth(), candidates) == [expected, expected]
    assert thresholds.fit(logits, *empty_truth()) == [0, 0]


def test_default_quarter_step_and_absent_class(thresholds):
    logits = np.full((1, 3, 12, 12), -10.0)
    logits[:, 0] = 0
    logits[0, 1, 2, 2], logits[0, 1, 9, 9] = 1, 0.25
    assert thresholds.fit(logits, [[20, 20]], [0], [1]) == [0.25, 0.0]


def test_original_owners_and_class_filtering(thresholds):
    logits = np.full((3, 3, 12, 12), -10.0)
    logits[:, 0] = 0
    logits[0, 1, 9, 9] = 0.5
    logits[2, 1, 2, 2] = 1
    logits[1, 2, 9, 9] = 1
    centers = np.array([[76, 76], [20, 20]])
    assert thresholds.fit(logits, centers, [1, 2], [2, 1], [0, 0.5, 1]) == [0.5, 0.0]


def test_competing_classes_are_fit_independently(thresholds):
    logits = np.full((1, 3, 12, 12), -10.0)
    logits[:, 0] = 0
    logits[0, 1, 2, 2], logits[0, 2, 2, 2] = 1, 0.25
    logits[0, 1, 9, 9] = 0.5
    centers = np.array([[20, 20], [20, 20]])
    owners, classes = np.array([0, 0]), np.array([1, 2])
    candidates = [-1, -0.5, 0, 0.5, 1]

    def best(k, other, binary=False):
        scores = []
        for candidate in candidates:
            trial = [other, 0] if k == 2 else [0, other]
            trial[k - 1] = candidate
            if binary:
                grid = thresholds.decide(logits[:, [0, k]], [candidate]) == 1
            else:
                grid = thresholds.decide(logits, trial) == k
            score = thresholds.f1_score(grid, centers[classes == k], owners[classes == k])["f1"]
            scores.append((score, -abs(candidate)))
        return candidates[max(range(len(candidates)), key=lambda i: scores[i])]

    assert thresholds.fit(logits, centers, owners, classes, candidates) == [best(1, 0), best(2, 0)] == [0.5, -1]
    assert best(2, 0.5) == -0.5  # sequential search
    assert best(2, 0, binary=True) == 0  # background-only search


@pytest.mark.parametrize("argument,value", [
    ("logits", np.zeros((1, 2, 11, 12))), ("logits", np.zeros((1, 1, 12, 12))),
    ("logits", np.full((1, 2, 12, 12), np.nan)), ("logits", np.zeros((1, 2, 12, 12), dtype=complex)),
    ("centers", [1, 2]), ("centers", [[np.inf, 0]]), ("owner", [0.5]),
    ("owner", [1]), ("owner", []), ("center_class", [0]), ("center_class", [2]),
    ("candidates", []), ("candidates", [[0]]), ("candidates", ["0"]),
])
def test_invalid_fit_inputs(thresholds, argument, value):
    args = dict(logits=np.zeros((1, 2, 12, 12)), centers=np.array([[20, 20]]),
                owner=np.array([0]), center_class=np.array([1]), candidates=np.array([0]))
    args[argument] = value
    with pytest.raises(ValueError, match=argument):
        thresholds.fit(**args)


@pytest.mark.parametrize("value", [[], [[0]], [np.inf], ["0"], [1j]])
def test_invalid_thresholds_do_not_mutate_logits(thresholds, value):
    logits = np.zeros((1, 2, 12, 12))
    before = logits.copy()
    with pytest.raises(ValueError, match="thresholds"):
        thresholds.decide(logits, value)
    np.testing.assert_array_equal(logits, before)
