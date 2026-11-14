"""Multiclass FOMO scores on small hand-made grids, without VOC data."""
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOMO_DIR = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person")
ZERO = dict(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=0)
PERFECT = dict(precision=100.0, recall=100.0, f1=100.0, tp=1, fp=0, fn=0)


@pytest.fixture
def fomo(monkeypatch):
    # The adapter imports the generic name model; isolate it and the inserted path.
    monkeypatch.syspath_prepend(FOMO_DIR)
    monkeypatch.delitem(sys.modules, "model", raising=False)
    spec = importlib.util.spec_from_file_location("fomo_mc_metrics", os.path.join(FOMO_DIR, "fomo_mc_metrics.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("model", None)


def grid(*cells, value=1):
    g = np.zeros((12, 12), dtype=np.uint8)
    for r, c in cells:
        g[r, c] = value
    return g


def truth(*points, image=0):
    return np.array(points, dtype=np.float32).reshape(-1, 2), np.full(len(points), image, dtype=np.int32)


def test_binary_parity(fomo):
    grids = np.array([grid((0, 0), (5, 5), (11, 0)), grid()])
    centers = np.array([(4, 4), (44, 44), (80, 80)], dtype=np.float32)
    owner = np.array([0, 0, 1], dtype=np.int32)
    result = fomo.f1_per_class(grids, centers, owner, np.ones(3, np.int32), 1)
    expected = fomo.f1_score(grids, centers, owner)
    assert (expected["tp"], expected["fp"], expected["fn"]) == (2, 1, 1)
    assert result == {"per_class": [expected], "macro_f1": expected["f1"]}


def test_wrong_class(fomo):
    centers, owner = truth((44, 20))
    result = fomo.f1_per_class(np.array([grid((2, 5), value=2)]), centers, owner, np.array([1]), 2)
    assert result == {"per_class": [dict(ZERO, fn=1), dict(ZERO, fp=1)], "macro_f1": 0.0}


def test_empty_class_counts_in_macro(fomo):
    centers, owner = truth((44, 20))
    result = fomo.f1_per_class(np.array([grid((2, 5))]), centers, owner, np.array([1]), 2)
    assert result == {"per_class": [PERFECT, ZERO], "macro_f1": 50.0}


@pytest.mark.parametrize("n_images", [0, 2])
def test_all_empty(fomo, n_images):
    centers, owner = truth()
    result = fomo.f1_per_class(np.zeros((n_images, 12, 12), np.uint8), centers, owner, np.array([], np.int32), 3)
    assert result == {"per_class": [ZERO, ZERO, ZERO], "macro_f1": 0.0}


def test_owners_filtering_and_no_mutation(fomo):
    grids = np.array([grid((2, 5), value=2), grid(), grid((0, 0))])
    centers = np.array([(44, 20), (4, 4), (80, 80)], dtype=np.float32)
    owner = np.array([1, 2, 1], dtype=np.int32)
    classes = np.array([2, 1, 1], dtype=np.int32)
    inputs = (grids, centers, owner, classes)
    copies = [a.copy() for a in inputs]
    result = fomo.f1_per_class(*inputs, 2)
    again = fomo.f1_per_class(*inputs, 2)
    assert result["per_class"][0] == pytest.approx(dict(precision=100.0, recall=50.0, f1=200 / 3, tp=1, fp=0, fn=1))
    assert result["per_class"][1] == dict(ZERO, fp=1, fn=1)
    assert result["macro_f1"] == pytest.approx(100 / 3)
    assert result == again and result is not again and result["per_class"][0] is not again["per_class"][0]
    assert all(np.array_equal(a, b) for a, b in zip(inputs, copies))


def test_adjacent_classes_are_separate_components(fomo):
    g = grid((2, 5))
    g[2, 6] = 2
    centers, owner = truth((44, 20), (52, 20))
    result = fomo.f1_per_class(np.array([g]), centers, owner, np.array([1, 2]), 2)
    assert result == {"per_class": [PERFECT, PERFECT], "macro_f1": 100.0}


@pytest.mark.parametrize("distance, expected", [(12.0, PERFECT), (12.01, dict(ZERO, fp=1, fn=1))])
def test_matching_distance(fomo, distance, expected):
    grids = np.array([grid((2, 5), value=2)])
    centers, owner = truth((44 + distance, 20))
    classes = np.array([2])
    result = fomo.f1_per_class(grids, centers, owner, classes, 2)
    assert result["per_class"][1] == expected == fomo.f1_score(grids == 2, centers[classes == 2], owner[classes == 2])


@pytest.mark.parametrize("n_classes, error", [(0, ValueError), (-1, ValueError), (1.0, TypeError),
                                                ("1", TypeError), (True, TypeError), (np.bool_(False), TypeError)])
def test_invalid_class_count(fomo, n_classes, error):
    centers, owner = truth()
    with pytest.raises(error, match="^n_classes must be a positive integer$"):
        fomo.f1_per_class(np.zeros((0, 12, 12), np.uint8), centers, owner, np.array([], np.int32), n_classes)


def test_numpy_integer_class_count(fomo):
    centers, owner = truth()
    result = fomo.f1_per_class(np.zeros((0, 12, 12), np.uint8), centers, owner, np.array([], np.int32), np.int64(1))
    assert result == {"per_class": [ZERO], "macro_f1": 0.0}
