import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("cluster_head", Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/cluster_head.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_clustering_reproducible_and_separated():
    e = np.array([[10, 11], [12, 10], [-10, -11], [-12, -10]], dtype=np.int8)
    labels, centers = m.cluster_embeddings(e, 2, seed=9)
    again = m.cluster_embeddings(e, 2, seed=9)
    np.testing.assert_array_equal(labels, again[0])
    np.testing.assert_array_equal(centers, again[1])
    assert labels[0] == labels[1] != labels[2] == labels[3]
    assert centers.dtype == np.int8
    for c in range(2):
        np.testing.assert_array_equal(centers[c], np.rint(e[labels == c].mean(0)))


def test_integer_scores_full_coverage_and_conditional_argmax():
    rng = np.random.default_rng(4)
    w = rng.integers(-128, 128, (31, 7), dtype=np.int8)
    labels, centers = m.cluster_embeddings(w, 4)
    for _ in range(12):
        h = rng.integers(-128, 128, 7, dtype=np.int8)
        full = m.full_logits(h, w)
        oracle = [sum(int(a) * int(b) for a, b in zip(row, h)) for row in w]
        np.testing.assert_array_equal(full, oracle)
        for k in range(1, 5):
            ids, scores, selected = m.shortlist_logits(h, w, labels, centers, k)
            np.testing.assert_array_equal(scores, full[ids])
            assert len(selected) == k
            winner = int(full.argmax())
            if winner in ids:
                assert m.greedy(ids, scores) == winner
            if k == 4:
                np.testing.assert_array_equal(ids, np.arange(len(w)))


def test_ties_empty_and_missed_winner():
    w = np.array([[1], [10], [10]], dtype=np.int8)
    h = np.array([1], dtype=np.int8)
    centers = np.array([[2], [1], [3]], dtype=np.int8)
    labels = np.array([0, 1, 1])
    ids, scores, _ = m.shortlist_logits(h, w, labels, centers[:2], 1)
    assert m.greedy(ids, scores) == 0 != m.full_logits(h, w).argmax()
    ids, scores, _ = m.shortlist_logits(h, w, labels, centers, 3)
    assert m.greedy(ids, scores) == 1
    with pytest.raises(ValueError, match="no tokens"):
        m.shortlist_logits(h, w, labels, centers, 1)
    tied = np.zeros((2, 1), dtype=np.int8)
    assert m.shortlist_logits(h, w, labels, tied, 1)[2].tolist() == [0]


def test_invalid_contracts():
    w = np.ones((3, 2), dtype=np.int8)
    h = w[0]
    for labels, centers, k in (([0, 0, 4], w, 1), ([0, 0, 0], w, 0),
                                ([0.0, 0, 0], w, 1), ([0, 0, 0], w.astype(float), 1)):
        with pytest.raises(ValueError):
            m.shortlist_logits(h, w, labels, centers, k)
    with pytest.raises(ValueError):
        m.cluster_embeddings(w, 4)
