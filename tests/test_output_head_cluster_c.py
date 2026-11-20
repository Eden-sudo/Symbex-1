import ctypes
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"
spec = importlib.util.spec_from_file_location("cluster_reference", BASE / "cluster_head.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    output = tmp_path_factory.mktemp("cluster-c") / "head.dylib"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", str(BASE / "c/cluster_head.c"), "-o", str(output)],
                   check=True, capture_output=True, text=True, timeout=15)
    library = ctypes.CDLL(str(output))
    fn = library.symbex_cluster_head
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4
    fn.restype = ctypes.c_int
    return fn


def call(kernel, h, w, labels, centers, k):
    ids = np.full(len(w), -99, dtype=np.int32)
    scores = np.full(len(w), -99, dtype=np.int64)
    selected = np.full(max(k, 1), -99, dtype=np.int32)
    count = np.array([-99], dtype=np.int32)
    status = kernel(*(a.ctypes.data for a in (h, w, labels, centers)),
                    len(w), len(h), len(centers), k,
                    *(a.ctypes.data for a in (ids, scores, selected, count)))
    return status, ids, scores, selected, count


def test_bit_exact_random_extremes_and_ties(kernel):
    rng = np.random.default_rng(19)
    w = rng.integers(-128, 128, (29, 13), dtype=np.int8)
    labels, centers = m.cluster_embeddings(w, 5)
    inputs = [np.zeros(13, dtype=np.int8), np.full(13, -128, dtype=np.int8),
              np.full(13, 127, dtype=np.int8)]
    inputs += [rng.integers(-128, 128, 13, dtype=np.int8) for _ in range(10)]
    for h in inputs:
        for k in (1, 3, 5):
            expected = m.shortlist_logits(h, w, labels, centers, k)
            status, ids, scores, selected, count = call(kernel, h, w, labels, centers, k)
            assert status == 0 and count[0] == len(expected[0])
            np.testing.assert_array_equal(ids[:count[0]], expected[0])
            np.testing.assert_array_equal(scores[:count[0]], expected[1])
            np.testing.assert_array_equal(selected, expected[2])
            assert np.all(ids[count[0]:] == -99)
            assert np.all(scores[count[0]:] == -99)


def test_reject_before_writing_and_empty_shortlist(kernel):
    h = np.ones(2, dtype=np.int8)
    w = np.ones((3, 2), dtype=np.int8)
    centers = np.array([[1, 1], [2, 2]], dtype=np.int8)
    for labels, k, error in (([0, 0, 2], 1, -1), ([0, 0, 0], 0, -1),
                             ([0, 0, 0], 3, -1), ([0, 0, 0], 1, -2)):
        result = call(kernel, h, w, np.array(labels, dtype=np.int32), centers, k)
        assert result[0] == error
        assert all(np.all(a == -99) for a in result[1:])


def test_maximum_dimension_accumulation(kernel):
    h = np.full(65536, -128, dtype=np.int8)
    w = np.stack([h, np.full_like(h, 127)])
    labels = np.array([0, 0], dtype=np.int32)
    status, ids, scores, _, count = call(kernel, h, w, labels, w[:1], 1)
    assert status == 0 and count[0] == 2 and ids.tolist() == [0, 1]
    assert scores.tolist() == [65536 * 16384, -65536 * 16256]
