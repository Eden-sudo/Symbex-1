"""tiny_llm bpb.py: bytes counted over train.py's evaluation windows."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tiny_llm_bpb", os.path.join(HERE, "..", "benchmarks", "experiments", "tiny_llm", "bpb.py"))
bpb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpb)


def test_scored_bytes_counts_targets_of_each_window_and_drops_the_first_space_after_bos():
    bos = 1
    full = np.array([0, 1, 3, 5])     # id 2 is " ab", id 3 is " abcd"
    first = np.array([0, 1, 2, 4])    # the same pieces without their leading space
    ids = np.array([1, 2, 3, 3, 1, 2, 2, 3, 0], dtype=np.uint16)
    nbytes, tokens = bpb.scored_bytes(ids, seq=4, windows=2, sizes=(full, first), bos=bos)
    # window 0 targets 2 3 3 1 (that 2 follows BOS); window 1 targets 2 2 3 0 (its first 2 follows BOS)
    assert tokens == 8
    assert nbytes == (2 + 5 + 5 + 1) + (2 + 3 + 5 + 0)


def test_scored_bytes_never_reads_past_the_last_full_window():
    full = first = np.ones(4, dtype=np.int64)
    ids = np.array([1, 2, 3, 2, 3], dtype=np.uint16)
    nbytes, tokens = bpb.scored_bytes(ids, seq=4, windows=10, sizes=(full, first), bos=1)
    assert (nbytes, tokens) == (4, 4)
