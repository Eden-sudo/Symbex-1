"""Synthetic beat counts for the ECG validation split; these are not MIT-BIH measurements."""
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECG_DATA = os.path.join(REPO, "benchmarks", "conv1d", "ecg_mitbih", "ecg_data.py")


@pytest.fixture(scope="module")
def ecg():
    """the benchmark's ecg_data.py, loaded by path"""
    spec = importlib.util.spec_from_file_location("ecg_data_for_validation_split_test", ECG_DATA)
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load module from {ECG_DATA}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_counts(ecg, values):
    """independent synthetic N, S, V, F count arrays for every record"""
    return {r: np.array(values, dtype=np.int64) for r in ecg.DS1 + ecg.DS2}


def test_validation_records_are_sorted_distinct_and_training_only(ecg):
    counts = synthetic_counts(ecg, [0, 50, 50, 10])
    val = ecg.pick_validation(counts)
    assert isinstance(val, list)
    assert len(val) == ecg.VAL_RECORDS
    assert val == sorted(val)
    assert len(set(val)) == ecg.VAL_RECORDS
    assert set(val).issubset(ecg.DS1)
    assert set(val).isdisjoint(ecg.DS2)
    assert set(val).isdisjoint({208, 209})
    assert np.all(np.sum([counts[r] for r in val], axis=0)[1:] >= ecg.MIN_VAL)


def test_validation_is_repeatable_and_uses_first_seed_zero_draw(ecg):
    counts = synthetic_counts(ecg, [0, 50, 50, 10])
    pool = [r for r in ecg.DS1 if r not in (208, 209)]
    expected = sorted(np.random.default_rng(0).choice(pool, ecg.VAL_RECORDS, replace=False).tolist())
    assert ecg.pick_validation(counts) == ecg.pick_validation(counts) == expected


def test_validation_retries_until_enough_sparse_s_beats(ecg):
    pool = [r for r in ecg.DS1 if r not in (208, 209)]
    first = np.random.default_rng(0).choice(pool, ecg.VAL_RECORDS, replace=False).tolist()
    donors = set([r for r in pool if r not in first][:2])
    assert len(donors) == 2
    counts = synthetic_counts(ecg, [0, 0, 50, 10])
    for r in donors:
        counts[r][1] = 25
    rng = np.random.default_rng(0)
    for attempt in range(10000):
        candidate = sorted(rng.choice(pool, ecg.VAL_RECORDS, replace=False).tolist())
        if donors.issubset(candidate):
            expected = candidate
            break
    else:
        pytest.fail("synthetic sparse-S fixture has no valid seed-0 draw")
    assert attempt > 0
    val = ecg.pick_validation(counts)
    assert val == expected
    assert donors.issubset(val)
    have = np.sum([counts[r] for r in val], axis=0)
    assert have[1] == 50
    assert np.all(have[1:] >= ecg.MIN_VAL)


def test_zero_counts_raise_runtime_error(ecg):
    counts = synthetic_counts(ecg, [0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="^no validation draw holds enough S, V and F beats$"):
        ecg.pick_validation(counts)
