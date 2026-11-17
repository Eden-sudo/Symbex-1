"""The fixed subject split of the UniMiB-SHAR benchmark."""
import builtins
import importlib.util
import os

try:
    import numpy as np
except ImportError as exc:
    raise ImportError("NumPy is required for the HAR subject split tests") from exc


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAR_DATA = os.path.join(REPO, "benchmarks", "conv1d", "har_ultracompact", "har_data.py")


def load_har_data():
    spec = importlib.util.spec_from_file_location("har_data_for_subject_split", HAR_DATA)
    assert spec is not None and spec.loader is not None, "Cannot load the HAR data module by path"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_subjects_matches_original():
    har_data = load_har_data()
    order = np.random.default_rng(0).permutation(np.arange(1, 31))
    expected = (np.sort(order[:har_data.N_TEST]), np.sort(order[har_data.N_TEST:har_data.N_TEST + har_data.N_VAL]),
                np.sort(order[har_data.N_TEST + har_data.N_VAL:]))
    for actual, original in zip(har_data.split_subjects(), expected):
        np.testing.assert_array_equal(actual, original)
        assert actual.dtype == original.dtype


def test_split_subjects_partition():
    har_data = load_har_data()
    subjects = har_data.split_subjects()
    assert isinstance(subjects, tuple) and len(subjects) == 3
    for array, size in zip(subjects, (har_data.N_TEST, har_data.N_VAL, 30 - har_data.N_TEST - har_data.N_VAL)):
        assert array.ndim == 1 and len(array) == size
        np.testing.assert_array_equal(array, np.sort(array))
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.intersect1d(subjects[i], subjects[j]).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate(subjects)), np.arange(1, 31))


def test_split_subjects_repeatable():
    har_data = load_har_data()
    state_before = np.random.get_state()
    first = har_data.split_subjects()
    second = har_data.split_subjects()
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    expected = tuple(array.copy() for array in first)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
        a[0] = -1
    for actual, original in zip(har_data.split_subjects(), expected):
        np.testing.assert_array_equal(actual, original)


def test_import_and_split_without_scipy(monkeypatch):
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("HAR subject split must not import SciPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    assert len(load_har_data().split_subjects()) == 3
