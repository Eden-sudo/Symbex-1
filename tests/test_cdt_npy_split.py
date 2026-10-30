"""The CDT's npy data format (tools/cdt.py, load_data): optional val.npy and test.npy masks.

test.npy holds samples out (evaluated once, at the end); val.npy is the validation set, else validation_split of what
is left. The three sets must never share a sample: a small synthetic dataset checks it by the sample values.
"""
import copy
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import cdt  # noqa: E402

N = 50


def config(folder):
    cfg = copy.deepcopy(cdt.DEFAULTS)
    cfg["data"].update(format="npy", directory=str(folder), mean=[], std=[], input_bits=8, split_seed=0,
                       validation_split=0.2)
    return cfg


def write(folder, val=None, test=None):
    x = np.arange(N, dtype=np.float32).reshape(N, 1)       # the value identifies the sample
    np.save(folder / "x.npy", x)
    np.save(folder / "y.npy", np.arange(N) % 3)
    if val is not None:
        np.save(folder / "val.npy", val)
    if test is not None:
        np.save(folder / "test.npy", test)


def ids(ds):
    return set(int(ds[i][0].item()) for i in range(len(ds)))


def masks():
    test, val = np.zeros(N, bool), np.zeros(N, bool)
    test[:10], val[10:18] = True, True
    return val, test


def test_val_and_test_masks(tmp_path):
    val, test = masks()
    write(tmp_path, val, test)
    tr, va, te = cdt.load_data(config(tmp_path))
    assert ids(te) == set(range(10)) and ids(va) == set(range(10, 18)) and ids(tr) == set(range(18, N))


def test_test_mask_only_splits_the_rest(tmp_path):
    _, test = masks()
    write(tmp_path, test=test)
    tr, va, te = cdt.load_data(config(tmp_path))
    assert ids(te) == set(range(10))
    assert len(va) == int(40 * 0.2) and not (ids(va) & ids(tr)) and ids(va) | ids(tr) == set(range(10, N))


def test_no_masks_unchanged(tmp_path):
    write(tmp_path)
    tr, va, te = cdt.load_data(config(tmp_path))
    assert te is None and len(tr) + len(va) == N


def test_overlap_rejected(tmp_path):
    val, test = masks()
    val[0] = True                                          # also a test sample
    write(tmp_path, val, test)
    with pytest.raises(SystemExit):
        cdt.load_data(config(tmp_path))
