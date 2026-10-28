"""The validation files of an export (compiler_core/export/dense.py: save_validation / load_validation).

chip_test.py and the CDT's chip stage compare what the board prints against validation_sim. Classes fit in a byte,
but a per-cell head (FOMO) prints the 31-bit checksum of its grid: those must survive the round trip whole, or every
image looks different from the simulation on a correct chip. Values above 255 go to .u32 (int32), the rest stays .u8.
"""
import os

import numpy as np
import pytest

from compiler_core.export import boolean
from compiler_core.export.dense import load_validation, save_validation


def _files(d):
    return sorted(f for f in os.listdir(d) if f.startswith("validation_"))


def test_classes_round_trip_as_u8(tmp_path, rng):
    x = rng.integers(0, 256, (5, 12), dtype=np.uint8)
    y, sim = np.array([0, 3, 9, 1, 255]), np.array([0, 3, 8, 1, 255])
    save_validation(tmp_path, x, y, sim)
    assert _files(tmp_path) == ["validation_sim.u8", "validation_x.u8", "validation_y.u8"]
    x2, y2, sim2 = load_validation(tmp_path, 12)
    assert (x2 == x).all() and (y2 == y).all() and (sim2 == sim).all()


def test_grid_checksums_above_255_round_trip_whole_as_u32(tmp_path, rng):
    grids = rng.integers(0, 2, (6, 12, 12), dtype=np.uint8)
    checks = np.array([boolean._grid_checksum(g) for g in grids])
    assert checks.max() > 255                                         # 31-bit FNV-1a: far past a byte
    x = rng.integers(0, 256, (6, 30), dtype=np.uint8)
    save_validation(tmp_path, x, checks, checks)
    assert _files(tmp_path) == ["validation_sim.u32", "validation_x.u8", "validation_y.u32"]
    x2, y2, sim2 = load_validation(tmp_path, 30)
    assert (x2 == x).all()
    assert sim2.dtype == np.int64 and (sim2 == checks).all() and (y2 == checks).all()


def test_u32_export_removes_the_stale_u8_and_back(tmp_path, rng):
    x = rng.integers(0, 256, (3, 4), dtype=np.uint8)
    save_validation(tmp_path, x, [1, 2, 3], [1, 2, 3])                # an old export with classes
    save_validation(tmp_path, x, [1, 2, 3], [7, 2 ** 31 - 1, 300])     # re-exported with checksums
    assert _files(tmp_path) == ["validation_sim.u32", "validation_x.u8", "validation_y.u8"]
    assert (load_validation(tmp_path, 4)[2] == [7, 2 ** 31 - 1, 300]).all()
    save_validation(tmp_path, x, [1, 2, 3], [4, 5, 6])                # and back to classes: no stale .u32 left
    assert _files(tmp_path) == ["validation_sim.u8", "validation_x.u8", "validation_y.u8"]
    assert (load_validation(tmp_path, 4)[2] == [4, 5, 6]).all()


@pytest.mark.parametrize("bad", [-1, 2 ** 31])
def test_values_outside_int32_are_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="0 .. 2\\^31 - 1"):
        save_validation(tmp_path, np.zeros((1, 2), np.uint8), [0], [bad])
