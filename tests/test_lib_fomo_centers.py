"""The on-chip FOMO grouping and its public C interface."""
import ctypes as ct
import importlib.util
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "Symbex1_PIO/lib/lib_symbex"


class Center(ct.Structure):
    _fields_ = [("class_id", ct.c_uint8), ("n", ct.c_uint16),
                ("sum_r", ct.c_uint32), ("sum_c", ct.c_uint32)]


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    assert shutil.which("gcc"), "needs gcc on PATH"
    tmp = tmp_path_factory.mktemp("fomo")
    probe = tmp / "probe.c"
    probe.write_text('#include "symbex.h"\n'
                     'int probe(void) { struct SymbexFomoCenter c; '
                     'return symbex_fomo_centers(0, 0, 0, &c, 1, 0); }\n')
    so = tmp / "fomo.so"
    cmd = ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
           "-I", str(LIB / "include"), str(LIB / "src/layers/conv/symbex_fomo.c"),
           str(probe), "-o", str(so)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    fn = ct.CDLL(str(so)).symbex_fomo_centers
    fn.argtypes = [ct.POINTER(ct.c_uint8), ct.c_uint16, ct.c_uint16,
                   ct.POINTER(Center), ct.c_int, ct.POINTER(ct.c_uint16)]
    fn.restype = ct.c_int
    return fn


@pytest.fixture(scope="module")
def fomo():
    spec = importlib.util.spec_from_file_location(
        "fomo_model", ROOT / "benchmarks/conv2d/fomo_voc_person/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(grid):
    h, w = grid.shape
    seen, groups = set(), []
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if not value or (r, c) in seen:
                continue
            seen.add((r, c))
            pending, points = [(r, c)], []
            while pending:
                y, x = pending.pop()
                points.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if (0 <= yy < h and 0 <= xx < w and (yy, xx) not in seen
                                and grid[yy, xx] == value):
                            seen.add((yy, xx))
                            pending.append((yy, xx))
            groups.append((value, len(points), sum(p[0] for p in points), sum(p[1] for p in points)))
    return groups


def run_grid(kernel, grid):
    h, w = grid.shape
    cells = (ct.c_uint8 * grid.size)(*grid.flat)
    before = bytes(cells)
    scratch = (ct.c_uint16 * grid.size)()
    out = (Center * grid.size)()
    total = kernel(cells, h, w, out, grid.size, scratch)
    assert bytes(cells) == before
    return total, [(x.class_id, x.n, x.sum_r, x.sum_c) for x in out[:total]]


def test_random_multiclass(kernel):
    rng = random.Random(842)
    for _ in range(500):
        h, w = rng.randint(1, 12), rng.randint(1, 12)
        grid = np.array([rng.randrange(4) for _ in range(h * w)], dtype=np.uint8).reshape(h, w)
        total, groups = run_grid(kernel, grid)
        assert groups == reference(grid)
        assert total == len(groups)


@pytest.mark.parametrize("rows", [
    [[0, 0], [0, 0]], [[1]], [[1, 0], [0, 1]], [[1, 2], [2, 1]],
    [[1, 0], [1, 1]], [[3, 0, 1], [0, 0, 0], [2, 0, 3]],
    [[0, 0, 1], [1, 0, 0]], [[255, 0, 255]], [[1], [0], [1]],
])
def test_examples(kernel, rows):
    grid = np.array(rows, dtype=np.uint8)
    total, groups = run_grid(kernel, grid)
    assert groups == reference(grid)
    assert total == len(groups)


def test_binary_centers_match_model(kernel, fomo):
    rng = random.Random(719)
    grids = [np.array(rows, dtype=np.uint8) for rows in
             ([[0]], [[1]], [[1, 0], [1, 1]], [[1, 0, 1]], [[1, 0], [0, 1]])]
    for _ in range(150):
        h, w = rng.randint(1, 12), rng.randint(1, 12)
        grids.append(np.array([rng.randrange(2) for _ in range(h * w)], dtype=np.uint8).reshape(h, w))
    for grid in grids:
        _, groups = run_grid(kernel, grid)
        centers = [((sc / n + 0.5) * fomo.CELL, (sr / n + 0.5) * fomo.CELL)
                   for _, n, sr, sc in groups]
        assert centers == fomo.detections(grid)


def ptr(array, typ, offset=0):
    return ct.cast(ct.byref(array, offset), ct.POINTER(typ))


def test_truncation_and_guards(kernel):
    grid = np.array([[1, 0, 2, 0, 3, 0, 1]], dtype=np.uint8)
    expected = reference(grid)
    cells = (ct.c_uint8 * grid.size)(*grid.flat)
    size = ct.sizeof(Center)
    for cap in (0, 1, len(expected) - 1, len(expected), len(expected) + 2):
        out = (ct.c_uint8 * (size * 8))(*([0xA5] * (size * 8)))
        scratch = (ct.c_uint16 * (grid.size + 2))(*([0xBEEF] * (grid.size + 2)))
        total = kernel(cells, 1, grid.size, ptr(out, Center, size), cap, ptr(scratch, ct.c_uint16, 2))
        assert total == len(expected)
        records = ct.cast(ct.byref(out, size), ct.POINTER(Center))
        assert [(x.class_id, x.n, x.sum_r, x.sum_c) for x in records[:min(cap, total)]] == expected[:cap]
        assert bytes(out[:size]) == bytes([0xA5] * size)
        assert bytes(out[size * (1 + min(cap, total)):]) == bytes([0xA5] * (size * (7 - min(cap, total))))
        assert scratch[0] == scratch[-1] == 0xBEEF
    assert kernel(cells, 1, grid.size, None, 0, ptr(scratch, ct.c_uint16, 2)) == len(expected)


def test_invalid_and_empty(kernel):
    cells = (ct.c_uint8 * 4)(1, 2, 3, 4)
    scratch = (ct.c_uint16 * 4)(8, 9, 10, 11)
    out = (Center * 2)()
    ct.memset(ct.addressof(out), 0xA5, ct.sizeof(out))
    before = bytes(cells), bytes(scratch), bytes(out)
    for args in [(cells, 2, 2, out, -1, scratch), (None, 2, 2, out, 1, scratch),
                 (cells, 2, 2, out, 1, None), (cells, 2, 2, None, 1, scratch),
                 (cells, 256, 256, out, 1, scratch), (cells, 65535, 65535, out, 1, scratch)]:
        assert kernel(*args) == -1
        assert (bytes(cells), bytes(scratch), bytes(out)) == before
    for h, w in [(0, 0), (0, 65535), (65535, 0)]:
        assert kernel(None, h, w, None, 0, None) == 0
        assert kernel(None, h, w, out, 1, None) == 0
        assert kernel(None, h, w, out, -1, None) == -1
        assert kernel(None, h, w, None, 1, None) == -1
        assert (bytes(cells), bytes(scratch), bytes(out)) == before


@pytest.mark.parametrize("h,w", [(1, 65535), (65535, 1), (255, 257)])
def test_maximum_uniform_grid(kernel, h, w):
    n = h * w
    cells = (ct.c_uint8 * n)(*([1] * n))
    scratch = (ct.c_uint16 * (n + 2))(*([0xBEEF] * (n + 2)))
    out = (Center * 3)()
    ct.memset(ct.addressof(out), 0xA5, ct.sizeof(out))
    assert kernel(cells, h, w, ptr(out, Center, ct.sizeof(Center)), 1,
                  ptr(scratch, ct.c_uint16, 2)) == 1
    c = out[1]
    assert (c.class_id, c.n, c.sum_r, c.sum_c) == (1, n, w * h * (h - 1) // 2, h * w * (w - 1) // 2)
    assert scratch[0] == scratch[-1] == 0xBEEF
    assert bytes(out[0]) == bytes(out[2]) == bytes([0xA5] * ct.sizeof(Center))


def test_many_groups_and_scratch_reuse(kernel):
    n = 65535
    cells = (ct.c_uint8 * n)(*(1 + i % 2 for i in range(n)))
    scratch = (ct.c_uint16 * n)(*([42] * n))
    out = (Center * 3)()
    for cap in (0, 3, 3):
        assert kernel(cells, 1, n, out, cap, scratch) == n
        if cap:
            assert [(x.class_id, x.n, x.sum_r, x.sum_c) for x in out] == [(1, 1, 0, 0), (2, 1, 0, 1), (1, 1, 0, 2)]
        before = bytes(scratch), bytes(out), bytes(cells)
        assert kernel(cells, 1, n, out, -1, scratch) == -1
        assert (bytes(scratch), bytes(out), bytes(cells)) == before
