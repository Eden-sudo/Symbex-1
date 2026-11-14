"""Host-side checks for the FOMO connected-component API."""
import ctypes as ct
import importlib.util
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'Symbex1_PIO/lib/lib_symbex'


class Center(ct.Structure):
    _fields_ = [('class_id', ct.c_uint8), ('n', ct.c_uint32),
                ('sum_r', ct.c_uint32), ('sum_c', ct.c_uint32)]


def values(c):
    return c.class_id, c.n, c.sum_r, c.sum_c


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    gcc = shutil.which('gcc')
    assert gcc, 'gcc not available'
    tmp = tmp_path_factory.mktemp('fomo')
    probe = tmp / 'probe.c'
    probe.write_text('#include "symbex.h"\nint probe(void) { struct SymbexFomoCenter c; '
                     'return symbex_fomo_centers(0, 0, 0, &c, 1, 0); }\n')
    so = tmp / 'fomo.so'
    subprocess.run([gcc, '-std=c99', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I', str(LIB / 'include'), str(LIB / 'src/layers/conv/symbex_fomo.c'),
                    str(probe), '-o', str(so)], check=True)
    fn = ct.CDLL(str(so)).symbex_fomo_centers
    fn.argtypes = [ct.POINTER(ct.c_uint8), ct.c_uint16, ct.c_uint16,
                   ct.POINTER(Center), ct.c_int, ct.POINTER(ct.c_uint16)]
    fn.restype = ct.c_int
    return fn


def reference(grid):
    h, w = len(grid), len(grid[0])
    seen, result = set(), []
    for r in range(h):
        for c in range(w):
            cls = grid[r][c]
            if not cls or (r, c) in seen:
                continue
            todo, seen_component = [(r, c)], []
            seen.add((r, c))
            while todo:
                y, x = todo.pop()
                seen_component.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        p = y + dy, x + dx
                        if (0 <= p[0] < h and 0 <= p[1] < w and p not in seen
                                and grid[p[0]][p[1]] == cls):
                            seen.add(p)
                            todo.append(p)
            result.append((cls, len(seen_component), sum(p[0] for p in seen_component),
                           sum(p[1] for p in seen_component)))
    return result


def run(kernel, grid, capacity=None):
    h, w = len(grid), len(grid[0])
    flat = bytes(v for row in grid for v in row)
    cells = (ct.c_uint8 * len(flat)).from_buffer_copy(flat)
    scratch = (ct.c_uint16 * (len(flat) + 2))(*([0xaaaa] * (len(flat) + 2)))
    capacity = len(flat) if capacity is None else capacity
    output = (Center * (capacity + 2))(*([Center(99, 98, 97, 96)] * (capacity + 2)))
    ptr = ct.cast(ct.byref(scratch, 2), ct.POINTER(ct.c_uint16))
    out = ct.cast(ct.byref(output, ct.sizeof(Center)), ct.POINTER(Center))
    total = kernel(cells, h, w, out, capacity, ptr)
    assert bytes(cells) == flat
    assert scratch[0] == scratch[-1] == 0xaaaa
    assert values(output[0]) == values(output[-1]) == (99, 98, 97, 96)
    for i in range(min(total, capacity) + 1, capacity + 1):
        assert values(output[i]) == (99, 98, 97, 96)
    return total, [values(output[i + 1]) for i in range(min(total, capacity))]


def test_random_and_examples(kernel):
    rng = random.Random(1337)
    examples = [[[0]], [[1]], [[1, 0], [0, 1]], [[1, 0], [0, 2]],
                [[1, 2, 255]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                [[1, 0, 1], [1, 1, 1]], [[0, 1], [1, 0]],
                [[1, 0, 0, 0, 1]], [[1], [0], [1]], [[0, 1, 0], [0, 0, 1]]]
    for _ in range(500):
        h, w = rng.randint(1, 12), rng.randint(1, 12)
        examples.append([[rng.randrange(4) for _ in range(w)] for _ in range(h)])
    for grid in examples:
        expected = reference(grid)
        for capacity in {0, 1, max(0, len(expected) - 1), len(expected), len(expected) + 1}:
            total, result = run(kernel, grid, capacity)
            assert total == len(expected)
            assert result == expected[:capacity]


def test_model_centers(kernel):
    spec = importlib.util.spec_from_file_location('fomo_model', ROOT / 'benchmarks/conv2d/fomo_voc_person/model.py')
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    rng = random.Random(2025)
    grids = [[[0] * 12 for _ in range(12)], [[int((r, c) in {(0, 0), (1, 0), (1, 1)})
                                               for c in range(12)] for r in range(12)]]
    grids += [[[rng.randrange(2) for _ in range(12)] for _ in range(12)] for _ in range(100)]
    for grid in grids:
        _, groups = run(kernel, grid)
        centers = [((sc / n + .5) * model.CELL, (sr / n + .5) * model.CELL)
                   for _, n, sr, sc in groups]
        assert centers == model.detections(np.array(grid, dtype=np.uint8))


def test_limits(kernel):
    for h, w in [(1, 65535), (65535, 1), (255, 257)]:
        grid = [[1] * w for _ in range(h)]
        total, groups = run(kernel, grid, 1)
        assert total == 1
        assert groups == [(1, h * w, w * h * (h - 1) // 2, h * w * (w - 1) // 2)]
    grid = [[1 + i % 2 for i in range(65535)]]
    total, groups = run(kernel, grid, 1)
    assert total == 65535 and groups == [(1, 1, 0, 0)]


def test_validation_and_reuse(kernel):
    cells = (ct.c_uint8 * 4)(1, 0, 0, 2)
    scratch = (ct.c_uint16 * 4)(11, 12, 13, 14)
    out = (Center * 2)(Center(5, 6, 7, 8), Center(5, 6, 7, 8))
    for args, expected in [((None, 0, 3, None, 0, None), 0),
                           ((None, 3, 0, None, 0, None), 0),
                           ((None, 0, 0, None, -1, None), -1),
                           ((cells, 2, 2, out, -1, scratch), -1),
                           ((None, 2, 2, out, 2, scratch), -1),
                           ((cells, 2, 2, out, 2, None), -1),
                           ((cells, 2, 2, None, 2, scratch), -1),
                           ((cells, 256, 256, out, 2, scratch), -1),
                           ((cells, 65535, 65535, out, 2, scratch), -1)]:
        assert kernel(*args) == expected
        assert list(scratch) == [11, 12, 13, 14]
        assert [values(c) for c in out] == [(5, 6, 7, 8)] * 2
        assert list(cells) == [1, 0, 0, 2]
    assert kernel(cells, 2, 2, None, 0, scratch) == 2
    assert kernel(cells, 2, 2, out, 1, scratch) == 2
    assert values(out[0]) == (1, 1, 0, 0)
    cells[:] = [2, 2, 2, 2]
    assert kernel(cells, 2, 2, out, 2, scratch) == 1
    assert values(out[0]) == (2, 4, 2, 2)
