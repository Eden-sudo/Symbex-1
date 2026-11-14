"""Multiclass FOMO targets from hand-made VOC annotations, without downloading VOC."""
import copy
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOMO_DIR = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(FOMO_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fomo():
    return load("fomo_classes", "fomo_classes.py")


def obj(name, x, y, **extra):
    return {"name": name, "bndbox": {"xmin": str(x), "xmax": str(x), "ymin": str(y), "ymax": str(y)}, **extra}


def test_constants(fomo):
    assert fomo.VOC_CLASSES == ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                                'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor')
    assert len(set(fomo.VOC_CLASSES)) == 20
    assert (fomo.SIZE, fomo.CELL, fomo.GRID) == (96, 8, 12)


def test_person_compatibility_with_convert(fomo, tmp_path, monkeypatch):
    from PIL import Image
    data = load("fomo_data_compat", "fomo_data.py")
    objects = [obj('person', 10, 20), obj('person', 12, 21), obj('person', 50, 30, difficult='1'),
               obj('cat', 100, 40), obj('person', 140, 70)]
    image = Image.new('RGB', (192, 96))
    dataset = [(image, {'annotation': {'object': objects}})]
    monkeypatch.setattr(data, 'HERE', str(tmp_path))
    data.convert([dataset], 'compat')
    grid, centers, ids = fomo.grid_from_objects(objects, 192, 96, ['person'])
    with np.load(tmp_path / 'data' / 'fomo_compat.npz') as saved:
        np.testing.assert_array_equal(grid, saved['y'][0])
        np.testing.assert_array_equal(centers, saved['centers'])
    np.testing.assert_array_equal(ids, [1, 1, 1])


def test_class_order_and_geometry(fomo):
    objects = [obj('person', 12, 28), obj('cat', 60, 20)]
    grid, centers, ids = fomo.grid_from_objects(objects, 96, 96, ['person', 'cat'])
    reverse, centers2, ids2 = fomo.grid_from_objects(objects, 96, 96, ['cat', 'person'])
    assert (grid[3, 1], grid[2, 7]) == (1, 2)
    assert (reverse[3, 1], reverse[2, 7]) == (2, 1)
    np.testing.assert_array_equal(centers, [[12, 28], [60, 20]])
    np.testing.assert_array_equal(centers2, centers)
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_array_equal(ids2, [2, 1])


def test_collision_nearest_and_ties(fomo):
    far, near, third = obj('person', 0, 0), obj('cat', 4, 4), obj('person', 1, 1)
    for objects in ([far, near, third], [near, far, third]):
        grid, centers, ids = fomo.grid_from_objects(objects, 96, 96, ['person', 'cat'])
        assert grid[0, 0] == 2
        np.testing.assert_array_equal(centers, [[float(o['bndbox']['xmin']), float(o['bndbox']['ymin'])]
                                                for o in objects])
        np.testing.assert_array_equal(ids, [2 if o['name'] == 'cat' else 1 for o in objects])
    left, right = obj('person', 2, 4), obj('cat', 6, 4)
    assert fomo.grid_from_objects([left, right], 96, 96, ['person', 'cat'])[0][0, 0] == 1
    assert fomo.grid_from_objects([right, left], 96, 96, ['person', 'cat'])[0][0, 0] == 2


def test_scaling_and_boundaries(fomo):
    objects = [obj('person', 16, 4), obj('cat', 192, 48), obj('cat', 220, 60)]
    grid, centers, ids = fomo.grid_from_objects(objects, 192, 48, ['person', 'cat'])
    np.testing.assert_array_equal(centers, [[8, 8], [96, 96], [110, 120]])
    np.testing.assert_array_equal(ids, [1, 2, 2])
    assert grid[1, 1] == 1
    assert grid[11, 11] == 2


def assert_empty(result):
    grid, centers, ids = result
    assert grid.shape == (12, 12) and grid.dtype == np.uint8 and not grid.any()
    assert centers.shape == (0, 2) and centers.dtype == np.float32
    assert ids.shape == (0,) and ids.dtype == np.int32


def test_empty_filtering_and_single_dict(fomo):
    person = obj('person', 4, 4)
    for a, b in zip(fomo.grid_from_objects(person, 96, 96, ['person']),
                    fomo.grid_from_objects([person], 96, 96, ['person'])):
        np.testing.assert_array_equal(a, b)
    for objects, classes in [([], ['person']), ([{'name': 'cat'}], ['person']),
                             ([{'name': 'person', 'difficult': '1'}], ['person']), ([person], [])]:
        assert_empty(fomo.grid_from_objects(objects, 96, 96, classes))
    objects = [person, obj('person', 12, 4, difficult='0'), {'name': 'person', 'difficult': '1'}, {'name': 'cat'}]
    grid, centers, ids = fomo.grid_from_objects(objects, 96, 96, ['person'])
    assert grid.sum() == 2
    np.testing.assert_array_equal(centers, [[4, 4], [12, 4]])
    np.testing.assert_array_equal(ids, [1, 1])
    assert grid.dtype == np.uint8 and centers.dtype == np.float32 and ids.dtype == np.int32


def test_validation_and_independent_results(fomo):
    with pytest.raises(ValueError, match='unknown'):
        fomo.grid_from_objects([], 96, 96, ['unknown'])
    with pytest.raises(ValueError, match='person'):
        fomo.grid_from_objects([], 96, 96, ['person', 'person'])
    objects, classes = [obj('person', 4, 4)], ['person']
    original = copy.deepcopy((objects, classes))
    first = fomo.grid_from_objects(objects, 96, 96, classes)
    second = fomo.grid_from_objects(objects, 96, 96, classes)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
    assert (objects, classes) == original
    for bad, error in [({'name': 'person'}, KeyError), (obj('person', 'bad', 4), ValueError)]:
        before = copy.deepcopy(bad)
        with pytest.raises(error):
            fomo.grid_from_objects([bad], 96, 96, classes)
        assert bad == before and classes == original[1]
