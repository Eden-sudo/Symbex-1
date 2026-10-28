"""The detection score of fomo_voc_person (benchmarks/conv2d/fomo_voc_person/model.py): detections() and f1_score().

Both columns of the benchmark (binary and INT8) are scored by these two functions, so they are checked here on small
hand-made grids, without the VOC data:
    detections(grid)       8-connected groups of positive cells -> one center per group, in pixels
    f1_score(...)          greedy matching per image within MATCH_DISTANCE (1.5 cells = 12 px, the edge included)
The matching is greedy (each detection, in row order, takes the nearest free person), not optimal: one test documents
that on purpose.
"""
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOMO_DIR = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person")


@pytest.fixture(scope="module")
def fomo():
    """the benchmark's model.py, loaded by path under its own name (other benchmarks have a model.py too)"""
    spec = importlib.util.spec_from_file_location("fomo_model", os.path.join(FOMO_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid(*cells, value=1):
    """a 12 x 12 grid with `value` at every (row, col)"""
    g = np.zeros((12, 12), dtype=np.uint8)
    for r, c in cells:
        g[r, c] = value
    return g


def truth(*points, image=0):
    """centers (float32, as fomo_data.py saves them) and their owner image (int32), all in one image"""
    return np.array(points, dtype=np.float32).reshape(-1, 2), np.full(len(points), image, dtype=np.int32)


# ---------------------------------------------------------------- detections()

def test_constants_match_the_data_layout(fomo):
    assert (fomo.CELL, fomo.GRID, fomo.SIZE) == (8, 12, 96)
    assert fomo.MATCH_DISTANCE == pytest.approx(12.0)


def test_empty_grid_has_no_detections(fomo):
    assert fomo.detections(grid()) == []


def test_one_cell_is_one_detection_at_the_cell_center(fomo):
    assert fomo.detections(grid((2, 5))) == [pytest.approx((44.0, 20.0))]


def test_horizontal_neighbours_are_one_detection_at_their_mean(fomo):
    assert fomo.detections(grid((3, 3), (3, 4))) == [pytest.approx((32.0, 28.0))]


def test_diagonal_neighbours_are_one_detection_8_connectivity(fomo):
    assert fomo.detections(grid((0, 0), (1, 1))) == [pytest.approx((8.0, 8.0))]


def test_l_shape_is_one_detection_at_the_mean_of_its_cells(fomo):
    (x, y), = fomo.detections(grid((0, 0), (1, 0), (1, 1)))
    assert (x, y) == pytest.approx(((1 / 3 + 0.5) * 8, (2 / 3 + 0.5) * 8))


def test_separate_groups_are_separate_detections_in_row_order(fomo):
    assert fomo.detections(grid((5, 5), (0, 0))) == [pytest.approx((4.0, 4.0)), pytest.approx((44.0, 44.0))]


def test_cells_one_apart_are_not_connected(fomo):
    assert len(fomo.detections(grid((0, 0), (0, 2)))) == 2


def test_corner_cell_stays_inside_the_image(fomo):
    assert fomo.detections(grid((11, 11))) == [pytest.approx((92.0, 92.0))]


@pytest.mark.parametrize("value", [2, 255])
def test_any_positive_value_counts_as_a_person_cell(fomo, value):
    assert fomo.detections(grid((2, 5), value=value)) == [pytest.approx((44.0, 20.0))]


# ---------------------------------------------------------------- f1_score()

def test_perfect_prediction_scores_100(fomo):
    s = fomo.f1_score([grid((2, 5))], *truth((44.0, 20.0)))
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)
    assert (s["precision"], s["recall"], s["f1"]) == pytest.approx((100.0, 100.0, 100.0))


def test_no_detections_with_persons_is_recall_0_without_dividing_by_zero(fomo):
    s = fomo.f1_score([grid()], *truth((44.0, 20.0), (80.0, 80.0)))
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 2)
    assert (s["precision"], s["recall"], s["f1"]) == (0.0, 0.0, 0.0)


def test_nothing_at_all_scores_0_without_error(fomo):
    s = fomo.f1_score([grid(), grid()], np.zeros((0, 2), np.float32), np.zeros(0, np.int32))
    assert s == dict(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=0)


def test_detection_in_an_image_without_persons_is_a_false_positive(fomo):
    s = fomo.f1_score([grid((2, 5))], np.zeros((0, 2), np.float32), np.zeros(0, np.int32))
    assert (s["tp"], s["fp"], s["fn"], s["precision"]) == (0, 1, 0, 0.0)


def test_person_exactly_at_the_match_distance_counts(fomo):
    s = fomo.f1_score([grid((2, 5))], *truth((44.0 + 12.0, 20.0)))
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)


def test_person_just_past_the_match_distance_is_a_miss_and_a_false_positive(fomo):
    s = fomo.f1_score([grid((2, 5))], *truth((44.0 + 12.01, 20.0)))
    assert (s["tp"], s["fp"], s["fn"]) == (0, 1, 1)


def test_one_person_is_never_matched_twice(fomo):
    s = fomo.f1_score([grid((2, 4), (2, 6))], *truth((44.0, 20.0)))
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 0)


def test_detection_takes_the_nearest_person(fomo):
    # the detection at (44, 20): one person 4 px away, one 9 px away; it must take the 4 px one, the other is missed
    s = fomo.f1_score([grid((2, 5))], *truth((35.0, 20.0), (48.0, 20.0)))
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 1)


def test_matching_is_greedy_in_row_order_not_optimal(fomo):
    # detections A (4, 4) and B (20, 4); persons P (12, 4) and Q (4, 15). An optimal matching pairs A-Q (11 px) and
    # B-P (8 px): 2 hits. The greedy one lets A, first in row order, take its nearest person P (8 px); B is then
    # 19.4 px from Q and matches nothing: 1 hit, 1 false positive, 1 miss. This documents the score as it is.
    s = fomo.f1_score([grid((0, 0), (0, 2))], *truth((12.0, 4.0), (4.0, 15.0)))
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)


def test_persons_only_match_detections_of_their_own_image(fomo):
    # image 1 holds the person; images 0 and 2 have a detection at the same place: both are false positives
    centers, owner = truth((44.0, 20.0), image=1)
    s = fomo.f1_score([grid((2, 5)), grid((2, 5)), grid((2, 5))], centers, owner)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 0)


def test_precision_recall_and_f1_formula(fomo):
    # image 0: two hits and one false positive; image 1: one missed person -> tp 2, fp 1, fn 1
    centers = np.array([(4.0, 4.0), (44.0, 44.0), (80.0, 80.0)], dtype=np.float32)
    owner = np.array([0, 0, 1], dtype=np.int32)
    s = fomo.f1_score([grid((0, 0), (5, 5), (11, 0)), grid()], centers, owner)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    assert (s["precision"], s["recall"], s["f1"]) == pytest.approx((200 / 3, 200 / 3, 200 / 3))


# --- the validation split: split_indices() and held_out_centers() ---

def test_split_is_a_partition(fomo):
    tr, va = fomo.split_indices(100, 10)
    assert len(va) == 10 and len(tr) == 90
    assert set(tr).isdisjoint(va)
    assert sorted(np.concatenate([tr, va])) == list(range(100))


def test_split_is_fixed_whatever_the_training_seed(fomo):
    import torch
    torch.manual_seed(1)
    a = fomo.split_indices(16551)
    torch.manual_seed(2)
    b = fomo.split_indices(16551)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_split_is_not_the_last_images(fomo):
    """the case being fixed: the calibration set used to be the last 1,000 images (all of them from VOC 2012)"""
    _, va = fomo.split_indices(16551)
    assert len(va) == 1000
    assert sorted(va) != list(range(15551, 16551))
    assert (va < 5011).any()              # some come from VOC 2007 trainval (the first 5,011 images)


def test_held_out_centers_reindexes_to_val(fomo):
    centers = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    owner = np.array([0, 2, 4, 4])                       # persons in images 0, 2 and 4 (two in image 4)
    c, o = fomo.held_out_centers(centers, owner, np.array([4, 2]))
    assert c.tolist() == [[20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]
    assert o.tolist() == [1, 0, 0]                        # image 2 is val[1], image 4 is val[0]


def test_held_out_centers_image_without_persons(fomo):
    """a validation image without persons (and beyond the last owner) breaks nothing"""
    c, o = fomo.held_out_centers(np.array([[10.0, 10.0]]), np.array([0]), np.array([3, 0]))
    assert c.tolist() == [[10.0, 10.0]] and o.tolist() == [1]


def test_held_out_centers_score_perfectly_on_their_own_grids(fomo):
    """grids drawn from the returned centers, in val order, give F1 100 with f1_score"""
    centers = np.array([[12.0, 20.0], [60.0, 44.0], [84.0, 76.0]])
    owner = np.array([1, 3, 4])
    val = np.array([4, 1, 3])
    c, o = fomo.held_out_centers(centers, owner, val)
    grids = np.zeros((len(val), fomo.GRID, fomo.GRID), np.uint8)
    for (x, y), i in zip(c, o):
        grids[i, int(y // fomo.CELL), int(x // fomo.CELL)] = 1
    assert fomo.f1_score(grids, c, o)["f1"] == pytest.approx(100.0)
