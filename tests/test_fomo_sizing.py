"""The sizing study of fomo_voc_person (benchmarks/conv2d/fomo_voc_person/fomo_sizing.py): mark_pareto().

The variants are chosen on the validation F1 (f1_val); the test F1 (f1) is reported only. These hand-made rows check
that the Pareto front follows f1_val and never f1, without the VOC data or the runs/ folders.
"""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOMO_DIR = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person")


@pytest.fixture(scope="module")
def sizing():
    """the benchmark's fomo_sizing.py, loaded by path"""
    spec = importlib.util.spec_from_file_location("fomo_sizing", os.path.join(FOMO_DIR, "fomo_sizing.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(spec, f1_val, f1, s3_ms):
    return dict(spec=spec, f1_val=f1_val, f1=f1, s3_ms=s3_ms)


def front(rows):
    return {r["spec"] for r in rows if r["pareto"]}


def test_worse_on_validation_better_on_test_is_not_chosen(sizing):
    """b is slower and better on test, but worse on validation: a beats it on what decides"""
    rows = sizing.mark_pareto([row("a", 30.0, 28.0, 100.0), row("b", 29.0, 35.0, 120.0)])
    assert front(rows) == {"a"}


def test_the_front_ignores_the_test_f1(sizing):
    """the same rows with the test F1 swapped give the same front"""
    a = sizing.mark_pareto([row("a", 30.0, 28.0, 100.0), row("b", 29.0, 35.0, 120.0)])
    b = sizing.mark_pareto([row("a", 30.0, 35.0, 100.0), row("b", 29.0, 28.0, 120.0)])
    assert front(a) == front(b) == {"a"}


def test_slower_but_better_on_validation_stays(sizing):
    rows = sizing.mark_pareto([row("fast", 28.0, 30.0, 60.0), row("slow", 31.0, 29.0, 160.0)])
    assert front(rows) == {"fast", "slow"}


def test_tie_on_validation_keeps_both(sizing):
    """same f1_val, different latency: neither is strictly better on validation, so both stay"""
    rows = sizing.mark_pareto([row("a", 30.0, 30.0, 100.0), row("b", 30.0, 30.0, 120.0)])
    assert front(rows) == {"a", "b"}


def test_single_row_is_on_the_front(sizing):
    assert front(sizing.mark_pareto([row("only", 25.0, 25.0, 50.0)])) == {"only"}
