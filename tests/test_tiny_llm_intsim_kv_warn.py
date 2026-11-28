"""intsim.kv_mismatch: every cache setting that differs from training is listed (the warning before a wrong loss)."""
import importlib.util
from pathlib import Path
import sys

import pytest


TINY = Path(__file__).resolve().parent.parent / "benchmarks" / "experiments" / "tiny_llm"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def intsim():
    train = _load_module("tiny_llm_train_kv_warn_test", TINY / "train.py")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "train", train)
        return _load_module("tiny_llm_intsim_kv_warn_test", TINY / "intsim.py")


K1TRV2 = {"k_bits": 1, "v_bits": 2, "kv_recent": 32}     # train.py args of the step 9-11 binary model


def test_matching_cache_is_silent(intsim):
    assert intsim.kv_mismatch(K1TRV2, 1, 2, 1.0, 32) == []
    assert intsim.kv_mismatch({}, 8, 8, 1.0, 0) == []    # older runs: no cache keys = int8, factor 1, no window


def test_defaults_on_a_low_bit_model_list_every_difference(intsim):
    # the intsim_wrong_kv8.json case: the model trained with K1 / V2 / window 32, simulated with intsim's defaults
    assert intsim.kv_mismatch(K1TRV2, 8, 8, 1.0, 0) == [
        "--k-bits 8 (trained 1)", "--v-bits 8 (trained 2)", "--kv-recent 0 (trained 32)"]


def test_single_differences(intsim):
    assert intsim.kv_mismatch({"k_scale": 1.3}, 8, 8, 1.0, 0) == ["--k-scale 1 (trained 1.3)"]
    assert intsim.kv_mismatch({}, 2, 8, 1.0, 0) == ["--k-bits 2 (trained 8)"]   # a no-training cache study
    assert intsim.kv_mismatch(K1TRV2, 1, 2, 1.0, 16) == ["--kv-recent 16 (trained 32)"]
