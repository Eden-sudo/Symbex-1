"""Check the INT8 reference multiplier against the TFLite Q31 convention."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
CARRY_MULTIPLIERS = [math.nextafter(2.0**e, 0.0) for e in (-30, -1, 0, 4)]
MULTIPLIERS = [2.0**e for e in range(-31, 5)] + [0.1, 0.3, 0.75, 1.5, 3.7] + CARRY_MULTIPLIERS
X = np.array([-1000000, -65537, -257, -3, -2, -1, 0, 1, 2, 3, 257, 65537, 1000000], dtype=np.int64)


@pytest.fixture(scope="module")
def int8_reference():
    path = REPO / "tools" / "compiler_core" / "export" / "int8_reference.py"
    spec = importlib.util.spec_from_file_location("int8_reference_quantize_test", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"INT8 reference requires missing dependency: {exc.name}") from exc
    return mod


def _tflite_quantize_multiplier(M):
    if M == 0:
        return 0, 0
    q, shift = math.frexp(M)
    qf = int(round(q * 2**31))
    if qf == 2**31:
        qf //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return qf, shift


def test_quantize_multiplier_underflow(int8_reference):
    assert int8_reference.quantize_multiplier(2.0**-40) == (0, 0)


@pytest.mark.parametrize("M", MULTIPLIERS)
def test_quantize_multiplier_matches_tflite(int8_reference, M):
    q, shift = int8_reference.quantize_multiplier(M)
    assert (q, shift) == _tflite_quantize_multiplier(M)
    assert abs(q * 2.0**(shift - 31) - M) <= M * 2.0**-31


@pytest.mark.parametrize("M", CARRY_MULTIPLIERS)
def test_multiplier_exercises_carry(M):
    q, _ = math.frexp(M)
    assert round(q * 2**31) == 2**31


@pytest.mark.parametrize("M, expected", [
    (0.0, (0, 0)),
    (2.0**-33, (0, 0)),
    (2.0**-32, (2**30, -31)),
    (math.nextafter(2.0**-32, 0.0), (2**30, -31)),
])
def test_quantize_multiplier_boundary(int8_reference, M, expected):
    assert int8_reference.quantize_multiplier(M) == expected


@pytest.mark.parametrize("M", MULTIPLIERS)
def test_quantized_multiply(int8_reference, M):
    actual = int8_reference.quantized_multiply(X, *int8_reference.quantize_multiplier(M))
    y = X.astype(np.float64) * M
    expected = np.copysign(np.floor(np.abs(y) + 0.5), y).astype(np.int64)
    assert np.all(np.abs(actual - expected) <= 1)


def test_quantized_multiply_underflow(int8_reference):
    actual = int8_reference.quantized_multiply(X, *int8_reference.quantize_multiplier(2.0**-40))
    assert actual.shape == X.shape
    assert np.all(actual == 0)
