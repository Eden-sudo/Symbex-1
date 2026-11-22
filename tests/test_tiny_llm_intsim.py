"""CPU contracts for tiny_llm integer arithmetic and packed weights."""
import importlib.util
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parent.parent
TINY = REPO / "benchmarks" / "experiments" / "tiny_llm"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def intsim():
    train = _load_module("tiny_llm_train_intsim_test", TINY / "train.py")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "train", train)
        return _load_module("tiny_llm_intsim_test", TINY / "intsim.py")


def _tensor(values):
    return torch.tensor(values, dtype=torch.int64, device="cpu")


def _assert_result(result, expected, shape):
    assert result.shape == shape
    assert result.dtype == torch.int64
    assert result.device.type == "cpu"
    assert result.reshape(-1).tolist() == expected


def test_isqrt_matches_math_isqrt(intsim):
    roots = [1, 2, 3, 255, 65535, 2**26 + 1, 2**31 - 1, 2**31]
    values = [0, 1, 2, 3, 5, 8]
    values += [n for root in roots for n in (root * root - 1, root * root)]
    values += [2**62 + delta for delta in (-1024, -2, -1, 0, 1, 2, 1024)]
    rng = random.Random(1234)
    values += [rng.randrange(2**62) for _ in range(512)]
    for value in values[:20]:
        _assert_result(intsim.isqrt(_tensor(value)), [math.isqrt(value)], torch.Size([]))
    for tensor in (_tensor(values), _tensor(values[:24]).reshape(4, 6), _tensor([])):
        expected = [math.isqrt(n) for n in tensor.reshape(-1).tolist()]
        _assert_result(intsim.isqrt(tensor), expected, tensor.shape)


def _rounded(n, d):
    q, rem = divmod(abs(n), d)
    q += 2 * rem >= d
    return -q if n < 0 else q


def test_rdiv_rounds_half_away_from_zero(intsim):
    ties = [-5, -3, -1, 0, 1, 3, 5]
    _assert_result(intsim.rdiv(_tensor(ties), 2), [-3, -2, -1, 0, 1, 2, 3], torch.Size([7]))
    rng = random.Random(4321)
    nums = list(range(-10, 11)) + [-(2**40), 2**40]
    nums += [rng.randint(-(2**40), 2**40) for _ in range(512)]
    dens = [rng.randint(1, 2**20) for _ in nums]
    _assert_result(intsim.rdiv(_tensor(nums), 7), [_rounded(n, 7) for n in nums], torch.Size([len(nums)]))
    _assert_result(intsim.rdiv(_tensor(nums), _tensor(dens)),
                   [_rounded(n, d) for n, d in zip(nums, dens)], torch.Size([len(nums)]))
    nums = list(range(-10, 11))
    dens = [1, 2, 3, 4, 7]
    result = intsim.rdiv(_tensor(nums).reshape(-1, 1), _tensor(dens).reshape(1, -1))
    _assert_result(result, [_rounded(n, d) for n in nums for d in dens], torch.Size([len(nums), len(dens)]))


@pytest.mark.parametrize("bits", [0, 1, 7, 31, 62])
def test_check_accepts_bounds_and_returns_same_tensor(intsim, bits):
    bound = 2**bits - 1
    for tensor in (_tensor([0, -bound, bound]), _tensor([])):
        assert intsim.check(tensor, bits) is tensor
    tensor = _tensor([0, -(2**62 - 1), 2**62 - 1])
    assert intsim.check(tensor) is tensor


@pytest.mark.parametrize("bits", [0, 1, 7, 31, 62])
def test_check_rejects_limit(intsim, bits):
    for magnitude in (2**bits, 2**bits + 1):
        for value in (-magnitude, magnitude):
            with pytest.raises(AssertionError) as exc:
                intsim.check(_tensor([[0, value], [0, 0]]), bits, what="weights")
            assert str(exc.value) == f"weights: {magnitude} does not fit {bits} bits"
    with pytest.raises(AssertionError) as exc:
        intsim.check(_tensor([2**62]))
    assert str(exc.value) == f": {2**62} does not fit 62 bits"


# Mirrors linear() in c/tinylm.c, one bit per input column.
def _unpack_bits(row, width):
    return [1 if (int(row[i // 32]) >> (i % 32)) & 1 else -1 for i in range(width)]


# Mirrors emb2() in c/tinylm.h, including signed two-bit extension.
def _unpack_crumbs(row, width):
    result = []
    for i in range(width):
        byte = int(row[(i // 64) * 16 + i % 16])
        value = (byte >> (2 * ((i // 16) % 4))) & 3
        result.append(2 * (value - 4 if value >= 2 else value) + 1)
    return result


# Mirrors emb4() in c/tinylm.h, including signed nibble extension.
def _unpack_nibbles(row, width):
    result = []
    for i in range(width):
        byte = int(row[(i // 32) * 16 + i % 16])
        value = (byte >> (4 * ((i // 16) % 2))) & 15
        result.append(value - 16 if value >= 8 else value)
    return result


def test_pack_bits_layout(intsim):
    row = [-1] * 33
    for i in (0, 7, 8, 31, 32):
        row[i] = 1
    packed = intsim.pack_bits(_tensor([row, [-1] * 33, [1] * 33]))
    assert packed.shape == (3, 2)
    assert packed.dtype == np.dtype('<u4')
    assert packed[0].tolist() == [0x80000181, 1]
    assert packed[0].tobytes() == bytes.fromhex('81 01 00 80 01 00 00 00')
    assert packed[1].tolist() == [0, 0]
    assert packed[2].tolist() == [0xffffffff, 1]
    assert intsim.pack_bits(_tensor([[0]]))[0, 0] == 0


@pytest.mark.parametrize("width", [1, 7, 8, 31, 32, 33, 63, 64, 65, 97])
def test_pack_bits_round_trip(intsim, width):
    rng = random.Random(width)
    rows = [[rng.choice((-1, 1)) for _ in range(width)] for _ in range(4)]
    packed = intsim.pack_bits(_tensor(rows))
    assert packed.shape == (4, (width + 31) // 32)
    assert packed.dtype == np.dtype('<u4')
    assert [_unpack_bits(row, width) for row in packed] == rows
    if width % 32:
        assert all(int(row[-1]) >> (width % 32) == 0 for row in packed)


def test_pack_crumbs_layout(intsim):
    rows = [[-3] * 16 + [-1] * 16 + [1] * 16 + [3] * 16,
            [3] * 16 + [1] * 16 + [-1] * 16 + [-3] * 16]
    packed = intsim.pack_crumbs(_tensor(rows))
    assert packed.shape == (2, 16)
    assert packed.dtype == np.uint8
    assert packed[0].tolist() == [0x4e] * 16
    assert [_unpack_crumbs(row, 64) for row in packed] == rows


@pytest.mark.parametrize("width", [64, 128, 192])
def test_pack_crumbs_round_trip(intsim, width):
    rng = random.Random(width)
    rows = [[rng.choice((-3, -1, 1, 3)) for _ in range(width)] for _ in range(4)]
    rows[0][:4] = [-3, -1, 1, 3]
    packed = intsim.pack_crumbs(_tensor(rows))
    assert packed.shape == (4, width // 4)
    assert packed.dtype == np.uint8
    assert [_unpack_crumbs(row, width) for row in packed] == rows


def test_pack_nibbles_layout(intsim):
    rows = [[-7] * 16 + [7] * 16, [7] * 16 + [-7] * 16]
    packed = intsim.pack_nibbles(_tensor(rows))
    assert packed.shape == (2, 16)
    assert packed.dtype == np.uint8
    assert packed[0].tolist() == [0x79] * 16
    assert packed[1].tolist() == [0x97] * 16
    assert [_unpack_nibbles(row, 32) for row in packed] == rows


@pytest.mark.parametrize("width", [32, 64, 96])
def test_pack_nibbles_round_trip(intsim, width):
    rng = random.Random(width)
    rows = [[rng.randint(-7, 7) for _ in range(width)] for _ in range(4)]
    rows[0][:15] = list(range(-7, 8))
    packed = intsim.pack_nibbles(_tensor(rows))
    assert packed.shape == (4, width // 2)
    assert packed.dtype == np.uint8
    assert [_unpack_nibbles(row, width) for row in packed] == rows


@pytest.mark.parametrize("name,width", [("pack_crumbs", n) for n in (1, 63, 65)] +
                         [("pack_nibbles", n) for n in (1, 31, 33)])
def test_grouped_packers_reject_unaligned_widths(intsim, name, width):
    with pytest.raises(ValueError):
        getattr(intsim, name)(_tensor([[1] * width]))
