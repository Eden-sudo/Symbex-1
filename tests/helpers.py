"""Helpers shared by the tests (bit unpacking, ctypes structs). Fixtures live in conftest.py."""
import ctypes

import numpy as np


class SymbexLayerData(ctypes.Structure):
    """Mirror of struct SymbexLayerData (core/symbex_types.h)."""
    _fields_ = [("in_features", ctypes.c_uint16), ("out_features", ctypes.c_uint16),
                ("k_bits", ctypes.c_uint16), ("weights", ctypes.POINTER(ctypes.c_uint32))]


def unpack_lsb_words(words, n):
    """(rows, n_words) 32-bit words, bit i of word i // 32 = element i -> ((rows, n) 0/1, padding bits)."""
    words = np.asarray(words, dtype=np.uint64)
    bits = ((words[:, :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).reshape(words.shape[0], -1)
    return bits[:, :n].astype(np.int64), bits[:, n:].astype(np.int64)


def pack_lsb_words(bits01, n_words=None):
    """(rows, n) 0/1 -> (rows, n_words) uint32, LSB-first (the engine's layout). Test-side packer."""
    rows, n = bits01.shape
    n_words = (n + 31) // 32 if n_words is None else n_words
    full = np.zeros((rows, n_words * 32), dtype=np.uint64)
    full[:, :n] = bits01
    return (full.reshape(rows, n_words, 32) << np.arange(32, dtype=np.uint64)).sum(2).astype(np.uint32)


def popcount(words):
    """Total number of set bits in an array of 32-bit words."""
    w = np.asarray(words, dtype=np.uint64)
    return int(((w[..., None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).sum())


def as_ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))
