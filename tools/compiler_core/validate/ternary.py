"""
Bit-exact simulator of the ternary dense engine (lib_symbex symbex_ternary.c), and its check against a plain
numpy integer reference.

Two independent computations of the same folded integers (export/ternary.fold):
    simulate_bits()     what the C does, on PACKED 32-bit words: sign plane, nonzero-mask plane, +-1 input bits,
                        dot = popcount(mask & ~(a ^ sign)) - popcount(mask & (a ^ sign)); first layer through
                        0x00 / 0xFF masks on the uint8 inputs; thresholds packed LSB-first like symbex_dense_threshold
    numpy_reference()   integer matrix product with the {-1, 0, +1} codes (no bits at all)
verify_bit_exact() runs both and compares every layer's raw scores and the final logits with ==.
"""
import numpy as np

from ..export.dense import Q_ACT, Q_MULT
from ..export.ternary import planes

_BATCH = 16                                             # samples per step of the packed simulation (memory)


def _popcount32(v):
    """SWAR popcount of uint32 values (the generic HAL's portable path)."""
    v = v.astype(np.uint32)
    v = v - ((v >> np.uint32(1)) & np.uint32(0x55555555))
    v = (v & np.uint32(0x33333333)) + ((v >> np.uint32(2)) & np.uint32(0x33333333))
    v = (v + (v >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return ((v * np.uint32(0x01010101)) >> np.uint32(24)).astype(np.int64)


def _pack_bits(bits01, words):
    """(N, n) 0/1 -> (N, words) uint32, bit i of word i / 32 = column i, padding bits 0."""
    N, n = bits01.shape
    padded = np.zeros((N, words * 32), dtype=np.uint32)
    padded[:, :n] = bits01
    shifted = padded.reshape(N, words, 32) << np.arange(32, dtype=np.uint32)
    return np.bitwise_or.reduce(shifted, axis=2)


def _unpack(row_words, n):
    """(out, words) uint32 -> (out, n) 0/1 bits (to build the 0x00 / 0xFF masks of the uint8 layer)."""
    out, words = row_words.shape
    bits = (row_words[:, :, None] >> np.arange(32, dtype=np.uint32)) & np.uint32(1)
    return bits.reshape(out, words * 32)[:, :n]


def _layer_u8(x_u8, s, m, n_in):
    """symbex_ternary_input_u8_mask: sum of x where (mask & sign) - sum of x where (mask & ~sign)."""
    plus = _unpack(m & s, n_in).astype(np.int64)
    minus = _unpack(m & ~s, n_in).astype(np.int64)
    mask_plus = -plus                                   # 0 or -1 (all ones), like -(int32_t)bit in the C
    mask_minus = -minus
    x = x_u8.astype(np.int64)
    raw = np.empty((len(x), s.shape[0]), dtype=np.int64)
    for k in range(0, len(x), _BATCH):
        xb = x[k:k + _BATCH, None, :]
        raw[k:k + _BATCH] = (xb & mask_plus[None]).sum(2) - (xb & mask_minus[None]).sum(2)
    return raw


def _layer_bits(a, s, m):
    """symbex_ternary_forward on packed input bits a (N, words)."""
    raw = np.empty((len(a), s.shape[0]), dtype=np.int64)
    for k in range(0, len(a), _BATCH):
        d = a[k:k + _BATCH, None, :] ^ s[None]
        raw[k:k + _BATCH] = _popcount32(m[None] & ~d).sum(2) - _popcount32(m[None] & d).sum(2)
    return raw


def _logits(raw, d):
    sh = Q_MULT - Q_ACT
    return ((raw * d["scale"][None, :] + (1 << (sh - 1))) >> sh) + d["shift"][None, :]


def simulate_bits(layers, x_u8):
    """Packed-word simulation of the C engine. Returns dict(raws=[per layer], logits, pred, bits=[hidden +-1])."""
    raws, bits = [], []
    a = None
    for d in layers:
        out, n_in = d["codes"].shape
        s, m, words = planes(d["codes"])
        s, m = s.astype(np.uint32), m.astype(np.uint32)
        raw = _layer_u8(x_u8, s, m, n_in) if d["u8_input"] else _layer_bits(a, s, m)
        raws.append(raw)
        if d["last"]:
            logits = _logits(raw, d)
            return dict(raws=raws, logits=logits, pred=logits.argmax(1), bits=bits)
        one = (raw >= d["threshold"][None, :]).astype(np.uint32)
        bits.append(np.where(one == 1, 1, -1))
        a = _pack_bits(one, (out + 31) // 32)


def numpy_reference(layers, x_u8):
    """Integer reference: raw = act @ codes.T, act = +-1 from the thresholds. Same return shape as simulate_bits."""
    act = x_u8.astype(np.int64)
    raws, bits = [], []
    for d in layers:
        raw = act @ d["codes"].T
        raws.append(raw)
        if d["last"]:
            logits = _logits(raw, d)
            return dict(raws=raws, logits=logits, pred=logits.argmax(1), bits=bits)
        act = np.where(raw >= d["threshold"][None, :], 1, -1)
        bits.append(act)


def verify_bit_exact(layers, x_u8):
    """Runs both and compares every layer's raw scores, the logits and the predictions exactly.
    Returns dict(equal, layers, raw_equal=[per layer], logits_equal, _pred, _bits)."""
    sim, ref = simulate_bits(layers, x_u8), numpy_reference(layers, x_u8)
    raw_equal = [bool(np.array_equal(a, b)) for a, b in zip(sim["raws"], ref["raws"])]
    logits_equal = bool(np.array_equal(sim["logits"], ref["logits"]))
    return dict(equal=all(raw_equal) and logits_equal, layers=len(layers), raw_equal=raw_equal,
                logits_equal=logits_equal, _pred=sim["pred"], _bits=sim["bits"])
