"""
Bit-exact simulator of the ternary dense engine (lib_symbex symbex_ternary.c), and its check against a plain
numpy integer reference.

Two independent computations of the same folded integers (export/ternary.plegar):
    simular_bits()      what the C does, on PACKED 32-bit words: sign plane, nonzero-mask plane, +-1 input bits,
                        dot = popcount(mask & ~(a ^ sign)) - popcount(mask & (a ^ sign)); first layer through
                        0x00 / 0xFF masks on the uint8 inputs; thresholds packed LSB-first like symbex_dense_umbral
    referencia_numpy()  integer matrix product with the {-1, 0, +1} codes (no bits at all)
verificar_bit_exacto() runs both and compares every layer's raw scores and the final logits with ==.
"""
import numpy as np

from ..export.dense import Q_ACT, Q_MULT
from ..export.ternary import planos

_LOTE = 16                                              # samples per step of the packed simulation (memory)


def _popcount32(v):
    """SWAR popcount of uint32 values (the generic HAL's portable path)."""
    v = v.astype(np.uint32)
    v = v - ((v >> np.uint32(1)) & np.uint32(0x55555555))
    v = (v & np.uint32(0x33333333)) + ((v >> np.uint32(2)) & np.uint32(0x33333333))
    v = (v + (v >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return ((v * np.uint32(0x01010101)) >> np.uint32(24)).astype(np.int64)


def _empaquetar_bits(bits01, palabras):
    """(N, n) 0/1 -> (N, palabras) uint32, bit i of word i / 32 = column i, padding bits 0."""
    N, n = bits01.shape
    relleno = np.zeros((N, palabras * 32), dtype=np.uint32)
    relleno[:, :n] = bits01
    desplazado = relleno.reshape(N, palabras, 32) << np.arange(32, dtype=np.uint32)
    return np.bitwise_or.reduce(desplazado, axis=2)


def _desempaquetar(palabras_fila, n):
    """(out, palabras) uint32 -> (out, n) 0/1 bits (to build the 0x00 / 0xFF masks of the uint8 layer)."""
    out, pal = palabras_fila.shape
    bits = (palabras_fila[:, :, None] >> np.arange(32, dtype=np.uint32)) & np.uint32(1)
    return bits.reshape(out, pal * 32)[:, :n]


def _capa_u8(x_u8, s, m, n_in):
    """symbex_ternary_entrada_u8_mascara: sum of x where (mask & sign) - sum of x where (mask & ~sign)."""
    mas = _desempaquetar(m & s, n_in).astype(np.int64)
    menos = _desempaquetar(m & ~s, n_in).astype(np.int64)
    mascara_mas = -mas                                  # 0 or -1 (all ones), like -(int32_t)bit in the C
    mascara_menos = -menos
    x = x_u8.astype(np.int64)
    raw = np.empty((len(x), s.shape[0]), dtype=np.int64)
    for k in range(0, len(x), _LOTE):
        xb = x[k:k + _LOTE, None, :]
        raw[k:k + _LOTE] = (xb & mascara_mas[None]).sum(2) - (xb & mascara_menos[None]).sum(2)
    return raw


def _capa_bits(a, s, m):
    """symbex_ternary_forward on packed input bits a (N, palabras)."""
    raw = np.empty((len(a), s.shape[0]), dtype=np.int64)
    for k in range(0, len(a), _LOTE):
        d = a[k:k + _LOTE, None, :] ^ s[None]
        raw[k:k + _LOTE] = _popcount32(m[None] & ~d).sum(2) - _popcount32(m[None] & d).sum(2)
    return raw


def _logits(raw, d):
    sh = Q_MULT - Q_ACT
    return ((raw * d["escala"][None, :] + (1 << (sh - 1))) >> sh) + d["shift"][None, :]


def simular_bits(capas, x_u8):
    """Packed-word simulation of the C engine. Returns dict(raws=[per layer], logits, pred, bits=[hidden +-1])."""
    raws, bits = [], []
    a = None
    for d in capas:
        out, n_in = d["codigos"].shape
        s, m, pal = planos(d["codigos"])
        s, m = s.astype(np.uint32), m.astype(np.uint32)
        raw = _capa_u8(x_u8, s, m, n_in) if d["entrada_u8"] else _capa_bits(a, s, m)
        raws.append(raw)
        if d["ultima"]:
            logits = _logits(raw, d)
            return dict(raws=raws, logits=logits, pred=logits.argmax(1), bits=bits)
        uno = (raw >= d["umbral"][None, :]).astype(np.uint32)
        bits.append(np.where(uno == 1, 1, -1))
        a = _empaquetar_bits(uno, (out + 31) // 32)


def referencia_numpy(capas, x_u8):
    """Integer reference: raw = act @ codes.T, act = +-1 from the thresholds. Same return shape as simular_bits."""
    act = x_u8.astype(np.int64)
    raws, bits = [], []
    for d in capas:
        raw = act @ d["codigos"].T
        raws.append(raw)
        if d["ultima"]:
            logits = _logits(raw, d)
            return dict(raws=raws, logits=logits, pred=logits.argmax(1), bits=bits)
        act = np.where(raw >= d["umbral"][None, :], 1, -1)
        bits.append(act)


def verificar_bit_exacto(capas, x_u8):
    """Runs both and compares every layer's raw scores, the logits and the predictions exactly.
    Returns dict(iguales, capas, raw_iguales=[per layer], logits_iguales, _pred, _bits)."""
    sim, ref = simular_bits(capas, x_u8), referencia_numpy(capas, x_u8)
    raw_iguales = [bool(np.array_equal(a, b)) for a, b in zip(sim["raws"], ref["raws"])]
    logits_iguales = bool(np.array_equal(sim["logits"], ref["logits"]))
    return dict(iguales=all(raw_iguales) and logits_iguales, capas=len(capas), raw_iguales=raw_iguales,
                logits_iguales=logits_iguales, _pred=sim["pred"], _bits=sim["bits"])
