"""
vertical.py -- the vertical engine's sizes and limits, as lib_symbex computes them (symbex_bool_vertical.c): the
exporter of method "boolean" writes its buffers into the header with these, and validate_c checks them against the
library's own numbers.

    vertical_problem(convs, levels, in_size)        None if the engine runs these layers, else why not
    vertical_work_bytes(...)                        bytes of its work block
    vertical_prepared_bytes(convs, compact=False)   bytes of its prepared lists and thresholds
An int8 stem (a conv with int8=True, first in convs) runs as in the standard engine, its output turned into planes;
a pooled int8 classifier (pool_planes > 0) runs on the final map in HWC, its count planes after it.
"""
import numpy as np


def _vplane_words(ch, h, w):
    """words of a channel-plane map of the vertical engine (symbex_bool_vertical.h: plane stride + 4 zero words)"""
    return ch * (((h * w // 32 + 3) & ~3) + 4) + 4


def _hwc_words(ch, h, w):
    """words of an HWC map in the vertical engine's hwc buffer, rounded up to 4 (hwc_words in the C)"""
    return (h * w * ((ch + 31) // 32) + 3) & ~3


def vertical_problem(convs, levels, in_size):
    """None if the vertical engine runs these layers, else why not (symbex_bool_vertical_check, the same rules)"""
    if convs and convs[0].get("int8"):             # the int8 stem: only its output must suit the planes
        oh = convs[0]["oh"]
        if 32 % oh or oh * oh % 32:
            return "int8 stem output size (width must divide 32, h * w a multiple of 32)"
        convs = convs[1:]
    else:
        if 3 * levels > 32:
            return "thermometer with more than 32 bits per pixel"
        if 32 % in_size or in_size * in_size % 32:
            return "input size (width must divide 32, h * w a multiple of 32)"
    for c in convs:
        k, s, h, oh, K = c["k"], c["s"], c["h"], c["oh"], c["k"] * c["k"] * c["cin"]
        if k not in (1, 3):
            return "kernel other than 1 x 1 or 3 x 3"
        if (s == 1 and oh != h) or (s == 2 and (k != 3 or h != 2 * oh)) or s not in (1, 2):
            return f"stride {s} with kernel {k} ({h} -> {oh})"
        if 32 % h or 32 % oh or h * h % 32 or oh * oh % 32:
            return f"map size {h} -> {oh} (width must divide 32, h * w a multiple of 32)"
        if oh * oh > 256 * 128:
            return f"output map {oh} x {oh} above 32768 pixels (more than 256 tiles of 128)"
        if K % 8 or K >= 2048:
            return f"k * k * in_ch = {K}: not a multiple of 8 or not below 2048"
    return None


def vertical_work_bytes(convs, blocks, levels, in_size, fc_rw, pool_planes=0):
    """bytes of the vertical engine's work block: symbex_bool_vertical_work_bytes (net_sizes + carve), the same sums.
    pool_planes: the pooled int8 classifier's count planes (n_planes * its row words), 0 without it"""
    sz = dict(map=_vplane_words(3 * levels, in_size, in_size), phases=0, xq=0, ap=0, km=0, bt=0, ba=0, list=0, tiles=0)
    hwc = 0
    if convs and convs[0].get("int8"):             # its output planes, its HWC output and the 32-byte window
        st = convs[0]
        sz["map"] = _vplane_words(st["cout"], st["oh"], st["oh"])
        hwc = _hwc_words(st["cout"], st["oh"], st["oh"]) + 8
    for c in (convs[1:] if convs and convs[0].get("int8") else convs):
        nwo = c["oh"] * c["oh"] // 32
        TL, K = (nwo + 3) // 4, c["k"] * c["k"] * c["cin"]
        K8 = (K + 8) // 8 * 8
        for key, v in (("map", _vplane_words(c["cout"], c["oh"], c["oh"])),
                       ("phases", 4 * _vplane_words(c["cin"], c["oh"], c["oh"]) if c["s"] == 2 else 0),
                       ("xq", TL * K8), ("ap", TL * 12), ("km", TL * 9), ("bt", TL * 12),
                       ("ba", ((K8 // 8 + 7) & ~7) + 8), ("list", K + 18), ("tiles", TL)):
            sz[key] = max(sz[key], v)
    for ci, si, pool in blocks:
        if si is not None and pool:
            c = convs[ci]
            sz["map"] = max(sz["map"], _vplane_words(c["cin"], c["h"] // 2, c["h"] // 2))
    last = convs[blocks[-1][0]] if blocks else convs[0]
    hwc = max(hwc, last["oh"] * last["oh"] * ((last["cout"] + 31) // 32) + fc_rw + 4)
    if pool_planes:
        hwc = max(hwc, _hwc_words(last["cout"], last["oh"], last["oh"]) + pool_planes)
    r16 = lambda b: (b + 15) & ~15
    shared = max(r16(sz["km"] * 16) + r16(sz["bt"] * 16), r16(sz["phases"] * 4))    # the phases share km and bt
    return (4 * r16(sz["map"] * 4) + shared + r16(sz["xq"] * 16) + r16(sz["ap"] * 16) + 2 * r16(sz["ba"] * 16) + r16(16 * 16) + r16(15 * 16)
            + r16(sz["list"] * 2 + 4) + 3 * r16(sz["tiles"]) + r16(sz["tiles"] * 9) + r16(256 * 20) + r16(hwc * 4))


def _vertical_tile_types(c, with_kinds=False):
    """tile types of a conv in the vertical engine: border_kinds in symbex_bool_vertical.c (the tiles of 128 output
    pixels with the same border-kind masks), the same steps; with_kinds: (tile types, border kinds)"""
    k, s, H, OH = c["k"], c["s"], c["h"], c["oh"]
    pad, OW, W = k // 2, OH, H
    nwo = OH * OW // 32
    TL = (nwo + 3) // 4

    def valid(o, n):
        return sum(1 << i for i in range(k) if 0 <= o * s - pad + i < n)

    def kinds(n_out, n_in):
        vals, cnt = [], []
        for o in range(n_out):
            m = valid(o, n_in)
            if m not in vals:
                vals.append(m)
                cnt.append(0)
            cnt[vals.index(m)] += 1
        return vals, cnt
    cv, cc = kinds(OW, W)
    rv, rc = kinds(OH, H)
    cmask = [0] * len(cv)
    for ox in range(OW):
        for j in range(ox, 32, OW):
            cmask[cv.index(valid(ox, W))] |= 1 << j
    nc, NK = len(cv), len(rv) * len(cv)
    kbase = cc.index(max(cc)) + rc.index(max(rc)) * nc       # the first commonest, as the C
    rows_per_word, rowbits = 32 // OW, (1 << OW) - 1
    km = [[[0] * 4 for _ in range(NK)] for _ in range(TL)]
    for wd in range(4 * TL):
        t, j = divmod(wd, 4)
        if wd >= nwo:
            km[t][kbase][j] = 0xFFFFFFFF
            continue
        rbits = [0] * len(rv)
        for r in range(rows_per_word):
            rbits[rv.index(valid(wd * rows_per_word + r, H))] |= rowbits << (r * OW)
        for a in range(len(rv)):
            for b in range(nc):
                km[t][a * nc + b][j] = rbits[a] & cmask[b]
    ntt = len({tuple(tuple(m) for m in tile) for tile in km})
    return (ntt, NK) if with_kinds else ntt


def vertical_prepared_bytes(convs, compact=False):
    """bytes of the vertical engine's prepared lists and thresholds: symbex_bool_vertical_prepared_bytes (compact:
    symbex_bool_vertical_prepared_compact_bytes), the same sums (convs in the network's order: stem, then every
    block's conv and shortcut)"""
    r16 = lambda b: (b + 15) & ~15
    total = r16(12 * len(convs))
    for c in convs:
        if c.get("int8"):                          # the int8 stem has no lists (its directory entry stays 0)
            continue
        ntt, nk = _vertical_tile_types(c, with_kinds=True)
        if compact:                          # header, kind masks per tile type, their kinds, the values per channel
            total += 16 + ntt * nk * 16 + r16(ntt * 10) + r16(c["cout"] * nk * 2)
        else:
            total += c["cout"] * ntt * 12 * 16
        ones = [sum(bin(int(w)).count("1") for w in row) for row in np.asarray(c["rows"]).reshape(c["cout"], -1)]
        total += r16(sum(2 * (2 + (n + 7) // 8 * 8) for n in ones))
        total += r16(1 + (c["oh"] * c["oh"] // 32 + 3) // 4)     # NTT and the type of every tile
    return total
