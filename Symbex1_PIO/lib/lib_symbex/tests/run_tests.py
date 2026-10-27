"""
run_tests.py -- tests of lib_symbex on the PC: every piece is compiled with gcc and compared against an exact
reference written here in numpy, on thousands of random cases (sizes that are not multiples of 32, garbage in
the padding bits, thresholds at the extremes...).

    cd lib_symbex && python tests/run_tests.py
    python tests/run_tests.py --cases 200          # fewer cases, faster
    python tests/run_tests.py --only dense         # only the tests whose name matches "dense" ("hal|ops" for two)
    python tests/run_tests.py --stress             # stress test (see below)

--stress: AddressSanitizer + UndefinedBehaviorSanitizer (any read or write outside a buffer, any shift by 32
or more, any signed overflow stops the test), sizes up to 4x larger, every boundary size first (1, 31, 32, 33,
127, 128, 129, ...), extreme data (all zeros, all ones, alternating, inputs at 255) mixed with random data, and
2000 cases per test by default.

Every test runs three times: with hal/generic, with hal/esp32s3 in its C mode (SYMBEX_HAL_TEST_NO_PIE: the same
code as on the chip, with the PIE instructions replaced by C), and with the 8-bit code paths the library uses on
AVR (SYMBEX_U8_BYTEWISE, SYMBEX_POPCOUNT_FORCE_LUT8, SYMBEX_PLANES_PLANE_OUTER, SYMBEX_PLANES_PER_NEURON). The PIE
instructions and AVR itself
can only be checked on the
chip (chip_test.py, and the checksums of chip_test.py --kernels).

Covered:
    hal        popcount32, xnor_popcount, and_popcount, u8_prepare + u8_sum_bits, the whole-layer *_rows forms
               (xnor, and, u8 with padded and unpadded rows, and_planes), hal_name
    ops        symbex_and (raw, input_u8), symbex_bitops, symbex_bitpack, symbex_mac_1bit (the DSP fixed point)
    dense      forward, argmax, input_u8 + planes + mask (SWAR), threshold, forward_bits, input_u8_bits,
               input_u8_planes_bits (k = 1..8), binary_classifier_argmax; k_bits = 1
    networks   symbex_network_dense (chains of dense layers), symbex_network_conv (residual blocks with and without
               downsample, and the validation of add_block)
    conv       bpr (rgb_to_24bits), binarize_image, binary_pool_2x2, extract_window (HWC maps: pixel (y, x) starts at
               word (y * W + x) * PW, PW = ceil(C / 32), channel c is bit c % 32 of word c / 32)
    bool       fully binary networks (layers/conv/symbex_bool): conv with integer thresholds, thermometer, OR pooling,
               classifier, and whole random networks (symbex_bool_net_run) against a numpy reference
Not covered yet: gated (until the V2 methods are evaluated), dense with k_bits > 1 (known to be pending).
Requires: gcc, numpy.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

LIB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SOURCES = ["src/ops/symbex_and.c", "src/ops/symbex_bitops.c", "src/ops/symbex_bitpack.c", "src/ops/symbex_mac_1bit.c",
           "src/layers/dense/symbex_dense.c", "src/layers/dense/symbex_gated.c",
           "src/layers/conv/symbex_bpr.c", "src/layers/conv/symbex_conv2d.c", "src/layers/conv/symbex_binary_pool.c",
           "src/core/network/symbex_network_dense.c", "src/core/network/symbex_network_conv.c",
           "src/layers/conv/symbex_bool.c", "src/layers/conv/symbex_bool1d.c"]
HALS = {"generic": (["src/hal/generic/symbex_hal_impl.c"], []),
        "esp32s3 (C mode)": (["src/hal/esp32s3/symbex_hal_impl.c"],
                             ["-DCONFIG_IDF_TARGET_ESP32S3", "-DSYMBEX_HAL_TEST_NO_PIE"]),
        "generic, native count forced": (["src/hal/generic/symbex_hal_impl.c"], ["-DSYMBEX_POPCOUNT_FORCE_NATIVE"]),
        "generic, 8-bit paths (as on AVR)": (["src/hal/generic/symbex_hal_impl.c"],
                                             ["-DSYMBEX_U8_BYTEWISE", "-DSYMBEX_POPCOUNT_FORCE_LUT8", "-DSYMBEX_PLANES_PLANE_OUTER",
                                              "-DSYMBEX_PLANES_PER_NEURON"])}
RNG = np.random.default_rng(1234)
STRESS = False                             # set by --stress
SANITIZE = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-g"]
EDGES = [1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 255, 256, 257, 511, 512, 513]


def pick(lo, hi, i):
    """A size in [lo, hi): in stress mode, every boundary size first (case i), then random; the range is 4x wider."""
    if STRESS:
        hi = lo + (hi - lo) * 4
        edges = [e for e in EDGES if lo <= e < hi]
        if i < len(edges):
            return edges[i]
    return int(RNG.integers(lo, hi))


def bits_data(shape, i):
    """Random 0/1 data; in stress mode some cases are all zeros, all ones or alternating."""
    if STRESS and i % 7 in (1, 2, 3):
        flat = np.zeros(int(np.prod(shape)), dtype=np.int64)
        if i % 7 == 2:
            flat[:] = 1
        elif i % 7 == 3:
            flat[::2] = 1
        return flat.reshape(shape)
    return RNG.integers(0, 2, shape)


def u8_data(n, i):
    """Random bytes; in stress mode some cases are all 255 or all 0 (largest and smallest sums)."""
    if STRESS and i % 5 == 1:
        return np.full(n, 255, dtype=np.int64)
    if STRESS and i % 5 == 2:
        return np.zeros(n, dtype=np.int64)
    return RNG.integers(0, 256, n)
TMP = tempfile.mkdtemp(prefix="symbex_tests_")
_cache = {}


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------
def build(name, main_c, hal):
    """Compiles main_c with the library and the given HAL; returns the path of the program."""
    key = (name, hal)
    if key in _cache:
        return _cache[key]
    hal_src, flags = HALS[hal]
    src = os.path.join(TMP, f"{name}.c")
    exe = os.path.join(TMP, f"{name}_{abs(hash(hal)) % 10000}")
    open(src, "w").write('#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <string.h>\n'
                         '#include "symbex.h"\n' + main_c)
    stubs = os.path.join(TMP, "stubs")
    os.makedirs(stubs, exist_ok=True)
    open(os.path.join(stubs, "esp_heap_caps.h"), "w").write(
        "#pragma once\n#include <stdlib.h>\n#define MALLOC_CAP_INTERNAL 1\n#define MALLOC_CAP_8BIT 2\n"
        "static inline void* heap_caps_malloc(size_t n, int c) { (void)c; return malloc(n); }\n")
    cmd = ["gcc", "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror=implicit-function-declaration", *(SANITIZE if STRESS else []),
           "-I", os.path.join(LIB, "include"), "-I", stubs, *flags, "-o", exe, src,
           *[os.path.join(LIB, s) for s in SOURCES + hal_src]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile error ({name}, {hal}):\n{r.stderr[-1500:]}")
    _cache[key] = exe
    return exe


class Chunks(list):
    """Input data built piece by piece: `data += piece` appends (joined once in run()). Adding to a bytes object
    instead copies everything each time, which made the large stress tests take very long."""
    def __iadd__(self, piece):
        self.append(piece)
        return self


def run(exe, data):
    """Runs the program with `data` (bytes or Chunks) on stdin; returns the integers it printed."""
    if isinstance(data, list):
        data = b"".join(data)
    r = subprocess.run([exe], input=data, capture_output=True)
    if r.returncode != 0 or (STRESS and b"runtime error" in r.stderr):
        err = r.stderr.decode(errors="replace")
        line = next((l for l in err.splitlines() if "runtime error" in l or "ERROR: AddressSanitizer" in l), err[-300:])
        raise RuntimeError(f"{os.path.basename(exe)}: {line.strip()}\n{err[-1500:]}")
    return np.array([int(v) for v in r.stdout.split()], dtype=np.int64)


def words(bits, n_words):
    """bit matrix (..., n) -> uint32 words (..., n_words), bit i of word i // 32 = bit i (numpy packbits: fast)."""
    bits = np.asarray(bits, dtype=np.uint8) & 1
    pad = n_words * 32 - bits.shape[-1]
    if pad > 0:
        bits = np.concatenate([bits, np.zeros(bits.shape[:-1] + (pad,), dtype=np.uint8)], axis=-1)
    packed = np.packbits(bits[..., :n_words * 32], axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u4").reshape(bits.shape[:-1] + (n_words,))


def popcount(x):
    x = np.asarray(x, dtype=np.uint64)
    return np.array([bin(int(v)).count("1") for v in x.ravel()], dtype=np.int64).reshape(x.shape)


def i32(*vals):
    return np.array(vals, dtype=np.int32).tobytes()


def hwc_words(bits_hwc, pw):
    """(h, w, c) bits -> (h * w * pw,) uint32 HWC words."""
    h, w, c = bits_hwc.shape
    padded = np.zeros((h, w, pw * 32), dtype=np.int64)
    padded[:, :, :c] = bits_hwc
    return words(padded.reshape(h * w, pw * 32), pw).ravel()


# ---------------------------------------------------------------------------------------------------
# tests: each returns (cases, failures, first failure description)
# ---------------------------------------------------------------------------------------------------
def test_hal_popcount32(hal, n):
    exe = build("popcount32", """
int main(void) { uint32_t x; while (fread(&x, 4, 1, stdin) == 1) printf("%d\\n", symbex_hal_popcount32(x)); return 0; }""", hal)
    x = RNG.integers(0, 2 ** 32, n * 20, dtype=np.uint64).astype(np.uint32)
    x[:4] = [0, 0xFFFFFFFF, 1, 0x80000000]
    got, exp = run(exe, x.tobytes()), popcount(x)
    bad = np.nonzero(got != exp)[0]
    return len(x), len(bad), (f"x=0x{x[bad[0]]:08X}: got {got[bad[0]]}, expected {exp[bad[0]]}" if len(bad) else "")


def test_hal_name(hal, n):
    """symbex_hal_name: a non-empty, printable description of the counting path (for the logs and the paper)."""
    exe = build("halname", """
int main(void) { const char* s = symbex_hal_name(); size_t n = s ? strlen(s) : 0; int ok = n > 0 && n < 120;
  for (size_t i = 0; ok && i < n; i++) ok = s[i] >= 32 && s[i] < 127;
  printf("%d %d\\n", ok, (int)n); return 0; }""", hal)
    got = run(exe, b"")
    return 1, int(got[0] != 1), ("" if got[0] == 1 else f"empty, too long or not printable ({got[1]} chars)")


def test_hal_xnor_popcount(hal, n):
    exe = build("xnor", """
int main(void) { int nw, nb; static uint32_t a[256], b[256];
  while (fread(&nw, 4, 1, stdin) == 1 && fread(&nb, 4, 1, stdin) == 1 && fread(a, 4, nw, stdin) == (size_t)nw
         && fread(b, 4, nw, stdin) == (size_t)nw) printf("%d\\n", symbex_xnor_popcount(a, b, nw, nb));
  return 0; }""", hal)
    data, exp = Chunks(), []
    for _ in range(n):
        nw = pick(1, 41, _)
        nb = int(RNG.integers(1, nw * 32 + 1)) if not (STRESS and _ % 3 == 0) else nw * 32
        a = RNG.integers(0, 2 ** 32, nw, dtype=np.uint64).astype(np.uint32)
        b = RNG.integers(0, 2 ** 32, nw, dtype=np.uint64).astype(np.uint32)
        abits = (a[:, None].astype(np.uint64) >> np.arange(32, dtype=np.uint64) & 1).ravel()[:nb]
        bbits = (b[:, None].astype(np.uint64) >> np.arange(32, dtype=np.uint64) & 1).ravel()[:nb]
        exp.append(nb - 2 * int((abits != bbits).sum()))
        data += i32(nw, nb) + a.tobytes() + b.tobytes()
    got = run(exe, data)
    bad = np.nonzero(got != np.array(exp))[0]
    return n, len(bad), (f"case {bad[0]}: got {got[bad[0]]}, expected {exp[bad[0]]}" if len(bad) else "")


def test_hal_and_popcount(hal, n):
    exe = build("andpc", """
int main(void) { int nw; static uint32_t a[256] __attribute__((aligned(16))), b[256] __attribute__((aligned(16)));
  while (fread(&nw, 4, 1, stdin) == 1 && fread(a, 4, nw, stdin) == (size_t)nw && fread(b, 4, nw, stdin) == (size_t)nw)
    printf("%d\\n", symbex_and_popcount(a, b, nw));
  return 0; }""", hal)
    data, exp = Chunks(), []
    for _ in range(n):
        nw = pick(1, 41, _)
        a = RNG.integers(0, 2 ** 32, nw, dtype=np.uint64).astype(np.uint32)
        b = RNG.integers(0, 2 ** 32, nw, dtype=np.uint64).astype(np.uint32)
        exp.append(int(popcount(a & b).sum()))
        data += i32(nw) + a.tobytes() + b.tobytes()
    got = run(exe, data)
    bad = np.nonzero(got != np.array(exp))[0]
    return n, len(bad), (f"case {bad[0]}: got {got[bad[0]]}, expected {exp[bad[0]]}" if len(bad) else "")


def test_hal_u8_sum_bits(hal, n):
    exe = build("u8sum", """
int main(void) { int nw, ni; static uint8_t x[8192]; static uint8_t ws[8192] __attribute__((aligned(16)));
  static uint32_t w[256] __attribute__((aligned(16)));
  while (fread(&nw, 4, 1, stdin) == 1 && fread(&ni, 4, 1, stdin) == 1 && fread(x, 1, ni, stdin) == (size_t)ni
         && fread(w, 4, nw, stdin) == (size_t)nw) {
    symbex_u8_prepare(x, ni, ws, nw); printf("%d\\n", symbex_u8_sum_bits(ws, w, nw)); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for _ in range(n):
        nw = pick(1, 41, _)
        ni = int(RNG.integers(1, nw * 32 + 1)) if not (STRESS and _ % 3 == 0) else nw * 32
        x = u8_data(ni, _).astype(np.uint8)
        wbits = RNG.integers(0, 2, nw * 32)
        wbits[ni:] = 0                                       # weight padding is 0 (the exporter guarantees it)
        exp.append(int((x.astype(np.int64) * wbits[:ni]).sum()))
        data += i32(nw, ni) + x.tobytes() + words(wbits, nw).tobytes()
    got = run(exe, data)
    bad = np.nonzero(got != np.array(exp))[0]
    return n, len(bad), (f"case {bad[0]}: got {got[bad[0]]}, expected {exp[bad[0]]}" if len(bad) else "")


def test_hal_rows(hal, n):
    """The whole-layer HAL primitives (*_rows) against numpy: XNOR rows, AND rows, uint8 rows (padded stride and
    unpadded stride: the PIE layout and the C fallback), AND over k bit planes; garbage in the input past n_in."""
    exe = build("halrows", """
int main(void) { int a6[6]; static uint32_t a[64] __attribute__((aligned(16))), w[64 * 96] __attribute__((aligned(16)));
  static uint32_t wx[64 * 96] __attribute__((aligned(16))), planes[8 * 64];
  static uint8_t x[2048], work[2048] __attribute__((aligned(16))); static int32_t out[96];
  while (fread(a6, 4, 6, stdin) == 6) {
    int n_in = a6[0], n_out = a6[1], words = a6[2], stride = a6[3], k = a6[4], low = a6[5];
    if (fread(a, 4, stride, stdin) != (size_t)stride || fread(w, 4, n_out * stride, stdin) != (size_t)(n_out * stride)
        || fread(x, 1, n_in, stdin) != (size_t)n_in || fread(planes, 4, k * words, stdin) != (size_t)(k * words)) return 1;
    for (int r = 0; r < n_out; r++) memcpy(&wx[r * words], &w[r * stride], 4 * words);
    symbex_xnor_popcount_rows(a, wx, words, n_in, n_out, out); for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    symbex_and_popcount_rows(a, w, stride, n_out, out); for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    symbex_u8_prepare(x, n_in, work, stride); symbex_u8_sum_bits_rows(work, n_in, w, stride, n_out, out);
    for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    symbex_u8_prepare(x, n_in, work, words); symbex_u8_sum_bits_rows(work, n_in, wx, words, n_out, out);
    for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    symbex_and_planes_rows(planes, k, low, wx, words, n_out, out); for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c in range(n):
        n_in, n_out = min(pick(1, 1300, c), 2000), int(RNG.integers(1, 97))
        nwords = (n_in + 31) // 32
        stride = -(-nwords // 4) * 4
        k = int(RNG.integers(1, 9))
        low = int(RNG.integers(0, 9 - k))
        abits, wbits = bits_data(n_in, c), bits_data((n_out, n_in), c + 3)
        x = u8_data(n_in, c)
        pbits = RNG.integers(0, 2, (k, n_in))
        a = RNG.integers(0, 2 ** 32, stride, dtype=np.uint64).astype(np.uint32)      # garbage past n_in
        a[:nwords] = words(abits, nwords)
        if n_in % 32:
            a[nwords - 1] |= np.uint32((int(RNG.integers(0, 2 ** 32)) >> (n_in % 32)) << (n_in % 32))
        w = words(wbits, stride)
        sw = 2 * wbits - 1
        planes_ref = sum((wbits @ pbits[q]) << (low + q) for q in range(k))
        exp.append(np.concatenate([sw @ (2 * abits - 1), wbits @ abits, wbits @ x, wbits @ x, planes_ref]))
        data += i32(n_in, n_out, nwords, stride, k, low) + a.tobytes() + w.astype(np.uint32).tobytes() \
            + x.astype(np.uint8).tobytes() + words(pbits, nwords).astype(np.uint32).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_ops_and(hal, n):
    exe = build("andops", """
int main(void) { int n_in, n_out, stride;
  static uint32_t a[256] __attribute__((aligned(16))); static uint32_t w[256 * 160] __attribute__((aligned(16)));
  static int32_t cw[160], raw[160]; static uint8_t x[8192]; static uint8_t work[8192] __attribute__((aligned(16)));
  while (fread(&n_in, 4, 1, stdin) == 1 && fread(&n_out, 4, 1, stdin) == 1 && fread(&stride, 4, 1, stdin) == 1
         && fread(a, 4, stride, stdin) == (size_t)stride && fread(w, 4, n_out * stride, stdin) == (size_t)(n_out * stride)
         && fread(cw, 4, n_out, stdin) == (size_t)n_out && fread(x, 1, n_in, stdin) == (size_t)n_in) {
    symbex_and_raw(a, n_in, w, stride, n_out, cw, raw);
    for (int j = 0; j < n_out; j++) printf("%d ", raw[j]);
    symbex_and_input_u8(x, n_in, w, stride, n_out, work, raw);
    for (int j = 0; j < n_out; j++) printf("%d ", raw[j]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp, total = Chunks(), [], 0
    for c in range(n):
        n_in = pick(1, 1300, c // 2)
        n_out = pick(1, 40, c // 2)
        stride = -(-((n_in + 31) // 32) // 4) * 4 if c % 2 == 0 else (n_in + 31) // 32 + int(RNG.integers(0, 3))
        stride = min(stride, 256)
        if (n_in + 31) // 32 > stride:
            continue
        abits = bits_data(n_in, c)
        wbits = bits_data((n_out, n_in), c + 3)
        x = u8_data(n_in, c)
        a = words(np.concatenate([abits, np.zeros(stride * 32 - n_in, dtype=np.int64)]), stride)
        a[(n_in + 31) // 32:] = RNG.integers(0, 2 ** 32, stride - (n_in + 31) // 32, dtype=np.uint64).astype(np.uint32)
        if n_in % 32:                                        # garbage above n_in in the last real word
            a[n_in // 32] |= np.uint32((int(RNG.integers(0, 2 ** 32)) >> (n_in % 32)) << (n_in % 32))
        w = words(np.concatenate([wbits, np.zeros((n_out, stride * 32 - n_in), dtype=np.int64)], axis=1), stride)
        data += i32(n_in, n_out, stride) + a.tobytes() + w.tobytes() + wbits.sum(1).astype(np.int32).tobytes() \
            + x.astype(np.uint8).tobytes()
        exp.append(np.concatenate([(2 * abits - 1) @ (2 * wbits - 1).T, x @ (2 * wbits - 1).T]))
        total += 1
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return total, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_ops_bitops(hal, n):
    exe = build("bitops", """
int main(void) { uint32_t plane[4], orig[4]; uint32_t start, value; uint8_t nb;
  while (fread(orig, 4, 4, stdin) == 4 && fread(&start, 4, 1, stdin) == 1 && fread(&nb, 1, 1, stdin) == 1
         && fread(&value, 4, 1, stdin) == 1) {
    memcpy(plane, orig, sizeof plane);
    uint32_t e = symbex_extract_bits(orig, start, nb);
    symbex_insert_bits(plane, start, nb, value);
    printf("%u %u %u %u %u %d\\n", e, plane[0], plane[1], plane[2], plane[3], symbex_popcount_and(orig, plane, 4)); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for _ in range(n * 5):
        orig = RNG.integers(0, 2 ** 32, 4, dtype=np.uint64).astype(np.uint32)
        nb = int(RNG.integers(1, 33))
        start = int(RNG.integers(0, 128 - nb + 1))
        value = int(RNG.integers(0, 2 ** 32))                # bits above nb set on purpose
        bits = (orig[:, None].astype(np.uint64) >> np.arange(32, dtype=np.uint64) & 1).ravel().astype(np.int64)
        e = sum(int(bits[start + i]) << i for i in range(nb))
        nbits = bits.copy()
        for i in range(nb):
            nbits[start + i] = (value >> i) & 1
        plane = words(nbits, 4)
        exp.append([e, *[int(v) for v in plane], int(popcount(orig & plane).sum())])
        data += orig.tobytes() + i32(start) + bytes([nb]) + np.array([value], dtype=np.uint32).tobytes()
    got = run(exe, data).reshape(-1, 6)
    exp = np.array(exp, dtype=np.int64)
    bad = np.nonzero((got != exp).any(1))[0]
    what = ""
    if len(bad):
        col = int(np.nonzero(got[bad[0]] != exp[bad[0]])[0][0])
        what = (f"case {bad[0]}: " + ["extract_bits", "insert_bits word 0", "insert_bits word 1", "insert_bits word 2",
                                     "insert_bits word 3", "popcount_and"][col] + f": got {got[bad[0], col]}, expected {exp[bad[0], col]}")
    return len(exp), len(bad), what


def test_ops_load_line_buffer(hal, n):
    exe = build("linebuf", """
int main(void) { int ic, ih, iw, k, s, p, oh; static uint32_t src[16384], dst[16384];
  while (fread(&ic, 4, 1, stdin) == 1 && fread(&ih, 4, 1, stdin) == 1 && fread(&iw, 4, 1, stdin) == 1 &&
         fread(&k, 4, 1, stdin) == 1 && fread(&s, 4, 1, stdin) == 1 && fread(&p, 4, 1, stdin) == 1 && fread(&oh, 4, 1, stdin) == 1) {
    int wpr = (iw + 31) / 32;
    if (fread(src, 4, ic * ih * wpr, stdin) != (size_t)(ic * ih * wpr)) return 1;
    memset(dst, 0xAB, sizeof dst);
    symbex_load_line_buffer(src, dst, ic, ih, iw, k, s, p, oh);
    for (int i = 0; i < ic * k * wpr; i++) printf("%u ", dst[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp, total = Chunks(), [], 0
    for _ in range(n):
        ic, ih, iw = int(RNG.integers(1, 5)), int(RNG.integers(1, 12)), pick(1, 70, _)
        k = int(RNG.choice([1, 3, 5]))
        s, p = int(RNG.integers(1, 3)), int(RNG.integers(0, k // 2 + 1))
        oh_max = (ih + 2 * p - k) // s
        if oh_max < 0:
            continue
        oh = int(RNG.integers(0, oh_max + 1))
        wpr = (iw + 31) // 32
        src = RNG.integers(0, 2 ** 32, ic * ih * wpr, dtype=np.uint64).astype(np.uint32)
        out = []
        for c in range(ic):
            for kh in range(k):
                r = oh * s - p + kh
                out.extend(src[(c * ih + r) * wpr:(c * ih + r + 1) * wpr] if 0 <= r < ih else [0] * wpr)
        exp.append(np.array(out, dtype=np.int64))
        data += i32(ic, ih, iw, k, s, p, oh) + src.tobytes()
        total += 1
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return total, len(bad), (f"word {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_ops_bitpack(hal, n):
    exe = build("bitpack", """
int main(void) { int nn; static int32_t s[16384]; static uint32_t out[600];
  while (fread(&nn, 4, 1, stdin) == 1 && fread(s, 4, nn, stdin) == (size_t)nn) {
    memset(out, 0xFF, sizeof out);
    symbex_bitpack_from_scores(s, out, (uint16_t)nn);
    for (int i = 0; i < nn; i++) printf("%u ", (out[i / 32] >> (i % 32)) & 1u);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for _ in range(n):
        nn_ = pick(1, 3000, _)
        s = RNG.integers(-5, 6, nn_).astype(np.int32)        # many exact zeros: score >= 0 -> bit 1
        exp.append((s >= 0).astype(np.int64))
        data += i32(nn_) + s.tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"bit {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_dense_k1(hal, n):
    exe = build("dense", """
int main(void) { int n_in, n_out; static uint32_t inp[256], w[256 * 160]; static int32_t out[160], th[160];
  static uint8_t x[8192]; static uint32_t planes[8 * 256], bits[5];
  while (fread(&n_in, 4, 1, stdin) == 1 && fread(&n_out, 4, 1, stdin) == 1) {
    int pal = (n_in + 31) / 32;
    static int32_t th2[160];
    if (fread(inp, 4, pal, stdin) != (size_t)pal || fread(w, 4, n_out * pal, stdin) != (size_t)(n_out * pal) ||
        fread(x, 1, n_in, stdin) != (size_t)n_in || fread(th, 4, n_out, stdin) != (size_t)n_out ||
        fread(th2, 4, n_out, stdin) != (size_t)n_out) return 1;
    struct SymbexLayerData L = { (uint16_t)n_in, (uint16_t)n_out, 1, w };
    symbex_dense_forward(inp, out, &L); for (int j = 0; j < n_out; j++) printf("%d ", out[j]);
    printf("%d ", symbex_dense_argmax(inp, &L));
    symbex_dense_input_u8(x, out, &L); for (int j = 0; j < n_out; j++) printf("%d ", out[j]);
    symbex_dense_input_u8_planes(x, planes, out, &L); for (int j = 0; j < n_out; j++) printf("%d ", out[j]);
    symbex_dense_input_u8_mask(x, out, &L); for (int j = 0; j < n_out; j++) printf("%d ", out[j]);
    symbex_dense_forward(inp, out, &L);
    memset(bits, 0xFF, sizeof bits);
    symbex_dense_threshold(out, th, bits, (uint16_t)n_out);
    for (int k = 0; k < (n_out + 31) / 32; k++) printf("%u ", bits[k]);
    memset(bits, 0xFF, sizeof bits);
    symbex_dense_forward_bits(inp, th, bits, &L);
    for (int k = 0; k < (n_out + 31) / 32; k++) printf("%u ", bits[k]);
    memset(bits, 0xFF, sizeof bits);
    symbex_dense_input_u8_bits(x, th2, bits, &L);
    for (int k = 0; k < (n_out + 31) / 32; k++) printf("%u ", bits[k]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp, total = Chunks(), [], 0
    for _ in range(n):
        n_in, n_out = pick(1, 1300, _), pick(1, 41, _)
        pal = (n_in + 31) // 32
        abits = bits_data(n_in, _)
        wbits = bits_data((n_out, n_in), _ + 3)
        x = u8_data(n_in, _)
        inp = words(np.concatenate([abits, np.zeros(pal * 32 - n_in, dtype=np.int64)]), pal)
        if n_in % 32:                                        # garbage in the padding bits of the input
            inp[-1] |= np.uint32((int(RNG.integers(0, 2 ** 32)) >> (n_in % 32)) << (n_in % 32))
        w = words(np.concatenate([wbits, np.zeros((n_out, pal * 32 - n_in), dtype=np.int64)], axis=1), pal)
        raw = (2 * abits - 1) @ (2 * wbits - 1).T
        th = RNG.integers(-20, 21, n_out) + raw              # thresholds around the scores (ties included)
        th[0] = -2 ** 31
        if n_out > 1:
            th[1] = 2 ** 31 - 1
        b = (raw >= th).astype(np.int64)
        tw = [int(sum(int(b[j]) << (j - 32 * k) for j in range(32 * k, min(32 * k + 32, n_out))))
              for k in range((n_out + 31) // 32)]                  # padding bits of the last word must be 0
        u8 = x @ (2 * wbits - 1).T
        th2 = RNG.integers(-20, 21, n_out) + u8                  # thresholds around the first-layer scores
        b2 = (u8 >= th2).astype(np.int64)
        tw2 = [int(sum(int(b2[j]) << (j - 32 * k) for j in range(32 * k, min(32 * k + 32, n_out))))
               for k in range((n_out + 31) // 32)]
        exp.append(np.concatenate([raw, [int(np.argmax(raw))], u8, u8, u8, tw, tw, tw2]))
        data += i32(n_in, n_out) + inp.tobytes() + w.tobytes() + x.astype(np.uint8).tobytes() + th.astype(np.int32).tobytes() \
            + th2.astype(np.int32).tobytes()
        total += 1
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return total, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_dense_planes_k(hal, n):
    exe = build("planesk", """
int main(void) { int n_in, n_out, k; static uint32_t w[256 * 160], planes[8 * 256], bits[5];
  static int32_t th[160]; static uint8_t x[8192];
  while (fread(&n_in, 4, 1, stdin) == 1 && fread(&n_out, 4, 1, stdin) == 1 && fread(&k, 4, 1, stdin) == 1) {
    int pal = (n_in + 31) / 32;
    if (fread(w, 4, n_out * pal, stdin) != (size_t)(n_out * pal) || fread(x, 1, n_in, stdin) != (size_t)n_in ||
        fread(th, 4, n_out, stdin) != (size_t)n_out) return 1;
    struct SymbexLayerData L = { (uint16_t)n_in, (uint16_t)n_out, 1, w };
    memset(bits, 0xFF, sizeof bits);
    symbex_dense_input_u8_planes_bits(x, (uint8_t)k, planes, th, bits, &L);
    for (int q = 0; q < (n_out + 31) / 32; q++) printf("%u ", bits[q]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c in range(n):
        n_in, n_out, k = pick(1, 1300, c), pick(1, 41, c), int(RNG.integers(1, 9))
        pal = (n_in + 31) // 32
        wbits = bits_data((n_out, n_in), c + 3)
        x = u8_data(n_in, c)                                     # NOT pre-quantized: the function must mask
        xq = (x >> (8 - k)) << (8 - k)
        u8 = xq @ (2 * wbits - 1).T
        th = RNG.integers(-20, 21, n_out) + u8
        b = (u8 >= th).astype(np.int64)
        exp.append(np.array([int(sum(int(b[j]) << (j - 32 * q) for j in range(32 * q, min(32 * q + 32, n_out))))
                             for q in range((n_out + 31) // 32)], dtype=np.int64))
        data += i32(n_in, n_out, k) + words(np.concatenate([wbits, np.zeros((n_out, pal * 32 - n_in), dtype=np.int64)], axis=1), pal).tobytes() \
            + x.astype(np.uint8).tobytes() + th.astype(np.int32).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"word {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_binary_classifier(hal, n):
    """symbex_binary_classifier_argmax: XNOR raw score per class, then (raw * scale + half) >> (q_mult - q_act)
    + shift_bn in int64 (round to nearest), and the FIRST class with the largest logit."""
    exe = build("classifier", """
int main(void) { int a4[4]; static uint32_t inp[64], w[64 * 160]; static int64_t sc[160]; static int32_t sh[160];
  while (fread(a4, 4, 4, stdin) == 4) {
    int n_in = a4[0], n_out = a4[1], qa = a4[2], qm = a4[3], pal = (n_in + 31) / 32;
    if (fread(inp, 4, pal, stdin) != (size_t)pal || fread(w, 4, n_out * pal, stdin) != (size_t)(n_out * pal)
        || fread(sc, 8, n_out, stdin) != (size_t)n_out || fread(sh, 4, n_out, stdin) != (size_t)n_out) return 1;
    struct SymbexLayerData L = { (uint16_t)n_in, (uint16_t)n_out, 1, w };
    printf("%d\\n", symbex_binary_classifier_argmax(inp, &L, sc, sh, (uint8_t)qa, (uint8_t)qm)); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c in range(n):
        n_in, n_out = min(pick(1, 1300, c), 2048), int(RNG.integers(2, 120))
        qa = int(RNG.integers(4, 13))
        qm = int(RNG.integers(qa + 1, 25))
        pal = (n_in + 31) // 32
        abits, wbits = bits_data(n_in, c), bits_data((n_out, n_in), c + 3)
        scale = RNG.integers(-(1 << 20), 1 << 20, n_out)
        shift = RNG.integers(-(1 << 12), 1 << 12, n_out)
        if c % 4 == 0:                                       # ties on purpose: the first class must win
            scale[:] = scale[0]
            shift[:] = shift[0]
        raw = (2 * wbits - 1) @ (2 * abits - 1)
        sr = qm - qa
        logit = ((raw * scale + (1 << (sr - 1))) >> sr) + shift
        exp.append(int(np.argmax(logit)))
        inp = words(abits, pal)
        if n_in % 32:
            inp[-1] |= np.uint32((int(RNG.integers(0, 2 ** 32)) >> (n_in % 32)) << (n_in % 32))
        data += i32(n_in, n_out, qa, qm) + inp.tobytes() + words(wbits, pal).astype(np.uint32).tobytes() \
            + scale.astype(np.int64).tobytes() + shift.astype(np.int32).tobytes()
    got = run(exe, data)
    bad = np.nonzero(got != np.array(exp))[0]
    return n, len(bad), (f"case {bad[0]}: got class {got[bad[0]]}, expected {exp[bad[0]]}" if len(bad) else "")


def test_network_dense(hal, n):
    exe = build("netdense", """
int main(void) { int nl; static uint32_t inp[256], wts[8][256 * 1200]; static uint32_t ba[256], bb[256];
  static int32_t scores[8192]; static bool act[64]; static struct SymbexLayerData L[8]; int dims[9];
  while (fread(&nl, 4, 1, stdin) == 1 && fread(dims, 4, nl + 1, stdin) == (size_t)(nl + 1)) {
    if (fread(inp, 4, (dims[0] + 31) / 32, stdin) != (size_t)((dims[0] + 31) / 32)) return 1;
    struct SymbexNetworkDense net; symbex_network_dense_init(&net);
    for (int i = 0; i < nl; i++) {
      int pal = (dims[i] + 31) / 32;
      if (fread(wts[i], 4, dims[i + 1] * pal, stdin) != (size_t)(dims[i + 1] * pal)) return 1;
      L[i].in_features = dims[i]; L[i].out_features = dims[i + 1]; L[i].k_bits = 1; L[i].weights = wts[i];
      symbex_network_dense_add_dense(&net, &L[i]); }
    printf("%d\\n", symbex_network_dense_run(&net, inp, ba, bb, scores, act)); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    n = min(n, 500) if STRESS else n                # whole networks of up to 4 large layers: 500 are plenty
    for _ in range(n):
        nl = int(RNG.integers(1, 5))
        dims = [pick(1, 700, _)] + [pick(1, 300, _ + k) for k in range(nl - 1)] + [int(RNG.integers(2, 12))]
        a = RNG.integers(0, 2, dims[0])
        data += i32(nl, *dims) + words(np.concatenate([a, np.zeros(-(-dims[0] // 32) * 32 - dims[0], dtype=np.int64)]),
                                       -(-dims[0] // 32)).tobytes()
        cls = None
        for i in range(nl):
            wb = RNG.integers(0, 2, (dims[i + 1], dims[i]))
            pal = -(-dims[i] // 32)
            data += words(np.concatenate([wb, np.zeros((dims[i + 1], pal * 32 - dims[i]), dtype=np.int64)], axis=1), pal).tobytes()
            raw = (2 * a - 1) @ (2 * wb - 1).T
            if i == nl - 1:
                cls = int(np.argmax(raw))
            else:
                a = (raw >= 0).astype(np.int64)
        exp.append(cls)
    got = run(exe, data)
    bad = np.nonzero(got != np.array(exp))[0]
    return n, len(bad), (f"network {bad[0]}: got class {got[bad[0]]}, expected {exp[bad[0]]}" if len(bad) else "")


# ---------------------------------------------------------------------------------------------------
# convolutional path: BPR, binary maps, windows, DSP fixed point, residual networks
# ---------------------------------------------------------------------------------------------------
def test_bpr(hal, n):
    exe = build("bpr", """
int main(void) { int h, w; static uint8_t img[64 * 64 * 3]; static uint32_t out[64 * 64];
  while (fread(&h, 4, 1, stdin) == 1 && fread(&w, 4, 1, stdin) == 1 && fread(img, 1, h * w * 3, stdin) == (size_t)(h * w * 3)) {
    symbex_rgb_to_24bits(img, out, (uint16_t)h, (uint16_t)w);
    for (int i = 0; i < h * w; i++) printf("%u ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c in range(n):
        h, w = int(RNG.integers(1, 17)), int(RNG.integers(1, 17))
        img = u8_data(h * w * 3, c).reshape(h, w, 3)
        word = np.zeros(h * w, dtype=np.int64)
        for color in range(3):
            for j in range(8):
                word |= ((img[:, :, color].ravel() >> (7 - j)) & 1) << (color * 8 + j)
        exp.append(word)
        data += i32(h, w) + img.astype(np.uint8).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"word {bad[0]}: got 0x{got[bad[0]]:06X}, expected 0x{flat[bad[0]]:06X}" if len(bad) else "")


def test_binarize_image(hal, n):
    exe = build("binimg", """
int main(void) { int c, h, w; static float x[96 * 16 * 16]; static uint32_t out[16 * 16 * 3];
  while (fread(&c, 4, 1, stdin) == 1 && fread(&h, 4, 1, stdin) == 1 && fread(&w, 4, 1, stdin) == 1
         && fread(x, 4, c * h * w, stdin) == (size_t)(c * h * w)) {
    int pw = (c + 31) / 32;
    memset(out, 0xAB, sizeof out);
    symbex_binarize_image(x, out, (uint16_t)c, (uint16_t)h, (uint16_t)w);
    for (int i = 0; i < h * w * pw; i++) printf("%u ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for k in range(n):
        ch, h, w = min(pick(1, 97, k), 96), int(RNG.integers(1, 17)), int(RNG.integers(1, 17))
        x = RNG.normal(size=(ch, h, w)).astype(np.float32)
        x[RNG.random((ch, h, w)) < 0.1] = 0.0                     # exact zeros: >= 0 -> bit 1
        x[RNG.random((ch, h, w)) < 0.05] = -0.0
        bits = (x >= 0).astype(np.int64).transpose(1, 2, 0)       # HWC
        exp.append(hwc_words(bits, (ch + 31) // 32).astype(np.int64))
        data += i32(ch, h, w) + x.tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"word {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_binary_pool(hal, n):
    exe = build("pool", """
int main(void) { int c, h, w; static uint32_t inp[32 * 32 * 4], out[16 * 16 * 4];
  while (fread(&c, 4, 1, stdin) == 1 && fread(&h, 4, 1, stdin) == 1 && fread(&w, 4, 1, stdin) == 1) {
    int pw = (c + 31) / 32;
    if (fread(inp, 4, h * w * pw, stdin) != (size_t)(h * w * pw)) return 1;
    memset(out, 0xAB, sizeof out);
    symbex_binary_pool_2x2(inp, out, (uint16_t)c, (uint16_t)h, (uint16_t)w);
    for (int i = 0; i < (h / 2) * (w / 2) * pw; i++) printf("%u ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp_bits, total = Chunks(), [], 0
    for k in range(n):
        ch, h, w = min(pick(1, 129, k), 128), int(RNG.integers(2, 33)), int(RNG.integers(2, 33))
        pw = (ch + 31) // 32
        bits = bits_data((h, w, ch), k)
        words_in = hwc_words(bits, pw)
        if ch % 32:                                              # garbage in the unused channel bits
            garbage = RNG.integers(0, 2 ** 32, h * w, dtype=np.uint64).astype(np.uint32)
            words_in.reshape(h * w, pw)[:, -1] |= (garbage >> (ch % 32)) << (ch % 32)
        oh, ow = h // 2, w // 2
        blocks = bits[:2 * oh, :2 * ow].reshape(oh, 2, ow, 2, ch).sum((1, 3))
        exp_bits.append(((blocks >= 2).astype(np.int64), pw, ch))
        data += i32(ch, h, w) + words_in.astype(np.uint32).tobytes()
        total += 1
    got = run(exe, data)
    pos, bad, first = 0, 0, ""
    for e, pw, ch in exp_bits:
        oh, ow = e.shape[:2]
        g = got[pos:pos + oh * ow * pw].reshape(oh * ow, pw)
        pos += oh * ow * pw
        gb = ((g[:, :, None] >> np.arange(32)) & 1).reshape(oh * ow, pw * 32)[:, :ch].reshape(oh, ow, ch)
        if (gb != e).any():
            bad += 1
            if not first:
                y, x, c = (int(v) for v in np.argwhere(gb != e)[0])
                first = f"case {bad - 1}: pixel ({y},{x}) channel {c}: got {gb[y, x, c]}, expected {e[y, x, c]}"
    return total, bad, first


def test_extract_window(hal, n):
    exe = build("window", """
int main(void) { int a[9]; static uint32_t map[32 * 32 * 8], patch[512], mask[512];
  while (fread(a, 4, 9, stdin) == 9) {
    int c = a[0], h = a[1], w = a[2], k = a[3], s = a[4], p = a[5], oh = a[6], ow = a[7], pw = (c + 31) / 32;
    if (fread(map, 4, h * w * pw, stdin) != (size_t)(h * w * pw)) return 1;
    memset(patch, 0xAB, sizeof patch); memset(mask, 0xAB, sizeof mask);
    symbex_conv2d_extract_window(map, patch, mask, (uint16_t)c, (uint16_t)h, (uint16_t)w, (uint16_t)k, (uint16_t)s,
                                 (uint16_t)p, (uint16_t)oh, (uint16_t)ow);
    for (int i = 0; i < a[8]; i++) printf("%u ", patch[i]);
    for (int i = 0; i < a[8]; i++) printf("%u ", mask[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp, total = Chunks(), [], 0
    for kcase in range(n):
        ch = min(pick(1, 97, kcase), 160)
        k = int(RNG.choice([1, 3, 5]))
        if ch * k * k > 4096:
            ch = max(1, 4096 // (k * k))
        h, w = int(RNG.integers(1, 17)), int(RNG.integers(1, 17))
        s, p = int(RNG.integers(1, 3)), int(RNG.integers(0, k // 2 + 1))
        out_h, out_w = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        if out_h < 1 or out_w < 1:
            continue
        oh, ow = int(RNG.integers(0, out_h)), int(RNG.integers(0, out_w))
        pw = (ch + 31) // 32
        bits = bits_data((h, w, ch), kcase)
        m = hwc_words(bits, pw)
        if ch % 32:                                              # garbage above the channels: must be masked
            garbage = RNG.integers(0, 2 ** 32, h * w, dtype=np.uint64).astype(np.uint32)
            m.reshape(h * w, pw)[:, -1] |= (garbage >> (ch % 32)) << (ch % 32)
        elems = ch * k * k
        patch_words = (((elems + 31) // 32) + 3) & ~3
        pb = np.zeros(patch_words * 32, dtype=np.int64)
        mb = np.zeros(patch_words * 32, dtype=np.int64)
        for kh in range(k):
            for kw in range(k):
                y, x = oh * s - p + kh, ow * s - p + kw
                off = (kh * k + kw) * ch
                if 0 <= y < h and 0 <= x < w:
                    pb[off:off + ch] = bits[y, x]
                else:
                    mb[off:off + ch] = 1
        pb[elems:] = 1
        exp.append(np.concatenate([words(pb, patch_words), words(mb, patch_words)]).astype(np.int64))
        data += i32(ch, h, w, k, s, p, oh, ow, patch_words) + m.astype(np.uint32).tobytes()
        total += 1
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return total, len(bad), (f"word {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def dsp_ref(raw, ident, p, q_act, q_mult):
    """symbex_dsp_apply_layer for arrays (channel axis 0 of raw/ident, per-channel params as arrays of shape (C,))."""
    sr = q_mult - q_act
    sh = lambda a: np.asarray(a, dtype=np.int64).reshape((-1,) + (1,) * (raw.ndim - 1))
    y = ((raw.astype(np.int64) * sh(p["scale"])) + (1 << (sr - 1))) >> sr
    y = y + sh(p["shift_bn"]) + ident.astype(np.int64)
    diff = y - sh(p["gamma"])
    if p["use_shift"]:
        neg = np.where(diff > 0, 0, -diff)
        l2 = sh(p["log2_beta"])
        mag = np.where(l2 >= 0, neg << np.maximum(l2, 0), neg >> np.maximum(-l2, 0))
        negb = np.where(sh(p["sign_beta"]) >= 0, -mag, mag)
    else:
        negb = (diff * sh(p["slope"]) + (1 << (q_act - 1))) >> q_act
    return np.where(diff > 0, diff, negb) + sh(p["zeta"])


def rand_dsp(c, q_act):
    use_shift = bool(RNG.integers(0, 2))
    return dict(scale=RNG.integers(-2047, 2048, c), shift_bn=RNG.integers(-(1 << 10), 1 << 10, c),
                gamma=RNG.integers(-(1 << 9), 1 << 9, c), zeta=RNG.integers(-(1 << 9), 1 << 9, c), use_shift=use_shift,
                log2_beta=RNG.integers(-5, 3, c), sign_beta=RNG.choice([-1, 1], c),
                slope=RNG.integers(-(2 << q_act), 2 << q_act, c))


def dsp_bytes(p):
    return (np.concatenate([p["scale"], p["shift_bn"], p["gamma"], p["zeta"], p["log2_beta"], p["sign_beta"],
                            p["slope"]]).astype(np.int32).tobytes() + i32(int(p["use_shift"])))


C_READ_DSP = """
static int32_t pool32[1 << 20]; static int8_t pool8[1 << 16]; static size_t n32, n8;
static const int32_t* take32(int n) { int32_t* p = &pool32[n32]; if (fread(p, 4, n, stdin) != (size_t)n) exit(2); n32 += n; return p; }
static void read_dsp(SymbexDspParams* d, int c, int qa, int qm) {
  d->scale = take32(c); d->shift_bn = take32(c); d->gamma_rprelu = take32(c); d->zeta = take32(c);
  const int32_t* l2 = take32(c); const int32_t* sg = take32(c); d->slope_q = take32(c);
  int8_t* L = &pool8[n8]; int8_t* S = &pool8[n8 + c]; n8 += 2 * c;
  for (int i = 0; i < c; i++) { L[i] = (int8_t)l2[i]; S[i] = (int8_t)sg[i]; }
  d->log2_beta = L; d->sign_beta = S; int us; if (fread(&us, 4, 1, stdin) != 1) exit(2); d->use_shift = us != 0;
  d->q_act = (uint8_t)qa; d->q_mult = (uint8_t)qm; }
"""


def test_dsp(hal, n):
    exe = build("dsp", C_READ_DSP + """
int main(void) { int c, qa, qm;
  while (fread(&c, 4, 1, stdin) == 1 && fread(&qa, 4, 1, stdin) == 1 && fread(&qm, 4, 1, stdin) == 1) {
    n32 = n8 = 0; SymbexDspParams d; read_dsp(&d, c, qa, qm);
    const int32_t* raw = take32(c); const int32_t* id = take32(c); const int32_t* beta = take32(c);
    SymbexDspParamsSimple s = { d.scale, d.shift_bn, (uint8_t)qa, (uint8_t)qm };
    for (int i = 0; i < c; i++) printf("%d %d %d ", symbex_dsp_apply_layer(raw[i], id[i], (uint16_t)i, &d),
                                       (int)symbex_dsp_binarize(raw[i], beta[i]), symbex_dsp_apply_downsample(raw[i], (uint16_t)i, &s));
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for k in range(n):
        c = int(RNG.integers(1, 65))
        qa = int(RNG.integers(4, 13))
        qm = int(RNG.integers(qa + 1, 25))
        p = rand_dsp(c, qa)
        raw = RNG.integers(-1200, 1201, c)
        ident = RNG.integers(-(1 << 14), 1 << 14, c)
        beta = raw + RNG.integers(-2, 3, c)                      # ties included: value >= beta -> 1
        a = dsp_ref(raw, ident, p, qa, qm)
        sr = qm - qa
        ds = ((raw * p["scale"] + (1 << (sr - 1))) >> sr) + p["shift_bn"]
        exp.append(np.stack([a, (raw >= beta).astype(np.int64), ds], 1).ravel())
        data += i32(c, qa, qm) + dsp_bytes(p) + np.concatenate([raw, ident, beta]).astype(np.int32).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    what = ["apply_layer", "binarize", "apply_downsample"]
    return n, len(bad), (f"value {bad[0]} ({what[bad[0] % 3]}): got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def conv_raw(bits, wb, k, s, p):
    """bits (C, H, W) 0/1, wb (O, K, K, C) 0/1 -> raw (O, OH, OW): sum of (2a-1)(2w-1) over the in-image positions
    (zero padding contributes nothing, as in PyTorch)."""
    c, h, w = bits.shape
    oh_n, ow_n = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
    a = np.zeros((c, h + 2 * p, w + 2 * p), dtype=np.int64)
    a[:, p:p + h, p:p + w] = 2 * bits - 1                       # padding stays 0
    sw = 2 * wb.astype(np.int64) - 1
    out = np.zeros((wb.shape[0], oh_n, ow_n), dtype=np.int64)
    for y in range(oh_n):
        for x in range(ow_n):
            patch = a[:, y * s:y * s + k, x * s:x * s + k].transpose(1, 2, 0)      # (K, K, C)
            out[:, y, x] = (sw * patch[None]).sum((1, 2, 3))
    return out


def conv_weight_words(wb):
    """(O, K, K, C) bits -> O * patch_words words, bit (kh*K + kw)*C + c, zeros after the window."""
    o, k, _, c = wb.shape
    elems = c * k * k
    pwds = (((elems + 31) // 32) + 3) & ~3
    flat = np.zeros((o, pwds * 32), dtype=np.int64)
    flat[:, :elems] = wb.reshape(o, elems)
    return words(flat, pwds).ravel()


def test_network_conv(hal, n):
    exe = build("netconv", C_READ_DSP + """
int main(void) { int hdr[6];
  static struct SymbexResidualBlock B[4]; static int32_t ba[8192], bb[8192], ids[8192], out[8192];
  static uint32_t bits[1024], bfull[1024], bpool[1024], line[64], patch[64]; static uint32_t wpool[1 << 18];
  while (fread(hdr, 4, 6, stdin) == 6) {
    int nb = hdr[0], c = hdr[1], h = hdr[2], w = hdr[3], qa = hdr[4], qm = hdr[5]; size_t nw = 0;
    n32 = n8 = 0;
    const int32_t* x0 = take32(c * h * w);
    struct SymbexNetworkConv net; symbex_network_conv_init(&net);
    int last_c = c, last_h = h, last_w = w;
    for (int i = 0; i < nb; i++) {
      int bh[3]; if (fread(bh, 4, 3, stdin) != 3) return 2;
      int ds = bh[0], oc = bh[1], st = bh[2], h1 = (last_h + 2 - 3) / st + 1, w1 = (last_w + 2 - 3) / st + 1;
      struct SymbexResidualBlock* b = &B[i]; memset(b, 0, sizeof *b);
      int pw1 = ((((last_c * 9) + 31) / 32) + 3) & ~3, pw2 = ((((oc * 9) + 31) / 32) + 3) & ~3;
      read_dsp(&b->conv1.dsp, oc, qa, qm);
      if (fread(&wpool[nw], 4, oc * pw1, stdin) != (size_t)(oc * pw1)) return 2;
      b->conv1.weights = &wpool[nw]; nw += oc * pw1;
      b->conv1.in_ch = last_c; b->conv1.in_h = last_h; b->conv1.in_w = last_w;
      b->conv1.out_ch = oc; b->conv1.kernel = 3; b->conv1.stride = st; b->conv1.padding = 1;
      b->beta_rsign1 = take32(last_c); b->beta_rsign2 = take32(oc);
      read_dsp(&b->conv2.dsp, oc, qa, qm);
      if (fread(&wpool[nw], 4, oc * pw2, stdin) != (size_t)(oc * pw2)) return 2;
      b->conv2.weights = &wpool[nw]; nw += oc * pw2;
      b->conv2.in_ch = oc; b->conv2.in_h = h1; b->conv2.in_w = w1;
      b->conv2.out_ch = oc; b->conv2.kernel = 3; b->conv2.stride = 1; b->conv2.padding = 1;
      b->has_downsample = ds != 0;
      if (ds) {
        b->beta_rsign_downsample = take32(last_c);
        b->downsample_dsp.scale = take32(oc); b->downsample_dsp.shift_bn = take32(oc);
        b->downsample_dsp.q_act = (uint8_t)qa; b->downsample_dsp.q_mult = (uint8_t)qm;
        if (fread(&wpool[nw], 4, oc * pw1, stdin) != (size_t)(oc * pw1)) return 2;
        b->downsample_weights = &wpool[nw]; nw += oc * pw1;
        b->downsample_kernel = 3; b->downsample_padding = 1; }
      symbex_network_conv_add_block(&net, b);
      last_c = oc; last_h = h1; last_w = w1; }
    if (!symbex_network_conv_run_ext(&net, x0, ba, bb, bits, ids, bfull, bpool, line, patch, out)) { printf("FAIL\\n"); continue; }
    for (int i = 0; i < last_c * last_h * last_w; i++) printf("%d ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    n = min(n, 120) if not STRESS else min(n, 300)      # whole networks: a few hundred are plenty
    for k in range(n):
        qa = int(RNG.integers(6, 11))
        qm = int(RNG.integers(qa + 4, 22))
        c, h, w = int(RNG.choice([1, 3, 8, 16, 24, 33])), int(RNG.choice([4, 6, 8])), int(RNG.choice([4, 6, 8]))
        nb = int(RNG.integers(1, 4))
        x = RNG.integers(-(1 << 11), 1 << 11, (c, h, w))
        hdr = i32(nb, c, h, w, qa, qm) + x.astype(np.int32).tobytes()
        cur, cc, ch_, cw = x.astype(np.int64), c, h, w
        blocks = b""
        for i in range(nb):
            ds = int(RNG.random() < 0.5 and ch_ % 2 == 0 and cw % 2 == 0 and ch_ >= 4)   # even sizes for the 2x2 pool
            oc = int(RNG.choice([8, 16, 32, 40])) if ds else cc
            st = 2 if ds else 1
            p1, p2 = rand_dsp(oc, qa), rand_dsp(oc, qa)
            w1 = RNG.integers(0, 2, (oc, 3, 3, cc))
            w2 = RNG.integers(0, 2, (oc, 3, 3, oc))
            beta1 = RNG.integers(-200, 200, cc)
            beta2 = RNG.integers(-200, 200, oc)
            blocks_b = i32(ds, oc, st) + dsp_bytes(p1) + conv_weight_words(w1).astype(np.uint32).tobytes() \
                + np.concatenate([beta1, beta2]).astype(np.int32).tobytes() + dsp_bytes(p2) \
                + conv_weight_words(w2).astype(np.uint32).tobytes()
            if ds:
                beta_ds = RNG.integers(-200, 200, cc)
                ds_scale, ds_shift = RNG.integers(-2047, 2048, oc), RNG.integers(-(1 << 10), 1 << 10, oc)
                wds = RNG.integers(0, 2, (oc, 3, 3, cc))
                blocks_b += np.concatenate([beta_ds, ds_scale, ds_shift]).astype(np.int32).tobytes() \
                    + conv_weight_words(wds).astype(np.uint32).tobytes()
                bits_ds = (cur >= beta_ds[:, None, None]).astype(np.int64)
                pooled = (bits_ds.reshape(cc, ch_ // 2, 2, cw // 2, 2).sum((2, 4)) >= 2).astype(np.int64)
                sr = qm - qa
                raw = conv_raw(pooled, wds, 3, 1, 1)
                ident = ((raw * ds_scale[:, None, None] + (1 << (sr - 1))) >> sr) + ds_shift[:, None, None]
            else:
                ident = cur
            b1 = (cur >= beta1[:, None, None]).astype(np.int64)
            y1 = dsp_ref(conv_raw(b1, w1, 3, st, 1), ident, p1, qa, qm)
            b2 = (y1 >= beta2[:, None, None]).astype(np.int64)
            cur = dsp_ref(conv_raw(b2, w2, 3, 1, 1), y1, p2, qa, qm)
            cc, ch_, cw = oc, cur.shape[1], cur.shape[2]
            blocks += blocks_b
        data += hdr + blocks
        exp.append(cur.ravel())
    got = run(exe, data)
    flat = np.concatenate(exp)
    if len(got) != len(flat):
        return n, n, f"printed {len(got)} values, expected {len(flat)} (a network failed to run)"
    bad = np.nonzero(got != flat)[0]
    ends = np.cumsum([len(e) for e in exp])
    nets = sorted({int(np.searchsorted(ends, b, side="right")) for b in bad})          # failing networks
    return n, len(nets), (f"network {nets[0]}, value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_network_conv_limits(hal, n):
    """symbex_network_conv_add_block must accept valid blocks and reject the ones that do not fit the library."""
    exe = build("convlimits", """
static uint32_t W[1 << 16]; static int32_t I[4096]; static int8_t B8[4096];
static void dsp(SymbexDspParams* d, int qa, int qm) { memset(d, 0, sizeof *d); d->scale = I; d->shift_bn = I; d->gamma_rprelu = I;
  d->zeta = I; d->slope_q = I; d->log2_beta = B8; d->sign_beta = B8; d->q_act = (uint8_t)qa; d->q_mult = (uint8_t)qm; }
static void conv(struct SymbexConv2DLayer* L, int ic, int h, int oc, int k, int s, int p, int qa, int qm) {
  memset(L, 0, sizeof *L); dsp(&L->dsp, qa, qm); L->weights = W; L->in_ch = ic; L->in_h = h; L->in_w = h;
  L->out_ch = oc; L->kernel = k; L->stride = s; L->padding = p; }
int main(void) { int a[10];
  while (fread(a, 4, 10, stdin) == 10) {   /* in_ch, h, out_ch, stride, downsample, qa, qm, conv2_in_ch, ds_kernel, out_ch2 */
    struct SymbexResidualBlock b; memset(&b, 0, sizeof b);
    int ic = a[0], h = a[1], oc = a[2], s = a[3], ds = a[4], qa = a[5], qm = a[6];
    int h1 = (h + 2 - 3) / s + 1;
    conv(&b.conv1, ic, h, oc, 3, s, 1, qa, qm); conv(&b.conv2, a[7], h1, a[9], 3, 1, 1, qa, qm);
    b.beta_rsign1 = I; b.beta_rsign2 = I; b.has_downsample = ds != 0;
    if (ds) { b.downsample_weights = W; b.beta_rsign_downsample = I; b.downsample_dsp.scale = I; b.downsample_dsp.shift_bn = I;
              b.downsample_dsp.q_act = (uint8_t)qa; b.downsample_dsp.q_mult = (uint8_t)qm; b.downsample_kernel = (uint16_t)a[8]; b.downsample_padding = 1; }
    struct SymbexNetworkConv net; symbex_network_conv_init(&net);
    printf("%d\\n", (int)symbex_network_conv_add_block(&net, &b)); }
  return 0; }""", hal)
    #          in_ch h  out stride ds qa qm  c2in ds_k c2out  expected
    cases = [((16, 8, 16, 1, 0, 8, 20, 16, 3, 16), 1, "valid block"),
             ((16, 8, 32, 2, 1, 8, 20, 32, 3, 32), 1, "valid block with downsample"),
             ((227, 8, 227, 1, 0, 8, 20, 227, 3, 227), 1, "227 channels: the largest window that fits"),
             ((228, 8, 228, 1, 0, 8, 20, 228, 3, 228), 0, "228 channels: window over 64 words"),
             ((16, 8, 257, 2, 1, 8, 20, 257, 3, 257), 0, "257 output channels"),
             ((16, 8, 16, 1, 0, 8, 8, 16, 3, 16), 0, "q_mult == q_act"),
             ((16, 8, 16, 1, 0, 0, 20, 16, 3, 16), 0, "q_act == 0"),
             ((16, 8, 16, 1, 0, 8, 20, 8, 3, 16), 0, "conv2 input channels differ from conv1 output"),
             ((16, 8, 32, 1, 0, 8, 20, 32, 3, 32), 0, "no downsample but the channels change"),
             ((16, 8, 32, 2, 0, 8, 20, 32, 3, 32), 0, "no downsample but the size halves"),
             ((16, 8, 32, 2, 1, 8, 20, 32, 5, 32), 0, "downsample 5x5 (2x2 output vs conv1's 4x4)"),
             ((16, 8, 32, 2, 1, 8, 20, 32, 7, 32), 0, "downsample whose output size differs from conv1's")]
    data = Chunks()
    for args, _, _ in cases:
        data += i32(*args)
    got = run(exe, data)
    bad = [i for i, (_, e, _) in enumerate(cases) if got[i] != e]
    return len(cases), len(bad), (f"{cases[bad[0]][2]}: add_block returned {got[bad[0]]}, expected {cases[bad[0]][1]}" if bad else "")


# ---------------------------------------------------------------------------------------------------
# fully binary networks (layers/conv/symbex_bool)
# ---------------------------------------------------------------------------------------------------
def bool_conv_ref(bits, wb, thr, k, s):
    """bits (H, W, C) 0/1, wb (O, k, k, C) 0/1 -> out bits (Ho, Wo, O): raw over the in-image window >= thr"""
    h, w, c = bits.shape
    p = k // 2
    a = np.zeros((h + 2 * p, w + 2 * p, c), dtype=np.int64)
    a[p:p + h, p:p + w] = 2 * bits - 1                      # outside the image: 0, adds nothing
    sw = 2 * wb.astype(np.int64) - 1
    ho, wo = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
    out = np.zeros((ho, wo, wb.shape[0]), dtype=np.int64)
    for y in range(ho):
        for x in range(wo):
            raw = (sw * a[y * s:y * s + k, x * s:x * s + k][None]).sum((1, 2, 3))
            out[y, x] = raw >= thr
    return out


def bool_conv_words(wb):
    """(O, k, k, C) bits -> (O * row_words) words, PACKED (the C bits of kernel pixel q at bit q * C), rows zero
    padded to a multiple of 4 words; counts (O, k*k)"""
    o, k, _, c = wb.shape
    n = -(-(k * k * c) // 32)
    rw = -(-n // 4) * 4
    rows = np.zeros((o, rw), dtype=np.uint32)
    rows[:, :n] = words(wb.reshape(o, k * k * c), n)
    return rows.ravel(), rw, wb.reshape(o, k * k, c).sum(2)


def test_bool_conv(hal, n):
    exe = build("boolconv", """
#include "layers/conv/symbex_bool.h"
int main(void) { int a[6]; static uint32_t in[16 * 16 * 4], out[16 * 16 * 4], w[128 * 40] __attribute__((aligned(16)));
  static uint32_t win[64] __attribute__((aligned(16))); static int32_t thr[128], cnt32[128 * 9], sc[256]; static int16_t cnt[128 * 9];
  while (fread(a, 4, 6, stdin) == 6) {
    int cin = a[0], cout = a[1], k = a[2], s = a[3], H = a[4], W = a[5], pin = (cin + 31) / 32, pout = (cout + 31) / 32;
    int st = ((k * k * cin + 31) / 32 + 3) / 4 * 4, ho = (H + 2 * (k / 2) - k) / s + 1, wo = (W + 2 * (k / 2) - k) / s + 1;
    if (fread(in, 4, H * W * pin, stdin) != (size_t)(H * W * pin) || fread(w, 4, cout * st, stdin) != (size_t)(cout * st) ||
        fread(thr, 4, cout, stdin) != (size_t)cout || fread(cnt32, 4, cout * k * k, stdin) != (size_t)(cout * k * k)) return 1;
    for (int i = 0; i < cout * k * k; i++) cnt[i] = (int16_t)cnt32[i];
    struct SymbexBoolConv L = { (uint16_t)cin, (uint16_t)cout, (uint16_t)k, (uint16_t)s, (uint16_t)H, (uint16_t)W,
                                (uint16_t)ho, (uint16_t)wo, (uint16_t)st, w, thr, cnt };
    for (int mode = 1; mode <= 2; mode++) {                 /* both count forms: AND, XOR */
      symbex_bool_force_mode = mode;
      memset(out, 0xAB, sizeof out);
      symbex_bool_conv(&L, in, out, win, sc);
      for (int i = 0; i < ho * wo * pout; i++) printf("%u ", out[i]); }
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c_ in range(n):
        cin, cout = min(pick(1, 97, c_), 128), int(RNG.integers(1, 71))
        k, s = int(RNG.choice([1, 3])), int(RNG.integers(1, 3))
        h, w = int(RNG.integers(1, 13)), int(RNG.integers(1, 13))
        bits = bits_data((h, w, cin), c_)
        wb = bits_data((cout, k, k, cin), c_ + 3)
        thr = RNG.integers(-k * k * cin, k * k * cin + 1, cout)
        if c_ % 5 == 0:
            thr[:] = RNG.integers(-3, 4, cout)                  # thresholds near 0: many ties
        wwords, rw, cnts = bool_conv_words(wb)
        pin, pout = (cin + 31) // 32, (cout + 31) // 32
        m = hwc_words(bits, pin)
        o = bool_conv_ref(bits, wb, thr, k, s)
        exp += [hwc_words(o, pout).astype(np.int64)] * 2                  # AND form, then XOR form
        data += i32(cin, cout, k, s, h, w) + m.astype(np.uint32).tobytes() + wwords.tobytes() \
            + thr.astype(np.int32).tobytes() + cnts.astype(np.int32).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"word {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_bool_pieces(hal, n):
    """thermometer, 2x2 OR pooling, OR of maps and the integer classifier"""
    exe = build("boolpieces", """
#include "layers/conv/symbex_bool.h"
int main(void) { int a[5]; static uint8_t img[16 * 16 * 3]; static uint32_t t[16 * 16 * 4], m[16 * 16 * 4], p[16 * 16 * 4];
  static uint32_t fw[10 * 64] __attribute__((aligned(16))), bits[64] __attribute__((aligned(16)));
  static int32_t cnt[10], sc[10], bi[10], scr[10];
  while (fread(a, 4, 5, stdin) == 5) {
    int h = a[0], w = a[1], lv = a[2], ch = a[3], nw = a[4], tp = (3 * lv + 31) / 32, pp = (ch + 31) / 32, rw = (nw + 3) / 4 * 4;
    if (fread(img, 1, h * w * 3, stdin) != (size_t)(h * w * 3) || fread(m, 4, h * w * pp, stdin) != (size_t)(h * w * pp) ||
        fread(bits, 4, rw, stdin) != (size_t)rw || fread(fw, 4, 10 * rw, stdin) != (size_t)(10 * rw) ||
        fread(cnt, 4, 10, stdin) != 10 || fread(sc, 4, 10, stdin) != 10 || fread(bi, 4, 10, stdin) != 10) return 1;
    symbex_thermometer(img, (uint16_t)h, (uint16_t)w, (uint8_t)lv, t);
    for (int i = 0; i < h * w * tp; i++) printf("%u ", t[i]);
    symbex_bool_or_pool(m, p, (uint16_t)ch, (uint16_t)h, (uint16_t)w);
    for (int i = 0; i < (h / 2) * (w / 2) * pp; i++) printf("%u ", p[i]);
    symbex_bool_or(m, t, (uint32_t)(h * w * pp < h * w * tp ? h * w * pp : h * w * tp));
    for (int i = 0; i < h * w * pp; i++) printf("%u ", m[i]);
    printf("%d\\n", symbex_bool_classify(bits, (uint16_t)nw, (uint32_t)(nw * 32), fw, (uint16_t)rw, 10, cnt, sc, bi, scr)); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c_ in range(n):
        h, w = 2 * int(RNG.integers(1, 9)), 2 * int(RNG.integers(1, 9))
        lv = int(RNG.integers(1, 11))
        ch = int(RNG.integers(1, 100))
        nw = int(RNG.integers(1, 60))
        img = u8_data(h * w * 3, c_).reshape(h, w, 3)
        thr = (np.arange(lv) + 1) * 256 // (lv + 1)
        tbits = (img[:, :, :, None] >= thr[None, None, None, :]).reshape(h, w, 3 * lv).astype(np.int64)
        tp, pp = (3 * lv + 31) // 32, (ch + 31) // 32
        t_words = hwc_words(tbits, tp)
        mb = bits_data((h, w, ch), c_)
        m_words = hwc_words(mb, pp)
        pooled = mb.reshape(h // 2, 2, w // 2, 2, ch).max((1, 3))
        n_or = min(h * w * pp, h * w * tp)
        ored = m_words.astype(np.uint64).copy()
        ored[:n_or] |= t_words[:n_or].astype(np.uint64)
        rw = -(-nw // 4) * 4
        fbits = RNG.integers(0, 2 ** 32, rw, dtype=np.uint64).astype(np.uint32)
        fbits[nw:] = RNG.integers(0, 2 ** 32, rw - nw, dtype=np.uint64).astype(np.uint32)   # garbage past n_words
        fw = np.zeros((10, rw), dtype=np.uint32)
        fw[:, :nw] = RNG.integers(0, 2 ** 32, (10, nw), dtype=np.uint64).astype(np.uint32)
        cnt = popcount(fw).sum(1)
        scale = RNG.integers(-(1 << 20), 1 << 20, 10)
        bias = RNG.integers(-(1 << 24), 1 << 24, 10)
        if c_ % 4 == 0:
            scale[:] = scale[0]
            bias[:] = bias[0]                                   # ties: the first class wins
        diff = popcount(fbits[None, :nw] ^ fw[:, :nw]).sum(1)
        logit = scale.astype(np.int64) * (nw * 32 - 2 * diff) + bias
        exp.append(np.concatenate([t_words.astype(np.int64), hwc_words(pooled, pp).astype(np.int64),
                                   ored.astype(np.int64), [int(np.argmax(logit))]]))
        data += i32(h, w, lv, ch, nw) + img.astype(np.uint8).tobytes() + m_words.astype(np.uint32).tobytes() \
            + fbits.tobytes() + fw.tobytes() + cnt.astype(np.int32).tobytes() + scale.astype(np.int32).tobytes() \
            + bias.astype(np.int32).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_bool_net(hal, n):
    """whole random networks (thermometer, stem, blocks with identity / pooled 1x1 / 1x1 shortcuts, classifier)
    through symbex_bool_net_run against a numpy reference"""
    exe = build("boolnet", """
#include "layers/conv/symbex_bool.h"
static uint32_t pool32[1 << 17] __attribute__((aligned(16))); static size_t np32;
static int32_t poolI[1 << 15]; static size_t nI; static int16_t pool16[1 << 15]; static size_t n16;
static uint32_t* take_w(int n) { uint32_t* p = &pool32[np32]; if (fread(p, 4, n, stdin) != (size_t)n) exit(2); np32 += (n + 3) / 4 * 4; return p; }
static int32_t* take_i(int n) { int32_t* p = &poolI[nI]; if (fread(p, 4, n, stdin) != (size_t)n) exit(2); nI += n; return p; }
static int16_t* take_c(int n) { int16_t* p = &pool16[n16]; for (int i = 0; i < n; i++) { int32_t v; if (fread(&v, 4, 1, stdin) != 1) exit(2); p[i] = (int16_t)v; } n16 += n; return p; }
static void read_conv(struct SymbexBoolConv* L) { int a[8]; if (fread(a, 4, 8, stdin) != 8) exit(2);
  L->in_ch = a[0]; L->out_ch = a[1]; L->kernel = a[2]; L->stride = a[3]; L->in_h = a[4]; L->in_w = a[5];
  L->out_h = (a[4] + 2 * (a[2] / 2) - a[2]) / a[3] + 1; L->out_w = (a[5] + 2 * (a[2] / 2) - a[2]) / a[3] + 1;
  L->row_words = a[6]; L->weights = take_w(a[1] * a[6]); L->thresholds = take_i(a[1]); L->weight_counts = take_c(a[1] * a[2] * a[2]); (void)a[7]; }
static uint32_t M[4][16 * 16 * 8] __attribute__((aligned(16))), WIN[128] __attribute__((aligned(16))); static int32_t SCR[512];
static struct SymbexBoolBlock B[8]; static struct SymbexBoolConv SC[8];
int main(void) { int hdr[5]; static uint8_t img[16 * 16 * 3];
  while (fread(hdr, 4, 5, stdin) == 5) {
    np32 = nI = n16 = 0;
    struct SymbexBoolNet net; memset(&net, 0, sizeof net);
    net.in_h = (uint16_t)hdr[0]; net.in_w = (uint16_t)hdr[1]; net.levels = (uint8_t)hdr[2]; net.n_blocks = (uint16_t)hdr[3];
    if (fread(img, 1, hdr[0] * hdr[1] * 3, stdin) != (size_t)(hdr[0] * hdr[1] * 3)) return 2;
    read_conv(&net.stem);
    for (int i = 0; i < net.n_blocks; i++) { int f[2]; if (fread(f, 4, 2, stdin) != 2) return 2;
      read_conv(&B[i].conv); B[i].shortcut = NULL; B[i].shortcut_pool = (uint8_t)f[1];
      if (f[0]) { read_conv(&SC[i]); B[i].shortcut = &SC[i]; } }
    net.blocks = B;
    int fc[4]; if (fread(fc, 4, 4, stdin) != 4) return 2;
    net.n_classes = (uint16_t)fc[0]; net.fc_words = (uint16_t)fc[1]; net.fc_row_words = (uint16_t)fc[2]; net.fc_bits = (uint32_t)fc[3];
    net.fc_weights = take_w(fc[0] * fc[2]); net.fc_weight_counts = take_i(fc[0]); net.fc_scale = take_i(fc[0]); net.fc_bias = take_i(fc[0]);
    for (int i = 0; i < 4; i++) { memset(M[i], 0x5A, sizeof M[i]); net.maps[i] = M[i]; }
    net.window = WIN; net.scratch = SCR;
    for (int mode = 1; mode <= 2; mode++) { symbex_bool_force_mode = mode; printf("%d ", symbex_bool_net_run(&net, img)); }
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []

    def conv_bytes(cin, cout, k, s, h, w, c_):
        wb = bits_data((cout, k, k, cin), c_)
        thr = RNG.integers(-k * k * cin // 3, k * k * cin // 3 + 1, cout)
        wwords, rw, cnts = bool_conv_words(wb)
        return (i32(cin, cout, k, s, h, w, rw, 0) + wwords.tobytes() + thr.astype(np.int32).tobytes()
                + cnts.astype(np.int32).tobytes()), wb, thr

    for c_ in range(n):
        size = int(RNG.choice([4, 8, 12, 16]))
        lv = int(RNG.integers(1, 9))
        c0 = int(RNG.choice([8, 16, 24, 33]))
        img = u8_data(size * size * 3, c_).reshape(size, size, 3)
        thr_t = (np.arange(lv) + 1) * 256 // (lv + 1)
        x = (img[:, :, :, None] >= thr_t[None, None, None, :]).reshape(size, size, 3 * lv).astype(np.int64)
        blob, wb, thr = conv_bytes(3 * lv, c0, 3, 1, size, size, c_)
        n_blocks = int(RNG.integers(1, 5))
        body = blob
        x = bool_conv_ref(x, wb, thr, 3, 1)
        ch, h = c0, size
        for i in range(n_blocks):
            mode = int(RNG.integers(0, 3)) if h >= 4 else 0       # 0 identity, 1 stride 2 + pooled 1x1, 2 channel change 1x1
            cout = ch if mode == 0 else int(RNG.choice([8, 16, 32, 40, 64]))
            s = 2 if mode == 1 else 1
            blob, wb, thr = conv_bytes(ch, cout, 3, s, h, h, c_ + i + 1)
            y = bool_conv_ref(x, wb, thr, 3, s)
            body += i32(1 if mode else 0, 1 if mode == 1 else 0) + blob
            if mode:
                src = x.reshape(h // 2, 2, h // 2, 2, ch).max((1, 3)) if mode == 1 else x
                hs = h // 2 if mode == 1 else h
                blob2, wb2, thr2 = conv_bytes(ch, cout, 1, 1, hs, hs, c_ + 7 * i + 3)
                body += blob2
                a = bool_conv_ref(src, wb2, thr2, 1, 1)
            else:
                a = x
            x = np.maximum(y, a)
            ch, h = cout, y.shape[0]
        pp = (ch + 31) // 32
        fwords = h * h * pp
        rw = -(-fwords // 4) * 4
        fbits = RNG.integers(0, 2, (10, h, h, ch))
        fw = np.zeros((10, rw), dtype=np.uint32)
        for o in range(10):
            fw[o, :fwords] = hwc_words(fbits[o], pp)
        cnt = fbits.reshape(10, -1).sum(1)
        scale = RNG.integers(1, 1 << 16, 10)
        bias = RNG.integers(-(1 << 20), 1 << 20, 10)
        raw = (2 * fbits.reshape(10, -1) - 1) @ (2 * x.reshape(-1) - 1)
        exp += [int(np.argmax(scale.astype(np.int64) * raw + bias))] * 2   # AND form, then XOR form
        data += i32(size, size, lv, n_blocks, 0) + img.astype(np.uint8).tobytes() + body + i32(10, fwords, rw, h * h * ch) \
            + fw.tobytes() + cnt.astype(np.int32).tobytes() + scale.astype(np.int32).tobytes() + bias.astype(np.int32).tobytes()
    got = run(exe, data)
    bad = np.nonzero(got != np.array(exp))[0]
    return n, len(bad), (f"network {bad[0]}: got class {got[bad[0]]}, expected {exp[bad[0]]}" if len(bad) else "")


def test_hal_xor_rows(hal, n):
    """symbex_xor_popcount_rows, and the *_rows_n forms (AND and XOR over the first n words of rows `stride` apart)
    against numpy: aligned rows of 4k words (the PIE path on the chip), any stride; symbex_hal_prefers_xor is 0 or 1"""
    exe = build("xorrows", """
int main(void) { int a[3]; static uint32_t x[256] __attribute__((aligned(16))), w[64 * 256] __attribute__((aligned(16)));
  static int32_t out[64];
  int pref = symbex_hal_prefers_xor(); if (pref != 0 && pref != 1) return 3;
  while (fread(a, 4, 3, stdin) == 3) {
    int st = a[0], rows = a[1], nw = a[2];
    if (fread(x, 4, st, stdin) != (size_t)st || fread(w, 4, rows * st, stdin) != (size_t)(rows * st)) return 1;
    symbex_xor_popcount_rows(x, w, st, rows, out);
    for (int r = 0; r < rows; r++) printf("%d ", out[r]);
    symbex_xor_popcount_rows_n(x, w, nw, st, rows, out);
    for (int r = 0; r < rows; r++) printf("%d ", out[r]);
    symbex_and_popcount_rows_n(x, w, nw, st, rows, out);
    for (int r = 0; r < rows; r++) printf("%d ", out[r]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c in range(n):
        st = int(RNG.choice([4, 8, 12, 16, 36, 64, 128, 256])) if c % 2 else min(pick(1, 256, c), 256)
        rows = int(RNG.integers(1, 65))
        x = RNG.integers(0, 2 ** 32, st, dtype=np.uint64).astype(np.uint32)
        w = RNG.integers(0, 2 ** 32, (rows, st), dtype=np.uint64).astype(np.uint32)
        if c % 7 == 0:
            x[:] = 0xFFFFFFFF if c % 2 else 0
        nw = int(RNG.integers(1, st + 1))
        n4 = min(-(-nw // 4) * 4, st)
        x[nw:n4] = 0                                     # the words up to the next multiple of 4: 0 on both sides
        w[:, nw:n4] = 0
        exp.append(popcount(x[None, :] ^ w).sum(1).astype(np.int64))
        exp.append(popcount(x[None, :nw] ^ w[:, :nw]).sum(1).astype(np.int64))
        exp.append(popcount(x[None, :nw] & w[:, :nw]).sum(1).astype(np.int64))
        data += i32(st, rows, nw) + x.tobytes() + w.tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_bool_cell_head(hal, n):
    """symbex_bool_cell_head (the class of every cell, FOMO-style) and symbex_bool_grid_checksum against numpy:
    any channel count (bits above it are 0), ties (the first class wins)"""
    exe = build("cellhead", """
#include "layers/conv/symbex_bool.h"
int main(void) { int a[5]; static uint32_t m[24 * 24 * 4], w[16 * 8] __attribute__((aligned(16)));
  static int32_t sc[16], bi[16], scr[16]; static uint8_t out[24 * 24];
  while (fread(a, 4, 5, stdin) == 5) {
    int h = a[0], wd = a[1], ch = a[2], nc = a[3], rw = a[4], pp = (ch + 31) / 32;
    if (fread(m, 4, h * wd * pp, stdin) != (size_t)(h * wd * pp) || fread(w, 4, nc * rw, stdin) != (size_t)(nc * rw) ||
        fread(sc, 4, nc, stdin) != (size_t)nc || fread(bi, 4, nc, stdin) != (size_t)nc) return 1;
    symbex_bool_cell_head(m, (uint16_t)h, (uint16_t)wd, (uint16_t)ch, w, (uint16_t)rw, (uint16_t)nc, sc, bi, scr, out);
    for (int i = 0; i < h * wd; i++) printf("%d ", out[i]);
    printf("%d\\n", symbex_bool_grid_checksum(out, (uint32_t)(h * wd))); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c in range(n):
        h, w = int(RNG.integers(1, 25)), int(RNG.integers(1, 25))
        ch = int(RNG.choice([8, 16, 31, 32, 33, 64, 100, 128]))
        nc = int(RNG.integers(2, 17))
        pp = (ch + 31) // 32
        rw = -(-pp // 4) * 4
        cells = bits_data((h, w, ch), c)
        wb = RNG.integers(0, 2, (nc, ch))
        rows = np.zeros((nc, rw), dtype=np.uint32)
        rows[:, :pp] = words(wb, pp)
        scale = RNG.integers(1, 1 << 16, nc)
        bias = RNG.integers(-(1 << 20), 1 << 20, nc)
        if c % 5 == 0:
            scale[:] = scale[0]
            bias[:] = bias[0]                                   # ties: the first class wins
        raw = (2 * cells.reshape(-1, ch) - 1) @ (2 * wb - 1).T  # (cells, classes)
        cls = np.argmax(scale.astype(np.int64)[None] * raw + bias[None], axis=1)
        hsh = 2166136261
        for v in cls:
            hsh = ((hsh ^ int(v)) * 16777619) & 0xFFFFFFFF
        exp.append(np.concatenate([cls, [hsh & 0x7FFFFFFF]]).astype(np.int64))
        data += i32(h, w, ch, nc, rw) + hwc_words(cells, pp).astype(np.uint32).tobytes() + rows.tobytes() \
            + scale.astype(np.int32).tobytes() + bias.astype(np.int32).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def bool1d_case(c, u8):
    """one random 1D layer: (sizes, input, weights, thresholds, expected output words)"""
    k = pick(1, 16, c)
    ch = pick(1, 41, c) if not u8 else pick(1, 13, c)
    o = pick(1, 41, c)
    s = int(RNG.integers(1, 4))
    p = int(RNG.integers(1, 5))
    n_pre_min = p                                            # at least one pooled output
    length = k + s * (n_pre_min - 1) + pick(0, 60, c)
    n_pre = (length - k) // s + 1
    out_len = n_pre // p
    n = k * ch
    rw = -(-(-(-n // 32)) // 4) * 4
    wb = RNG.integers(0, 2, (o, n))
    rows = np.zeros((o, rw), dtype=np.uint32)
    rows[:, :-(-n // 32)] = words(wb, -(-n // 32))
    x = u8_data(length * ch, c).reshape(length, ch) if u8 else bits_data((length, ch), c)
    win = np.stack([x[t * s:t * s + k].ravel() for t in range(out_len * p)])          # (steps, n)
    raw = win @ (2 * wb - 1).T if u8 else (2 * win - 1) @ (2 * wb - 1).T
    spread = 255 * n // 4 if u8 else n // 2
    thr = RNG.integers(-spread - 1, spread + 2, o)
    bits = (raw >= thr[None]).astype(np.int64).reshape(out_len, p, o).max(axis=1)        # OR pool
    out_words = -(-out_len * o // 32) + 1
    exp = words(bits.ravel(), out_words).astype(np.int64)
    in_data = (x.astype(np.uint8).tobytes() if u8 else
               words(x.ravel(), -(-length * ch // 32) + 1).astype(np.uint32).tobytes())
    return (length, ch, o, k, s, p, out_len, rw), in_data, rows, wb.sum(1), thr, exp


def test_bool1d(hal, n):
    """symbex_bool1d_conv (binary input) and symbex_bool1d_conv_u8 (uint8 input) against numpy: any kernel, stride,
    channel count (not only powers of 2), OR pool fused, windows at every bit offset of the stream; AND and XOR forms"""
    exe = build("bool1d", """
#include "layers/conv/symbex_bool.h"
#include "layers/conv/symbex_bool1d.h"
int main(void) { int a[10]; static uint32_t in[8192], out[8192], w[200 * 320] __attribute__((aligned(16)));
  static uint32_t win[640] __attribute__((aligned(16))); static uint8_t ws[320 * 32] __attribute__((aligned(16)));
  static uint8_t in8[32768]; static int32_t thr[256], cnt[256], scr[256];
  while (fread(a, 4, 10, stdin) == 10) {
    int u8 = a[0], form = a[1], len = a[2], ch = a[3], o = a[4], k = a[5], s = a[6], p = a[7], ol = a[8], rw = a[9];
    int in_w = (len * ch + 31) / 32 + 1, out_w = (ol * o + 31) / 32 + 1;
    if (u8) { if (fread(in8, 1, (size_t)(len * ch), stdin) != (size_t)(len * ch)) return 1; }
    else if (fread(in, 4, (size_t)in_w, stdin) != (size_t)in_w) return 1;
    if (fread(w, 4, (size_t)(o * rw), stdin) != (size_t)(o * rw) || fread(cnt, 4, (size_t)o, stdin) != (size_t)o ||
        fread(thr, 4, (size_t)o, stdin) != (size_t)o) return 1;
    struct SymbexBool1D L = {(uint16_t)ch, (uint16_t)o, (uint16_t)k, (uint16_t)s, (uint16_t)p, (uint16_t)len,
                             (uint16_t)ol, (uint16_t)rw, w, thr, cnt};
    symbex_bool_force_mode = form;
    for (int i = 0; i < out_w; i++) out[i] = 0xDEADBEEFu;          /* the layer must overwrite every word */
    if (u8) symbex_bool1d_conv_u8(&L, in8, out, ws, scr); else symbex_bool1d_conv(&L, in, out, win, scr);
    for (int i = 0; i < out_w; i++) printf("%u ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = Chunks(), []
    for c in range(n):
        for u8 in (0, 1):
            (length, ch, o, k, s, p, ol, rw), in_data, rows, cnt, thr, e = bool1d_case(c, u8)
            for form in ((0,) if u8 else (1, 2)):
                exp.append(e)
                data += i32(u8, form, length, ch, o, k, s, p, ol, rw) + in_data + rows.tobytes() \
                    + cnt.astype(np.int32).tobytes() + thr.astype(np.int32).tobytes()
    got = run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


TESTS = [("hal: popcount32", test_hal_popcount32), ("hal: name", test_hal_name),
         ("hal: xnor_popcount (real bits only)", test_hal_xnor_popcount),
         ("hal: and_popcount", test_hal_and_popcount), ("hal: u8_prepare + u8_sum_bits", test_hal_u8_sum_bits),
         ("hal: *_rows (xnor, and, u8 padded / unpadded, and_planes)", test_hal_rows),
         ("ops: and_raw + and_input_u8", test_ops_and), ("ops: extract / insert / popcount_and", test_ops_bitops),
         ("ops: load_line_buffer", test_ops_load_line_buffer), ("ops: bitpack_from_scores", test_ops_bitpack),
         ("ops: dsp apply_layer / binarize / apply_downsample (fixed point)", test_dsp),
         ("dense (k=1): forward, argmax, input_u8 x3, threshold, *_bits", test_dense_k1),
         ("dense: input_u8_planes_bits (k = 1..8 input bits)", test_dense_planes_k),
         ("dense: binary_classifier_argmax (int64 logits, ties)", test_binary_classifier),
         ("network_dense: chains of dense layers", test_network_dense),
         ("conv: bpr rgb_to_24bits", test_bpr), ("conv: binarize_image (float CHW -> HWC bits)", test_binarize_image),
         ("conv: binary_pool_2x2 (majority of 4, HWC)", test_binary_pool),
         ("conv: extract_window (patch, padding mask, tail)", test_extract_window),
         ("network_conv: residual blocks, with and without downsample", test_network_conv),
         ("network_conv: add_block accepts valid blocks, rejects the rest", test_network_conv_limits),
         ("bool: conv with integer thresholds, AND and XOR forms (borders, strides, 1x1 / 3x3)", test_bool_conv),
         ("bool: thermometer, OR pool, OR, integer classifier", test_bool_pieces),
         ("bool: whole random networks, AND and XOR forms (symbex_bool_net_run)", test_bool_net),
         ("hal: xor_popcount_rows, *_rows_n (AND / XOR over the first n words)", test_hal_xor_rows),
         ("bool: per-cell head (FOMO) + grid checksum", test_bool_cell_head),
         ("bool1d: 1D conv + fused OR pool, binary and uint8 input, AND and XOR forms", test_bool1d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400, help="random cases per test (some tests multiply it)")
    ap.add_argument("--only", default="", help="run only the tests whose name matches this regular expression "
                                               "(e.g. 'dense' or 'hal|ops')")
    ap.add_argument("--stress", action="store_true",
                    help="sanitizers (out-of-bounds, undefined behavior), every boundary size, extreme data, larger sizes")
    args = ap.parse_args()
    global STRESS
    STRESS = args.stress
    if STRESS and args.cases == 400:
        args.cases = 2000
    print(f"[*] lib_symbex tests | library: {LIB} | {args.cases} cases per test"
          + (" | STRESS: AddressSanitizer + UBSan, boundary sizes, extreme data, sizes x4" if STRESS else "") + "\n")
    fails = 0
    t_all = time.time()
    for hal in HALS:
        print(f"  HAL {hal}")
        print(f"  {'test':<66}{'cases':>8}{'failures':>10}{'time s':>9}{'ms/case':>9}   result")
        print("  " + "-" * 114)
        t_hal = time.time()
        for name, fn in TESTS:
            if args.only and not re.search(args.only, name):
                continue
            t0 = time.time()
            try:
                cases, bad, what = fn(hal, args.cases)
                status = "OK" if bad == 0 else f"[X] {what}"
            except RuntimeError as e:
                cases, bad, status = 0, 1, "[X] " + str(e).splitlines()[0] + ("\n" + "\n".join(
                    "        " + l for l in str(e).splitlines()[1:12]) if "\n" in str(e) else "")
            dt = time.time() - t0
            fails += bad > 0
            per = f"{1000 * dt / cases:>9.2f}" if cases else f"{'-':>9}"
            print(f"  {name:<66}{cases:>8}{bad:>10}{dt:>9.1f}{per}   {status}")
        print(f"  {'(this HAL, total; includes compiling every test program)':<84}{time.time() - t_hal:>9.1f}\n")
    print(("[*] ALL OK" if fails == 0 else f"[X] {fails} test(s) failed") + f"  ({time.time() - t_all:.1f} s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
