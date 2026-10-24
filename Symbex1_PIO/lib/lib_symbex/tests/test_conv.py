"""
test_conv.py -- tests of the convolutional path of lib_symbex, added on top of run_tests.py (same folder).

    cd lib_symbex && python tests/test_conv.py                 # every test: run_tests.py's and these
    python tests/test_conv.py --only conv                      # only these
    python tests/test_conv.py --only conv --stress             # these, with sanitizers and boundary sizes

Same method as run_tests.py (it reuses its helpers and options): every function is compiled with gcc against
each HAL and compared with an exact numpy reference on random cases.

Also: the whole-layer HAL primitives (symbex_*_rows) against numpy.

Covered here (binary maps in HWC format: pixel (y, x) starts at word (y * W + x) * PW, PW = ceil(C / 32),
channel c is bit c % 32 of word c / 32 of that pixel):
    bpr: rgb_to_24bits          RGB uint8 (HWC) -> one word per pixel, bit (color * 8 + j) = bit (7 - j) of color
    conv: binarize_image        float CHW -> HWC bits, bit = value >= 0
    conv: binary_pool_2x2       output bit 1 when at least 2 of the 4 input bits are 1 (real channels only)
    conv: extract_window        patch bit (kh * K + kw) * C + c = input bit, 0 at padding; mask 1 exactly at
                                padding; patch tail up to a multiple of 4 words set to 1, mask 0 there
    dsp: apply_layer / binarize / apply_downsample
                                the fixed-point formula of ops/symbex_mac_1bit.c (round-to-nearest shifts, both
                                RPReLU variants, log2_beta of either sign), against Python integers
    network_conv: run_ext       whole networks of residual blocks, with and without downsample, against a numpy
                                re-implementation of the same chain (binarize -> conv with exact zero padding ->
                                DSP + identity; downsample: binarize -> 2x2 pool -> 3x3 conv -> affine BN). This also
                                covers extract_window + conv2d_process_position (padding correction) inside a network.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tests as rt                                   # noqa: E402

for s in ["src/ops/symbex_mac_1bit.c", "src/layers/conv/symbex_bpr.c", "src/layers/conv/symbex_conv2d.c",
          "src/layers/conv/symbex_binary_pool.c", "src/core/network/symbex_network_conv.c"]:
    if s not in rt.SOURCES:
        rt.SOURCES.append(s)
RNG = rt.RNG


def hwc_words(bits_hwc, pw):
    """(h, w, c) bits -> (h * w * pw,) uint32 HWC words."""
    h, w, c = bits_hwc.shape
    padded = np.zeros((h, w, pw * 32), dtype=np.int64)
    padded[:, :, :c] = bits_hwc
    return rt.words(padded.reshape(h * w, pw * 32), pw).ravel()


def test_bpr(hal, n):
    exe = rt.build("bpr", """
int main(void) { int h, w; static uint8_t img[64 * 64 * 3]; static uint32_t out[64 * 64];
  while (fread(&h, 4, 1, stdin) == 1 && fread(&w, 4, 1, stdin) == 1 && fread(img, 1, h * w * 3, stdin) == (size_t)(h * w * 3)) {
    symbex_rgb_to_24bits(img, out, (uint16_t)h, (uint16_t)w);
    for (int i = 0; i < h * w; i++) printf("%u ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = rt.Chunks(), []
    for c in range(n):
        h, w = int(RNG.integers(1, 17)), int(RNG.integers(1, 17))
        img = rt.u8_data(h * w * 3, c).reshape(h, w, 3)
        word = np.zeros(h * w, dtype=np.int64)
        for color in range(3):
            for j in range(8):
                word |= ((img[:, :, color].ravel() >> (7 - j)) & 1) << (color * 8 + j)
        exp.append(word)
        data += rt.i32(h, w) + img.astype(np.uint8).tobytes()
    got = rt.run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"word {bad[0]}: got 0x{got[bad[0]]:06X}, expected 0x{flat[bad[0]]:06X}" if len(bad) else "")


def test_binarize_image(hal, n):
    exe = rt.build("binimg", """
int main(void) { int c, h, w; static float x[96 * 16 * 16]; static uint32_t out[16 * 16 * 3];
  while (fread(&c, 4, 1, stdin) == 1 && fread(&h, 4, 1, stdin) == 1 && fread(&w, 4, 1, stdin) == 1
         && fread(x, 4, c * h * w, stdin) == (size_t)(c * h * w)) {
    int pw = (c + 31) / 32;
    memset(out, 0xAB, sizeof out);
    symbex_binarize_image(x, out, (uint16_t)c, (uint16_t)h, (uint16_t)w);
    for (int i = 0; i < h * w * pw; i++) printf("%u ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = rt.Chunks(), []
    for k in range(n):
        ch, h, w = min(rt.pick(1, 97, k), 96), int(RNG.integers(1, 17)), int(RNG.integers(1, 17))
        x = RNG.normal(size=(ch, h, w)).astype(np.float32)
        x[RNG.random((ch, h, w)) < 0.1] = 0.0                     # exact zeros: >= 0 -> bit 1
        x[RNG.random((ch, h, w)) < 0.05] = -0.0
        bits = (x >= 0).astype(np.int64).transpose(1, 2, 0)       # HWC
        exp.append(hwc_words(bits, (ch + 31) // 32).astype(np.int64))
        data += rt.i32(ch, h, w) + x.tobytes()
    got = rt.run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"word {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_binary_pool(hal, n):
    exe = rt.build("pool", """
int main(void) { int c, h, w; static uint32_t inp[32 * 32 * 4], out[16 * 16 * 4];
  while (fread(&c, 4, 1, stdin) == 1 && fread(&h, 4, 1, stdin) == 1 && fread(&w, 4, 1, stdin) == 1) {
    int pw = (c + 31) / 32;
    if (fread(inp, 4, h * w * pw, stdin) != (size_t)(h * w * pw)) return 1;
    memset(out, 0xAB, sizeof out);
    symbex_binary_pool_2x2(inp, out, (uint16_t)c, (uint16_t)h, (uint16_t)w);
    for (int i = 0; i < (h / 2) * (w / 2) * pw; i++) printf("%u ", out[i]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp_bits, total = rt.Chunks(), [], 0
    for k in range(n):
        ch, h, w = min(rt.pick(1, 129, k), 128), int(RNG.integers(2, 33)), int(RNG.integers(2, 33))
        pw = (ch + 31) // 32
        bits = rt.bits_data((h, w, ch), k)
        words_in = hwc_words(bits, pw)
        if ch % 32:                                              # garbage in the unused channel bits
            garbage = RNG.integers(0, 2 ** 32, h * w, dtype=np.uint64).astype(np.uint32)
            words_in.reshape(h * w, pw)[:, -1] |= (garbage >> (ch % 32)) << (ch % 32)
        oh, ow = h // 2, w // 2
        blocks = bits[:2 * oh, :2 * ow].reshape(oh, 2, ow, 2, ch).sum((1, 3))
        exp_bits.append(((blocks >= 2).astype(np.int64), pw, ch))
        data += rt.i32(ch, h, w) + words_in.astype(np.uint32).tobytes()
        total += 1
    got = rt.run(exe, data)
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
    exe = rt.build("window", """
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
    data, exp, total = rt.Chunks(), [], 0
    for kcase in range(n):
        ch = min(rt.pick(1, 97, kcase), 160)
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
        bits = rt.bits_data((h, w, ch), kcase)
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
        exp.append(np.concatenate([rt.words(pb, patch_words), rt.words(mb, patch_words)]).astype(np.int64))
        data += rt.i32(ch, h, w, k, s, p, oh, ow, patch_words) + m.astype(np.uint32).tobytes()
        total += 1
    got = rt.run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return total, len(bad), (f"word {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


# ---------------------------------------------------------------------------------------------------
# DSP (fixed point) and whole convolutional networks
# ---------------------------------------------------------------------------------------------------
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
                            p["slope"]]).astype(np.int32).tobytes() + rt.i32(int(p["use_shift"])))


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
    exe = rt.build("dsp", C_READ_DSP + """
int main(void) { int c, qa, qm;
  while (fread(&c, 4, 1, stdin) == 1 && fread(&qa, 4, 1, stdin) == 1 && fread(&qm, 4, 1, stdin) == 1) {
    n32 = n8 = 0; SymbexDspParams d; read_dsp(&d, c, qa, qm);
    const int32_t* raw = take32(c); const int32_t* id = take32(c); const int32_t* beta = take32(c);
    SymbexDspParamsSimple s = { d.scale, d.shift_bn, (uint8_t)qa, (uint8_t)qm };
    for (int i = 0; i < c; i++) printf("%d %d %d ", symbex_dsp_apply_layer(raw[i], id[i], (uint16_t)i, &d),
                                       (int)symbex_dsp_binarize(raw[i], beta[i]), symbex_dsp_apply_downsample(raw[i], (uint16_t)i, &s));
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = rt.Chunks(), []
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
        data += rt.i32(c, qa, qm) + dsp_bytes(p) + np.concatenate([raw, ident, beta]).astype(np.int32).tobytes()
    got = rt.run(exe, data)
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
    return rt.words(flat, pwds).ravel()


def test_network_conv(hal, n):
    exe = rt.build("netconv", C_READ_DSP + """
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
    data, exp = rt.Chunks(), []
    n = min(n, 120) if not rt.STRESS else min(n, 300)    # whole networks: a few hundred are plenty
    for k in range(n):
        qa = int(RNG.integers(6, 11))
        qm = int(RNG.integers(qa + 4, 22))
        c, h, w = int(RNG.choice([1, 3, 8, 16, 24, 33])), int(RNG.choice([4, 6, 8])), int(RNG.choice([4, 6, 8]))
        nb = int(RNG.integers(1, 4))
        x = RNG.integers(-(1 << 11), 1 << 11, (c, h, w))
        hdr = rt.i32(nb, c, h, w, qa, qm) + x.astype(np.int32).tobytes()
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
            blocks_b = rt.i32(ds, oc, st) + dsp_bytes(p1) + conv_weight_words(w1).astype(np.uint32).tobytes() \
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
    got = rt.run(exe, data)
    flat = np.concatenate(exp)
    if len(got) != len(flat):
        return n, n, f"printed {len(got)} values, expected {len(flat)} (a network failed to run)"
    bad = np.nonzero(got != flat)[0]
    ends = np.cumsum([len(e) for e in exp])
    nets = sorted({int(np.searchsorted(ends, b, side="right")) for b in bad})          # failing networks
    return n, len(nets), (f"network {nets[0]}, value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


def test_network_conv_limits(hal, n):
    """symbex_network_conv_add_block must accept valid blocks and reject the ones that do not fit the library."""
    exe = rt.build("convlimits", """
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
    data = rt.Chunks()
    for args, _, _ in cases:
        data += rt.i32(*args)
    got = rt.run(exe, data)
    bad = [i for i, (_, e, _) in enumerate(cases) if got[i] != e]
    return len(cases), len(bad), (f"{cases[bad[0]][2]}: add_block returned {got[bad[0]]}, expected {cases[bad[0]][1]}" if bad else "")


def test_hal_rows(hal, n):
    """The whole-layer HAL primitives (*_rows) against numpy: XNOR rows, AND rows, uint8 rows (padded stride and
    unpadded stride: the PIE layout and the C fallback), with garbage in the input past n_in."""
    exe = rt.build("halrows", """
int main(void) { int a4[4]; static uint32_t a[64] __attribute__((aligned(16))), w[64 * 96] __attribute__((aligned(16)));
  static uint8_t x[2048], work[2048] __attribute__((aligned(16))); static int32_t out[96];
  while (fread(a4, 4, 4, stdin) == 4) {
    int n_in = a4[0], n_out = a4[1], words = a4[2], stride = a4[3];
    if (fread(a, 4, stride, stdin) != (size_t)stride || fread(w, 4, n_out * stride, stdin) != (size_t)(n_out * stride)
        || fread(x, 1, n_in, stdin) != (size_t)n_in) return 1;
    static uint32_t wx[64 * 96] __attribute__((aligned(16)));
    for (int r = 0; r < n_out; r++) memcpy(&wx[r * words], &w[r * stride], 4 * words);
    symbex_xnor_popcount_rows(a, wx, words, n_in, n_out, out); for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    symbex_and_popcount_rows(a, w, stride, n_out, out); for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    symbex_u8_prepare(x, n_in, work, stride); symbex_u8_sum_bits_rows(work, w, stride, n_out, out);
    for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    symbex_u8_prepare(x, n_in, work, words); symbex_u8_sum_bits_rows(work, wx, words, n_out, out);
    for (int r = 0; r < n_out; r++) printf("%d ", out[r]);
    printf("\\n"); }
  return 0; }""", hal)
    data, exp = rt.Chunks(), []
    for k in range(n):
        n_in, n_out = min(rt.pick(1, 1300, k), 2000), int(RNG.integers(1, 97))
        words = (n_in + 31) // 32
        stride = -(-words // 4) * 4
        abits, wbits = rt.bits_data(n_in, k), rt.bits_data((n_out, n_in), k + 3)
        x = rt.u8_data(n_in, k)
        a = RNG.integers(0, 2 ** 32, stride, dtype=np.uint64).astype(np.uint32)      # garbage past n_in
        a[:words] = rt.words(abits, words)
        if n_in % 32:
            a[words - 1] |= np.uint32((int(RNG.integers(0, 2 ** 32)) >> (n_in % 32)) << (n_in % 32))
        w = rt.words(wbits, stride)
        sw = 2 * wbits - 1
        exp.append(np.concatenate([sw @ (2 * abits - 1), wbits @ abits, wbits @ x, wbits @ x]))
        data += rt.i32(n_in, n_out, words, stride) + a.tobytes() + w.astype(np.uint32).tobytes() + x.astype(np.uint8).tobytes()
    got = rt.run(exe, data)
    flat = np.concatenate(exp)
    bad = np.nonzero(got != flat)[0]
    return n, len(bad), (f"value {bad[0]}: got {got[bad[0]]}, expected {flat[bad[0]]}" if len(bad) else "")


rt.TESTS += [("hal: *_rows (xnor, and, u8 padded and unpadded)", test_hal_rows)]
rt.TESTS += [("network_conv: add_block accepts valid blocks, rejects what does not fit", test_network_conv_limits)]
rt.TESTS += [("dsp: apply_layer / binarize / apply_downsample (fixed point)", test_dsp),
             ("network_conv: residual blocks, with and without downsample", test_network_conv)]
rt.TESTS += [("bpr: rgb_to_24bits", test_bpr), ("conv: binarize_image (float CHW -> HWC bits)", test_binarize_image),
             ("conv: binary_pool_2x2 (majority of 4, HWC)", test_binary_pool),
             ("conv: extract_window (patch, padding mask, tail)", test_extract_window)]

if __name__ == "__main__":
    rt.main()
