"""
Exports a fully binarized SYMBEX ResNet (method "plain") to the C header the bit-serial engine reads.

What goes into the header, per conv layer:
  - the weights, 1 bit each (>= 0 -> 1), packed LSB-first into 32-bit words,
    each filter padded to a multiple of 4 words (the engine loads 128 bits at a time)
  - BatchNorm folded with the weight scale alpha:  scale = alpha * gamma / sqrt(var + eps)
                                                    shift = beta - mean * gamma / sqrt(var + eps)
    stored as fixed point (scale: q_mult fractional bits, everything else: q_act)
  - the RPReLU parameters and the RSign thresholds of the layer's input, also fixed point

Weight order inside each filter (layout):
  - "hwc" (default, current engine): [kh][kw][ic] -- the channels of each pixel together
  - "chw" (old engine):              [ic][kh][kw]
  The header defines SYMBEX_LAYOUT_HWC when layout="hwc", so the C side can refuse old weights.

scale types:
  - conv and downsample: int32 (the engine's SymbexDspParams / SymbexDspParamsSimple)
  - classifier:          int64 (symbex_clasificador_binario_argmax still takes int64)

This is earlier-version code (the first v3 conv engine). The C names it writes into the header (`<LAYER>_pesos`,
`_escala`, `_beta_rsign_entrada`, `CLASIFICADOR_*`, and the wiring's `bloques[]`, `HACER_CONV`...) are that engine's
interface and stay as they were; only the Python side and the header comments are in English.

The function receives an already built model (it only reads attributes), so it does not import
any model class. It expects the SYMBEX ResNet layout:
  conv1/bn1/rprelu_stem, layer1..layerN (blocks with conv1/bn1/act1/rsign1, conv2/bn2/act2/rsign2,
  optional downsample.rsign/conv/bn), rsign_final, linear, bn_final.
"""
import os
import numpy as np

from .report import print_report


def _q(x, bits):
    """float -> fixed-point integer with `bits` fractional bits."""
    return np.round(x * (2 ** bits)).astype(np.int64)


def _pack_rows(rows, multiple=4):
    """rows: (n_rows, n_weights). Returns (packed words per row, words per row).
    multiple: each row is padded to a multiple of this many 32-bit words (4 for convs: 128-bit PIE loads;
    1 for the classifier, whose C code advances exactly ceil(n / 32) words per class)."""
    n = rows.shape[1]
    words = -(-((n + 31) // 32) // multiple) * multiple
    bits = np.where(rows >= 0, 1, 0)
    packed = []
    for row in bits:
        padded = np.concatenate([row, np.zeros(words * 32 - n, dtype=int)])
        packed.append([int(sum(b << i for i, b in enumerate(padded[k:k + 32])))
                       for k in range(0, words * 32, 32)])
    return packed, words


def _engine_rows(w, layout):
    """(out, in, kh, kw) -> (out, n) in the order the engine reads the window.
    hwc: [kh][kw][ic]  (current engine)    chw: [ic][kh][kw]  (old engine)"""
    if layout == "hwc":
        w = w.transpose(0, 2, 3, 1)
    return w.reshape(w.shape[0], -1)


def _padding(conv):
    """The conv's own integer padding (square), the value the C engine and the simulator must use."""
    p = conv.padding
    if isinstance(p, str):
        raise ValueError(f"[EXPORT]: padding={p!r} is not supported; give the conv an integer padding")
    if isinstance(p, int):
        return p
    if len(set(p)) != 1:
        raise ValueError(f"[EXPORT]: non-square padding {tuple(p)} is not supported by the C engine")
    return int(p[0])


def _fold_bn(bn, alpha, q_act, q_mult):
    """BatchNorm (+ weight scale alpha, if any) -> integer (scale, shift) per channel."""
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.detach().numpy()
    var = bn.running_var.detach().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    shift = beta - mean * scale
    folded = scale if alpha is None else alpha * scale
    return _q(folded, q_mult), _q(shift, q_act)


def _scale_int32(scale, name):
    """The engine stores the scale as int32 (conv and downsample). Checks that it fits."""
    m = int(np.abs(scale).max())
    assert m < 2 ** 31, f"{name}: scale {m} does not fit in int32 (Q_MULT too high)"
    return scale


def _rprelu_params(rprelu, q_act):
    """RPReLU -> integer (gamma, zeta, slope). gamma = bias1, zeta = bias2."""
    gamma = _q(rprelu.bias1.detach().numpy().flatten(), q_act)
    zeta = _q(rprelu.bias2.detach().numpy().flatten(), q_act)
    slope = _q(rprelu.prelu.weight.detach().numpy().flatten(), q_act)
    return gamma, zeta, slope


def _array(f, ctype, name, values, end="\n"):
    f.write(f"const {ctype} {name}[{len(values)}] = {{"
            + ", ".join(str(int(v)) for v in values) + "};" + end)


def _weights(f, name, packed, words, comment, label):
    f.write(f"const uint32_t {name}_pesos[{len(packed) * words}] = {{\n")
    if comment:
        f.write(f"    // {comment}\n")
    for i, row in enumerate(packed):
        f.write("    " + ", ".join(f"0x{v:08X}" for v in row) + f", // {label} {i}\n")
    f.write("};\n\n")


def _row(name, shape, w_shape, words, parts, positions):
    """One report row. `positions` = spatial positions of the layer's output (1 for linear)."""
    out_ch = w_shape[0]
    return dict(name=name, shape=shape, n_weights=int(np.prod(w_shape)), bits=1,
                weights_bytes=out_ch * words * 4, params_bytes=sum(parts.values()), parts=parts,
                map_bytes=out_ch * positions * 4,               # int32 activations
                xnor_words=positions * out_ch * words)


def _write_conv(f, name, filters, scale, shift, gamma, zeta, slope, beta_in,
                in_ch, out_ch, kernel, stride, padding):
    packed, words = _pack_rows(filters)
    f.write(f"#define {name}_IN_CH {in_ch}\n#define {name}_OUT_CH {out_ch}\n")
    f.write(f"#define {name}_KERNEL {kernel}\n#define {name}_STRIDE {stride}\n")
    f.write(f"#define {name}_PADDING {padding}\n#define {name}_WINDOW_ELEMS {filters.shape[1]}\n\n")
    _weights(f, name, packed, words,
             f"Integrity check: {out_ch} channels, {words} words per channel", "Channel")
    _array(f, "int32_t", f"{name}_escala", _scale_int32(scale, name))
    _array(f, "int32_t", f"{name}_shift_bn", shift)
    _array(f, "int32_t", f"{name}_gamma_rprelu", gamma)
    _array(f, "int32_t", f"{name}_zeta", zeta)
    f.write(f"#define {name}_USA_SHIFT 0\n")
    _array(f, "int8_t", f"{name}_log2_beta", [0] * out_ch)
    _array(f, "int8_t", f"{name}_signo_beta", [1] * out_ch)
    _array(f, "int32_t", f"{name}_slope", slope)
    _array(f, "int32_t", f"{name}_beta_rsign_entrada", beta_in, end="\n\n")
    return words


def _export_conv(f, name, conv, bn, act, rsign, stride, positions, q_act, q_mult, layout):
    """conv + bn + RPReLU (+ the RSign that binarizes its input; None for the stem)."""
    w = conv.weight.detach().numpy()
    scale, shift = _fold_bn(bn, np.abs(w).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma, zeta, slope = _rprelu_params(act, q_act)
    beta_in = np.zeros(w.shape[1]) if rsign is None else _q(rsign.beta.detach().numpy().flatten(), q_act)
    words = _write_conv(f, name, _engine_rows(w, layout), scale, shift, gamma, zeta, slope, beta_in,
                        w.shape[1], w.shape[0], w.shape[2], stride, _padding(conv))
    out, inp = w.shape[0], w.shape[1]
    parts = {"scale (int32)": 4 * out, "shift_bn (int32)": 4 * out,
             "rprelu (gamma, zeta, slope, log2, sign)": 14 * out, "thresholds (RSign, int32)": 4 * inp}
    return _row(name, f"{inp}->{out} k{w.shape[2]} s{stride}", w.shape, words, parts, positions)


def _export_downsample(f, name, ds, positions, q_act, q_mult, layout):
    """RSign + pooling + conv + bn (no activation: the block adds it to the shortcut)."""
    w = ds.conv.weight.detach().numpy()
    scale, shift = _fold_bn(ds.bn, np.abs(w).mean(axis=(1, 2, 3)), q_act, q_mult)
    beta_in = _q(ds.rsign.beta.detach().numpy().flatten(), q_act)
    packed, words = _pack_rows(_engine_rows(w, layout))
    f.write(f"#define {name}_IN_CH {w.shape[1]}\n#define {name}_OUT_CH {w.shape[0]}\n")
    f.write(f"#define {name}_KERNEL {w.shape[2]}\n#define {name}_PADDING {_padding(ds.conv)}\n")
    f.write(f"#define {name}_WINDOW_ELEMS {w.shape[1] * w.shape[2] * w.shape[3]}\n\n")
    _weights(f, name, packed, words,
             f"Downsample check: {w.shape[0]} output channels, {words} words per channel", "Channel")
    _array(f, "int32_t", f"{name}_escala", _scale_int32(scale, name))
    _array(f, "int32_t", f"{name}_shift_bn", shift)
    _array(f, "int32_t", f"{name}_beta_rsign_entrada", beta_in, end="\n\n")
    out, inp = w.shape[0], w.shape[1]
    parts = {"scale (int32)": 4 * out, "shift_bn (int32)": 4 * out, "thresholds (RSign, int32)": 4 * inp}
    return _row(name, f"{inp}->{out} k{w.shape[2]} pool", w.shape, words, parts, positions)


def _export_classifier(f, model, q_act, q_mult):
    """The classifier does NOT change: its scale stays int64 (symbex_dense takes it that way)."""
    w = model.linear.weight.detach().numpy()
    scale, shift = _fold_bn(model.bn_final, None, q_act, q_mult)    # no alpha: LinearBinaria has none
    beta_in = _q(model.rsign_final.beta.detach().numpy().flatten(), q_act)
    packed, words = _pack_rows(w, multiple=1)          # symbex_dense.c advances ceil(n / 32) words per class
    f.write(f"#define CLASIFICADOR_IN_FEATURES {w.shape[1]}\n#define CLASIFICADOR_OUT_FEATURES {w.shape[0]}\n\n")
    _weights(f, "CLASIFICADOR", packed, words, None, "Class")
    _array(f, "int64_t", "CLASIFICADOR_escala", scale)
    _array(f, "int32_t", "CLASIFICADOR_shift_bn", shift)
    _array(f, "int32_t", "CLASIFICADOR_beta_rsign_entrada", beta_in, end="\n\n")
    out, inp = w.shape
    parts = {"scale (int64)": 8 * out, "shift_bn (int32)": 4 * out, "thresholds (RSign, int32)": 4 * inp}
    return _row("CLASIFICADOR", f"{inp}->{out}", w.shape, words, parts, 1)


def export_resnet_dsp(model, out_path, q_act=16, q_mult=20, input_size=32, detail=False, layout="hwc"):
    """Writes the header for `model`, prints the input/output report, returns the rows.
    input_size: side of the (square) input image; the stem keeps it, every later stage halves it.
    layout: "hwc" (current engine) or "chw" (old engine) -- order of the weights in each filter."""
    assert layout in ("hwc", "chw"), layout
    assert hasattr(model, "_nombres_capas"), "model is not a SYMBEX ResNet (no _nombres_capas)"
    guard = os.path.basename(out_path).replace(".", "_").upper()
    rows = []
    res = input_size

    res_final = input_size // 2 ** (len(model._nombres_capas) - 1)    # the stem keeps the size, each later stage halves it
    last_ch = getattr(model, model._nombres_capas[-1])[-1].conv2.weight.shape[0]
    if model.linear.weight.shape[1] != last_ch * res_final * res_final:
        raise ValueError(f"[EXPORT]: input_size={input_size} does not match the classifier: it expects "
                         f"{model.linear.weight.shape[1]} features, the last stage gives "
                         f"{last_ch * res_final * res_final}")

    with open(out_path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
        f.write(f"#define SYMBEX_Q_ACT {q_act}\n#define SYMBEX_Q_MULT {q_mult}\n\n")
        if layout == "hwc":
            f.write("#define SYMBEX_LAYOUT_HWC 1\n\n")

        rows.append(_export_conv(f, "STEM_CONV1", model.conv1, model.bn1, model.rprelu_stem, None, 1,
                                 res * res, q_act, q_mult, layout))

        for s, stage in enumerate(model._nombres_capas):
            if s > 0:
                res //= 2
            for i, block in enumerate(getattr(model, stage)):
                name = f"{stage.upper()}_{i}"
                stride = block.conv1.stride
                stride = stride if isinstance(stride, int) else stride[0]
                rows.append(_export_conv(f, f"{name}_CONV1", block.conv1, block.bn1, block.act1,
                                         block.rsign1, stride, res * res, q_act, q_mult, layout))
                rows.append(_export_conv(f, f"{name}_CONV2", block.conv2, block.bn2, block.act2,
                                         block.rsign2, 1, res * res, q_act, q_mult, layout))
                if block.necesita_downsample:
                    rows.append(_export_downsample(f, f"{name}_DOWNSAMPLE", block.downsample,
                                                   res * res, q_act, q_mult, layout))

        rows.append(_export_classifier(f, model, q_act, q_mult))
        f.write(f"#endif // {guard}\n")

    print_report(model, rows, out_path, detail=detail)
    return rows


# =============================================================================
# THE C WIRING -- separate from the functions above (which only write weights
# and parameters). This writes the .h the first v3 engine's main.cpp needed to
# assemble the blocks: the struct SymbexResidualBlock declarations, the body of
# symbex_preparar_bloques() and the bloques[] array -- for N blocks, not the
# 8 that main.cpp had written by hand.
#
# Earlier-version interface: it uses that main.cpp's HACER_CONV macro and its
# struct fields (downsample_pesos...); the current library's SymbexResidualBlock
# names the field downsample_weights, so this header is for the archived engine.
# =============================================================================

def write_wiring_header(model, out_path, input_size=32):
    """
    Writes a .h with the block declarations + symbex_preparar_bloques()
    + the global bloques[] array (so main.cpp uses it directly instead of
    declaring its own local variables), for however many stages / blocks
    `model` has -- not fixed at 4/8. Returns the number of blocks.

    main.cpp needs to #include this file and use the globals
    `bloques`, `n_bloques`, `nombres_bloques` instead of declaring its own.
    """
    assert hasattr(model, "_nombres_capas"), "model is not a SYMBEX ResNet (no _nombres_capas)"

    blocks_info = []   # (name, has_downsample, conv1_input_res, conv2_res, conv2_out_ch)
    res = input_size
    for s, stage in enumerate(model._nombres_capas):
        if s > 0:
            res //= 2
        for i, block in enumerate(getattr(model, stage)):
            name = f"{stage.upper()}_{i}"
            stride = block.conv1.stride
            stride = stride if isinstance(stride, int) else stride[0]
            conv1_input_res = res * stride         # HACER_CONV takes the INPUT resolution, not the output
            conv2_out_ch = block.conv2.weight.shape[0]
            blocks_info.append((name, block.necesita_downsample, conv1_input_res, res, conv2_out_ch))

    with open(out_path, "w") as f:
        f.write("// Generated by compiler_core/export/plain.py:write_wiring_header() -- do not edit by hand.\n")
        f.write("// main.cpp must #include this file and use bloques[] / n_bloques / nombres_bloques[]\n")
        f.write("// instead of declaring its own.\n\n")
        f.write(f"#define RES_STEM {input_size}\n\n")

        for name, has_ds, res_in1, res_out, _ in blocks_info:
            var = f"bloque_{name.lower()}"
            f.write(f"struct SymbexResidualBlock {var} = {{ "
                    f"HACER_CONV({name}_CONV1, {res_in1}, {res_in1}), "
                    f"HACER_CONV({name}_CONV2, {res_out}, {res_out}), "
                    f"NULL, NULL, {'true' if has_ds else 'false'} }};\n")

        f.write("\nvoid symbex_preparar_bloques() {\n")
        for name, has_ds, _, _, _ in blocks_info:
            var = f"bloque_{name.lower()}"
            f.write(f"    {var}.beta_rsign1 = {name}_CONV1_beta_rsign_entrada;\n")
            f.write(f"    {var}.beta_rsign2 = {name}_CONV2_beta_rsign_entrada;\n")
            if has_ds:
                ds_var = f"ds_{name.lower()}"
                f.write(f"    static const SymbexDspParamsSimple {ds_var} = {{ {name}_DOWNSAMPLE_escala, "
                        f"{name}_DOWNSAMPLE_shift_bn, SYMBEX_Q_ACT, SYMBEX_Q_MULT }};\n")
                f.write(f"    {var}.downsample_pesos = {name}_DOWNSAMPLE_pesos;\n")
                f.write(f"    {var}.beta_rsign_downsample = {name}_DOWNSAMPLE_beta_rsign_entrada;\n")
                f.write(f"    {var}.downsample_dsp = {ds_var};\n")
                f.write(f"    {var}.downsample_kernel = {name}_DOWNSAMPLE_KERNEL;\n")
                f.write(f"    {var}.downsample_padding = {name}_DOWNSAMPLE_PADDING;\n")
            f.write("\n")
        f.write("}\n\n")

        f.write("struct SymbexBloqueEntry { struct SymbexResidualBlock* bloque; uint32_t n_salida; };\n")
        f.write("struct SymbexBloqueEntry bloques[] = {\n")
        for name, _, _, res_out, out_ch in blocks_info:
            var = f"bloque_{name.lower()}"
            f.write(f"    {{&{var}, (uint32_t){name}_CONV2_OUT_CH * {res_out} * {res_out}}},\n")
        f.write("};\n")
        f.write(f"const int n_bloques = {len(blocks_info)};\n")
        f.write("const char* nombres_bloques[] = {" +
                ", ".join(f'"{n.lower()}"' for n, _, _, _, _ in blocks_info) + "};\n\n")

        # The REAL output size of the last block -- buffer_final/bits_finales in
        # main.cpp must be sized with this, NOT with a fixed number. It used to
        # be hardcoded to 64*4*4 (the 4-stage case); with fewer stages the final
        # resolution is larger and that fixed size overflows the array --
        # confirmed with 3 stages (March 15, memory corruption).
        _, _, _, last_res, last_channels = blocks_info[-1]
        last_output = last_channels * last_res * last_res
        f.write(f"#define SYMBEX_ULTIMA_SALIDA {last_output}\n")

    return len(blocks_info)


def generar_wiring_cpp(model, ruta_salida, input_size=32):
    """The earlier Spanish name of write_wiring_header, kept as an alias (same arguments)."""
    return write_wiring_header(model, ruta_salida, input_size)
