"""
Exports a fully binarized SYMBEX ResNet (method "plain") to the C header the bit-serial engine reads.

What goes into the header, per conv layer:
  - the weights, 1 bit each (>= 0 -> 1), packed LSB-first into 32-bit words,
    each filter padded to a multiple of 4 words (the engine loads 128 bits at a time)
  - BatchNorm folded with the weight scale alpha:  escala = alpha * gamma / sqrt(var + eps)
                                                    shift  = beta - mean * gamma / sqrt(var + eps)
    stored as fixed point (escala: q_mult fractional bits, everything else: q_act)
  - the RPReLU parameters and the RSign thresholds of the layer's input, also fixed point

Weight order inside each filter (layout):
  - "hwc" (default, current engine): [kh][kw][ic] -- the channels of each pixel together
  - "chw" (old engine):              [ic][kh][kw]
  The header defines SYMBEX_LAYOUT_HWC when layout="hwc", so the C side can refuse old weights.

escala types:
  - conv and downsample: int32 (the engine's SymbexDspParams / SymbexDspParamsSimple)
  - classifier:          int64 (symbex_clasificador_binario_argmax still takes int64)

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


def _pack_rows(rows, multiplo=4):
    """rows: (n_rows, n_weights). Returns (packed words per row, words per row).
    multiplo: each row is padded to a multiple of this many 32-bit words (4 for convs: 128-bit PIE loads;
    1 for the classifier, whose C code advances exactly ceil(n / 32) words per class)."""
    n = rows.shape[1]
    words = -(-((n + 31) // 32) // multiplo) * multiplo
    bits = np.where(rows >= 0, 1, 0)
    packed = []
    for row in bits:
        padded = np.concatenate([row, np.zeros(words * 32 - n, dtype=int)])
        packed.append([int(sum(b << i for i, b in enumerate(padded[k:k + 32])))
                       for k in range(0, words * 32, 32)])
    return packed, words


def _filas(w, layout):
    """(out, in, kh, kw) -> (out, n) en el orden en que el motor lee la ventana.
    hwc: [kh][kw][ic]  (motor actual)    chw: [ic][kh][kw]  (motor viejo)"""
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
    """BatchNorm (+ weight scale alpha, if any) -> integer (escala, shift) per channel."""
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.detach().numpy()
    var = bn.running_var.detach().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    shift = beta - mean * scale
    escala = scale if alpha is None else alpha * scale
    return _q(escala, q_mult), _q(shift, q_act)


def _escala_int32(escala, nombre):
    """El motor guarda escala en int32 (conv y downsample). Verifica que quepa."""
    m = int(np.abs(escala).max())
    assert m < 2 ** 31, f"{nombre}: escala {m} no cabe en int32 (Q_MULT demasiado alto)"
    return escala


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


def _write_conv(f, name, filters, escala, shift, gamma, zeta, slope, beta_in,
                in_ch, out_ch, kernel, stride, padding):
    packed, words = _pack_rows(filters)
    f.write(f"#define {name}_IN_CH {in_ch}\n#define {name}_OUT_CH {out_ch}\n")
    f.write(f"#define {name}_KERNEL {kernel}\n#define {name}_STRIDE {stride}\n")
    f.write(f"#define {name}_PADDING {padding}\n#define {name}_WINDOW_ELEMS {filters.shape[1]}\n\n")
    _weights(f, name, packed, words,
             f"Verificación de integridad: {out_ch} canales, {words} palabras por canal", "Canal")
    _array(f, "int32_t", f"{name}_escala", _escala_int32(escala, name))
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
    escala, shift = _fold_bn(bn, np.abs(w).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma, zeta, slope = _rprelu_params(act, q_act)
    beta_in = np.zeros(w.shape[1]) if rsign is None else _q(rsign.beta.detach().numpy().flatten(), q_act)
    words = _write_conv(f, name, _filas(w, layout), escala, shift, gamma, zeta, slope, beta_in,
                        w.shape[1], w.shape[0], w.shape[2], stride, _padding(conv))
    out, inp = w.shape[0], w.shape[1]
    parts = {"escala (int32)": 4 * out, "shift_bn (int32)": 4 * out,
             "rprelu (gamma, zeta, slope, log2, signo)": 14 * out, "thresholds (RSign, int32)": 4 * inp}
    return _row(name, f"{inp}->{out} k{w.shape[2]} s{stride}", w.shape, words, parts, positions)


def _export_downsample(f, name, ds, positions, q_act, q_mult, layout):
    """RSign + pooling + conv + bn (no activation: the block adds it to the shortcut)."""
    w = ds.conv.weight.detach().numpy()
    escala, shift = _fold_bn(ds.bn, np.abs(w).mean(axis=(1, 2, 3)), q_act, q_mult)
    beta_in = _q(ds.rsign.beta.detach().numpy().flatten(), q_act)
    packed, words = _pack_rows(_filas(w, layout))
    f.write(f"#define {name}_IN_CH {w.shape[1]}\n#define {name}_OUT_CH {w.shape[0]}\n")
    f.write(f"#define {name}_KERNEL {w.shape[2]}\n#define {name}_PADDING {_padding(ds.conv)}\n")
    f.write(f"#define {name}_WINDOW_ELEMS {w.shape[1] * w.shape[2] * w.shape[3]}\n\n")
    _weights(f, name, packed, words,
             f"Verificación Downsample: {w.shape[0]} canales de salida, {words} palabras por canal", "Canal")
    _array(f, "int32_t", f"{name}_escala", _escala_int32(escala, name))
    _array(f, "int32_t", f"{name}_shift_bn", shift)
    _array(f, "int32_t", f"{name}_beta_rsign_entrada", beta_in, end="\n\n")
    out, inp = w.shape[0], w.shape[1]
    parts = {"escala (int32)": 4 * out, "shift_bn (int32)": 4 * out, "thresholds (RSign, int32)": 4 * inp}
    return _row(name, f"{inp}->{out} k{w.shape[2]} pool", w.shape, words, parts, positions)


def _export_classifier(f, model, q_act, q_mult):
    """El clasificador NO cambia: escala sigue en int64 (symbex_dense la pide asi)."""
    w = model.linear.weight.detach().numpy()
    escala, shift = _fold_bn(model.bn_final, None, q_act, q_mult)   # no alpha: LinearBinaria has none
    beta_in = _q(model.rsign_final.beta.detach().numpy().flatten(), q_act)
    packed, words = _pack_rows(w, multiplo=1)          # symbex_dense.c advances ceil(n / 32) words per class
    f.write(f"#define CLASIFICADOR_IN_FEATURES {w.shape[1]}\n#define CLASIFICADOR_OUT_FEATURES {w.shape[0]}\n\n")
    _weights(f, "CLASIFICADOR", packed, words, None, "Clase")
    _array(f, "int64_t", "CLASIFICADOR_escala", escala)
    _array(f, "int32_t", "CLASIFICADOR_shift_bn", shift)
    _array(f, "int32_t", "CLASIFICADOR_beta_rsign_entrada", beta_in, end="\n\n")
    out, inp = w.shape
    parts = {"escala (int64)": 8 * out, "shift_bn (int32)": 4 * out, "thresholds (RSign, int32)": 4 * inp}
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

    last_ch = getattr(model, model._nombres_capas[-1])[-1].conv2.weight.shape[0]
    if model.linear.weight.shape[1] != last_ch * res * res:
        print(f"[!] input_size={input_size} does not match the classifier: it expects "
              f"{model.linear.weight.shape[1]} features, the last stage gives {last_ch * res * res}")

    print_report(model, rows, out_path, detail=detail)
    return rows


# =============================================================================
# GENERAR EL WIRING DE C -- separado de las funciones de arriba (que solo
# escriben pesos/parametros). Esto genera el .h que main.cpp necesita para
# armar los bloques: las declaraciones struct SymbexResidualBlock, el cuerpo
# de symbex_preparar_bloques(), y el arreglo bloques[] -- para N bloques,
# no los 8 fijos que main.cpp tiene hoy escritos a mano.
#
# El motor de C (symbex_network_conv_procesar_bloque, symbex_network_conv.c)
# ya es generico en cantidad de bloques -- lo unico hardcodeado a 8 bloques
# era main.cpp. Esto genera lo que main.cpp necesitaria para dejar de estarlo.
# =============================================================================

def generar_wiring_cpp(model, ruta_salida, input_size=32):
    """
    Genera un .h con las declaraciones de bloques + symbex_preparar_bloques()
    + el arreglo bloques[] (global, para que main.cpp lo use directo en vez
    de declarar sus propias variables locales), para el numero de stages/
    bloques que tenga `model` -- el que sea, no fijo en 4/8.

    main.cpp necesita: #include este archivo, y usar los globales
    `bloques`, `n_bloques`, `nombres_bloques` en vez de declarar los suyos.
    """
    assert hasattr(model, "_nombres_capas"), "model no es un ResNet de SYMBEX (no tiene _nombres_capas)"

    bloques_info = []   # (nombre, tiene_downsample, res_entrada_conv1, res_conv2, out_ch_conv2)
    res = input_size
    for s, stage in enumerate(model._nombres_capas):
        if s > 0:
            res //= 2
        for i, block in enumerate(getattr(model, stage)):
            nombre = f"{stage.upper()}_{i}"
            stride = block.conv1.stride
            stride = stride if isinstance(stride, int) else stride[0]
            res_entrada_conv1 = res * stride       # HACER_CONV pide resolucion de ENTRADA, no salida
            out_ch_conv2 = block.conv2.weight.shape[0]
            bloques_info.append((nombre, block.necesita_downsample, res_entrada_conv1, res, out_ch_conv2))

    with open(ruta_salida, "w") as f:
        f.write("// Generado por compiler_core/export/plain.py:generar_wiring_cpp() -- no editar a mano.\n")
        f.write("// main.cpp debe #include este archivo y usar bloques[] / n_bloques / nombres_bloques[]\n")
        f.write("// en vez de declarar los suyos.\n\n")
        f.write(f"#define RES_STEM {input_size}\n\n")

        for nombre, tiene_ds, res_in1, res_out, _ in bloques_info:
            var = f"bloque_{nombre.lower()}"
            f.write(f"struct SymbexResidualBlock {var} = {{ "
                    f"HACER_CONV({nombre}_CONV1, {res_in1}, {res_in1}), "
                    f"HACER_CONV({nombre}_CONV2, {res_out}, {res_out}), "
                    f"NULL, NULL, {'true' if tiene_ds else 'false'} }};\n")

        f.write("\nvoid symbex_preparar_bloques() {\n")
        for nombre, tiene_ds, _, _, _ in bloques_info:
            var = f"bloque_{nombre.lower()}"
            f.write(f"    {var}.beta_rsign1 = {nombre}_CONV1_beta_rsign_entrada;\n")
            f.write(f"    {var}.beta_rsign2 = {nombre}_CONV2_beta_rsign_entrada;\n")
            if tiene_ds:
                ds_var = f"ds_{nombre.lower()}"
                f.write(f"    static const SymbexDspParamsSimple {ds_var} = {{ {nombre}_DOWNSAMPLE_escala, "
                        f"{nombre}_DOWNSAMPLE_shift_bn, SYMBEX_Q_ACT, SYMBEX_Q_MULT }};\n")
                f.write(f"    {var}.downsample_pesos = {nombre}_DOWNSAMPLE_pesos;\n")
                f.write(f"    {var}.beta_rsign_downsample = {nombre}_DOWNSAMPLE_beta_rsign_entrada;\n")
                f.write(f"    {var}.downsample_dsp = {ds_var};\n")
                f.write(f"    {var}.downsample_kernel = {nombre}_DOWNSAMPLE_KERNEL;\n")
                f.write(f"    {var}.downsample_padding = {nombre}_DOWNSAMPLE_PADDING;\n")
            f.write("\n")
        f.write("}\n\n")

        f.write("struct SymbexBloqueEntry { struct SymbexResidualBlock* bloque; uint32_t n_salida; };\n")
        f.write("struct SymbexBloqueEntry bloques[] = {\n")
        for nombre, _, _, res_out, out_ch in bloques_info:
            var = f"bloque_{nombre.lower()}"
            f.write(f"    {{&{var}, (uint32_t){nombre}_CONV2_OUT_CH * {res_out} * {res_out}}},\n")
        f.write("};\n")
        f.write(f"const int n_bloques = {len(bloques_info)};\n")
        f.write("const char* nombres_bloques[] = {" +
                ", ".join(f'"{n.lower()}"' for n, _, _, _, _ in bloques_info) + "};\n\n")

        # Tamano REAL de la salida del ultimo bloque -- buffer_final/bits_finales
        # en main.cpp deben dimensionarse con esto, NO con un numero fijo.
        # Antes estaba hardcodeado a 64*4*4 (el caso de 4 stages); con menos
        # stages la resolucion final es mayor y ese tamano fijo desborda el
        # arreglo -- confirmado con 3 stages (3/15, corrupcion de memoria).
        _, _, _, ultima_res, ultimos_canales = bloques_info[-1]
        ultima_salida = ultimos_canales * ultima_res * ultima_res
        f.write(f"#define SYMBEX_ULTIMA_SALIDA {ultima_salida}\n")

    return len(bloques_info)
