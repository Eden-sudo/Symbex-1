"""
Bit-exact CPU simulator for the "plain" conv method (bit-serial engine, ESP32-S3).

Reconstructs the SAME fixed-point arithmetic the C engine does (Q_ACT/Q_MULT,
folded BatchNorm, RPReLU) using PyTorch's own conv2d/avg_pool2d as the
XNOR+popcount stand-in (both operate on {-1,+1} tensors; conv2d on bipolar
inputs IS the XNOR+popcount count, just computed with floats instead of bits
-- this was cross-checked against the real hardware earlier in the project,
see resultado_simulador_dsp.txt). Comparing its predictions against the
model's own PyTorch forward pass is the "Hardware Math Fidelity" check.

Uses the SAME quantization functions as export/plain.py (_fold_bn,
_rprelu_params) instead of re-deriving the formula here -- export and
validate must measure the exact same math, or "validated" means nothing.

Takes an ALREADY BUILT AND LOADED model as input (does not import symbex_binario
itself, that lives at the project root, not inside compiler_core -- same
separation export/plain.py already uses).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..export.plain import _fold_bn, _padding, _rprelu_params


def _float_a_q(x, q_act):
    return np.round(x * (2 ** q_act)).astype(np.int64)


def _q_a_float(x_q, q_act):
    return x_q.astype(np.float64) / (2 ** q_act)


def _aplicar_rprelu(raw_np, identity_q_np, gamma_q, zeta_q, slope_q, escala_q, shift_bn_q, q_act, q_mult):
    """y = escala*raw + shift_bn + identity, then RPReLU(y - gamma) + zeta, all fixed point.
    Only the slope branch: export/plain.py never emits the alternate shift-based RPReLU
    (USA_SHIFT is always 0 in the exported header), so simulating that branch here would
    validate math the engine cannot actually receive from the exporter."""
    C = raw_np.shape[0]
    escala_q = escala_q.reshape(C, 1, 1)
    shift_bn_q = shift_bn_q.reshape(C, 1, 1)
    gamma_q = gamma_q.reshape(C, 1, 1)
    zeta_q = zeta_q.reshape(C, 1, 1)
    slope_q = slope_q.reshape(C, 1, 1)
    raw = raw_np.astype(np.int64)

    shift_amount = q_mult - q_act
    rounder = 1 << (shift_amount - 1)
    y_q = ((raw * escala_q + rounder) >> shift_amount) + shift_bn_q + identity_q_np

    diff_q = y_q - gamma_q
    rounder_act = 1 << (q_act - 1)
    # Siempre +mitad ANTES del shift aritmetico, sin importar el signo de diff_q --
    # con shift aritmetico (que redondea hacia -infinito), sumar siempre +mitad ya da
    # "redondeo al mas cercano" correcto para cualquier signo. Antes esto tenia una
    # correccion de signo (rounder_act * signo_diff) que NO existe en symbex_mac_1bit.c
    # y que rompia el redondeo -- confirmado numericamente: siempre 1 de mas hacia
    # negativo. La version de aca ahora es fiel a symbex_dsp_aplicar_capa (C).
    salida_negativa = ((diff_q * slope_q + rounder_act) >> q_act) + zeta_q
    salida_positiva = diff_q + zeta_q
    return np.where(diff_q <= 0, salida_negativa, salida_positiva)


def _aplicar_downsample(raw_np, escala_q, shift_bn_q, q_act, q_mult):
    C = raw_np.shape[0]
    escala_q = escala_q.reshape(C, 1, 1)
    shift_bn_q = shift_bn_q.reshape(C, 1, 1)
    shift_amount = q_mult - q_act
    rounder = 1 << (shift_amount - 1)
    y_q = (raw_np.astype(np.int64) * escala_q + rounder) >> shift_amount
    return y_q + shift_bn_q


def _preparar_bloque(bloque, q_act, q_mult):
    """Pre-computes everything about one residual block that does not depend on the input image."""
    w1 = bloque.conv1.weight.detach().numpy()
    escala1, shift1 = _fold_bn(bloque.bn1, np.abs(w1).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma1, zeta1, slope1 = _rprelu_params(bloque.act1, q_act)
    stride1 = bloque.conv1.stride
    stride1 = stride1 if isinstance(stride1, int) else stride1[0]

    w2 = bloque.conv2.weight.detach().numpy()
    escala2, shift2 = _fold_bn(bloque.bn2, np.abs(w2).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma2, zeta2, slope2 = _rprelu_params(bloque.act2, q_act)

    datos = dict(
        bloque=bloque, tiene_downsample=bloque.necesita_downsample,
        w1_bin=torch.from_numpy(np.where(w1 >= 0, 1.0, -1.0).astype(np.float32)), stride1=stride1,
        padding1=_padding(bloque.conv1), padding2=_padding(bloque.conv2),
        escala1=escala1, shift1=shift1, gamma1=gamma1, zeta1=zeta1, slope1=slope1,
        w2_bin=torch.from_numpy(np.where(w2 >= 0, 1.0, -1.0).astype(np.float32)),
        escala2=escala2, shift2=shift2, gamma2=gamma2, zeta2=zeta2, slope2=slope2,
    )
    if bloque.necesita_downsample:
        ds = bloque.downsample
        wd = ds.conv.weight.detach().numpy()
        escala_d, shift_d = _fold_bn(ds.bn, np.abs(wd).mean(axis=(1, 2, 3)), q_act, q_mult)
        datos.update(
            ds_w_bin=torch.from_numpy(np.where(wd >= 0, 1.0, -1.0).astype(np.float32)),
            ds_padding=_padding(ds.conv), ds_escala=escala_d, ds_shift=shift_d,
            ds_beta_rsign=_float_a_q(ds.rsign.beta.detach().numpy().flatten(), q_act),
        )
    return datos


def _simular_bloque(capa, x_q, q_act, q_mult):
    if capa["tiene_downsample"]:
        beta_ds = capa["ds_beta_rsign"].reshape(-1, 1, 1)
        bits_ds = torch.from_numpy(np.where(x_q >= beta_ds, 1.0, -1.0).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pooled = torch.where(F.avg_pool2d(bits_ds, 2) >= 0, 1.0, -1.0)
            raw_ds = F.conv2d(pooled, capa["ds_w_bin"], stride=1, padding=capa["ds_padding"])[0].numpy()
        identity_q = _aplicar_downsample(raw_ds, capa["ds_escala"], capa["ds_shift"], q_act, q_mult)
    else:
        identity_q = x_q

    beta_r1 = _float_a_q(capa["bloque"].rsign1.beta.detach().numpy().flatten(), q_act).reshape(-1, 1, 1)
    x_bits = torch.from_numpy(np.where(x_q >= beta_r1, 1.0, -1.0).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        raw1 = F.conv2d(x_bits, capa["w1_bin"], stride=capa["stride1"], padding=capa["padding1"])[0].numpy()
    salida1_q = _aplicar_rprelu(raw1, identity_q, capa["gamma1"], capa["zeta1"], capa["slope1"],
                                capa["escala1"], capa["shift1"], q_act, q_mult)

    beta_r2 = _float_a_q(capa["bloque"].rsign2.beta.detach().numpy().flatten(), q_act).reshape(-1, 1, 1)
    bits2 = torch.from_numpy(np.where(salida1_q >= beta_r2, 1.0, -1.0).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        raw2 = F.conv2d(bits2, capa["w2_bin"], stride=1, padding=capa["padding2"])[0].numpy()
    return _aplicar_rprelu(raw2, salida1_q, capa["gamma2"], capa["zeta2"], capa["slope2"],
                           capa["escala2"], capa["shift2"], q_act, q_mult)


def _detalle_clasificador(modelo, x_q, q_act, q_mult):
    """raw + logit_q por clase, para UNA imagen ya procesada hasta buffer_final (x_q).
    Reusa la MISMA _fold_bn que exporta el .h -- no reimplementa la formula aparte."""
    from ..export.plain import _fold_bn, _rprelu_params
    w = modelo.linear.weight.detach().numpy()
    w_bin = np.where(w >= 0, 1, -1)
    escala, shift = _fold_bn(modelo.bn_final, None, q_act, q_mult)
    beta = _float_a_q(modelo.rsign_final.beta.detach().numpy().flatten(), q_act)

    bits = np.where(x_q.flatten() >= beta, 1, -1)
    shift_reescalado = q_mult - q_act
    redondeo = 1 << (shift_reescalado - 1)
    resultado = []
    for n in range(w.shape[0]):
        raw = int((bits * w_bin[n]).sum())
        logit_q = ((raw * int(escala[n])) + redondeo) >> shift_reescalado
        logit_q += int(shift[n])
        resultado.append((raw, logit_q))
    return resultado


def _simular_stem(modelo, x_bits, q_act, q_mult):
    """
    Simula el stem EN PUNTO FIJO de verdad (XNOR+popcount entero +
    _aplicar_rprelu), igual que _simular_bloque -- no corre
    modelo.conv1/bn1/rprelu_stem en flotante. identity=0 porque el
    stem no tiene conexion residual.
    """
    w = modelo.conv1.weight.detach().numpy()
    w_bin = torch.from_numpy(np.where(w >= 0, 1.0, -1.0).astype(np.float32))
    with torch.no_grad():
        raw = F.conv2d(x_bits, w_bin, stride=1, padding=_padding(modelo.conv1))[0].numpy()
    escala, shift = _fold_bn(modelo.bn1, np.abs(w).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma, zeta, slope = _rprelu_params(modelo.rprelu_stem, q_act)
    identity_cero = np.zeros(raw.shape, dtype=np.int64)   # int64 explicito -- np.zeros_like(raw) heredaba float32 del conv2d y rompia el shift de mas abajo
    return _aplicar_rprelu(raw, identity_cero, gamma, zeta, slope, escala, shift, q_act, q_mult)


def verificar_fidelidad_conv(modelo, testloader, convertir_entrada, q_act=16, q_mult=20,
                            max_imagenes=None, verbose=True, detalle_imagen=None):
    """
    Runs the fixed-point simulation over `testloader` and compares it against
    the model's own PyTorch forward pass.

    convertir_entrada: the function that turns a raw image into whatever
    modelo.conv1 actually expects (e.g. RGB_a_24Bits from symbex_cifar10.py).
    Passed in by the caller instead of imported here, so this module stays
    independent of any specific project's model file -- the same separation
    export/plain.py already keeps.

    Returns a dict: agreement (simulator vs PyTorch -- the number that matters
    for validating the exporter), acc_simulador and acc_pytorch (vs the true
    label, for reference), total.
    """
    modelo.eval()
    capas_dsp = [(f"{stage}_{i}", _preparar_bloque(bloque, q_act, q_mult))
                for stage in modelo._nombres_capas for i, bloque in enumerate(getattr(modelo, stage))]

    coincidencias = correctos_sim = correctos_pt = total = 0
    t0 = time.time()
    for idx_img, (img, label) in enumerate(testloader):
        if max_imagenes is not None and total >= max_imagenes:
            break
        etiqueta_real = label.item()

        with torch.no_grad():
            clase_real = modelo(img).argmax(dim=1).item()
            x_bits_stem = convertir_entrada(img)
        x_q = _simular_stem(modelo, x_bits_stem, q_act, q_mult)

        debug = (detalle_imagen is not None and idx_img == detalle_imagen)
        if debug:
            print(f"\n[*] Imagen {idx_img} (simulador Python, bloque por bloque)")
            print(f"  stem | min={x_q.min()} max={x_q.max()} suma={x_q.sum()}")

        for nombre, capa in capas_dsp:
            x_q = _simular_bloque(capa, x_q, q_act, q_mult)
            if debug:
                print(f"  {nombre} | min={x_q.min()} max={x_q.max()} suma={x_q.sum()}")

        if detalle_imagen is not None and idx_img == detalle_imagen:
            print(f"\n[*] Imagen {idx_img} | etiqueta_real={etiqueta_real} "
                  f"| buffer_final (simulador): min={x_q.min()} max={x_q.max()} suma={x_q.sum()}")
            print("  --- clasificador (Python/simulador, misma formula que exporta el .h) ---")
            for n, (raw, logit_q) in enumerate(_detalle_clasificador(modelo, x_q, q_act, q_mult)):
                print(f"    clase {n} | raw={raw} | logit_q={logit_q}")

        # integer classifier, the same arithmetic as symbex_clasificador_binario_argmax (first strict maximum wins)
        logits_q = [logit_q for _, logit_q in _detalle_clasificador(modelo, x_q, q_act, q_mult)]
        clase_sim = int(np.argmax(logits_q))

        total += 1
        coincidencias += (clase_sim == clase_real)
        correctos_sim += (clase_sim == etiqueta_real)
        correctos_pt += (clase_real == etiqueta_real)

        if verbose and total % 500 == 0:
            print(f"  {total} imagenes | Coincide PyTorch: {100 * coincidencias / total:.2f}% | "
                  f"Simulador vs real: {100 * correctos_sim / total:.2f}% | "
                  f"PyTorch vs real: {100 * correctos_pt / total:.2f}% | {(time.time() - t0) / 60:.1f} min")

    return dict(agreement=100.0 * coincidencias / total, acc_simulador=100.0 * correctos_sim / total,
               acc_pytorch=100.0 * correctos_pt / total, total=total)
