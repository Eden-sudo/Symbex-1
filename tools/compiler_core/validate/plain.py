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
itself: that lives at the project root, not inside compiler_core -- same
separation export/plain.py already uses).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..export.plain import _fold_bn, _padding, _rprelu_params


def _float_to_q(x, q_act):
    return np.round(x * (2 ** q_act)).astype(np.int64)


def _q_to_float(x_q, q_act):
    return x_q.astype(np.float64) / (2 ** q_act)


def _apply_rprelu(raw_np, identity_q_np, gamma_q, zeta_q, slope_q, scale_q, shift_bn_q, q_act, q_mult):
    """y = scale*raw + shift_bn + identity, then RPReLU(y - gamma) + zeta, all fixed point.
    Only the slope branch: export/plain.py never emits the alternate shift-based RPReLU
    (USA_SHIFT is always 0 in the exported header), so simulating that branch here would
    validate math the engine cannot actually receive from the exporter."""
    C = raw_np.shape[0]
    scale_q = scale_q.reshape(C, 1, 1)
    shift_bn_q = shift_bn_q.reshape(C, 1, 1)
    gamma_q = gamma_q.reshape(C, 1, 1)
    zeta_q = zeta_q.reshape(C, 1, 1)
    slope_q = slope_q.reshape(C, 1, 1)
    raw = raw_np.astype(np.int64)

    shift_amount = q_mult - q_act
    rounder = 1 << (shift_amount - 1)
    y_q = ((raw * scale_q + rounder) >> shift_amount) + shift_bn_q + identity_q_np

    diff_q = y_q - gamma_q
    rounder_act = 1 << (q_act - 1)
    # Always +half BEFORE the arithmetic shift, whatever the sign of diff_q -- with an
    # arithmetic shift (which rounds toward -infinity), always adding +half already gives
    # correct round-to-nearest for any sign. This used to have a sign correction
    # (rounder_act * sign_diff) that does NOT exist in symbex_mac_1bit.c and broke the
    # rounding -- confirmed numerically: always 1 too many toward negative. This version
    # is faithful to symbex_dsp_aplicar_capa (C).
    out_negative = ((diff_q * slope_q + rounder_act) >> q_act) + zeta_q
    out_positive = diff_q + zeta_q
    return np.where(diff_q <= 0, out_negative, out_positive)


def _apply_downsample(raw_np, scale_q, shift_bn_q, q_act, q_mult):
    C = raw_np.shape[0]
    scale_q = scale_q.reshape(C, 1, 1)
    shift_bn_q = shift_bn_q.reshape(C, 1, 1)
    shift_amount = q_mult - q_act
    rounder = 1 << (shift_amount - 1)
    y_q = (raw_np.astype(np.int64) * scale_q + rounder) >> shift_amount
    return y_q + shift_bn_q


def _prepare_block(block, q_act, q_mult):
    """Pre-computes everything about one residual block that does not depend on the input image."""
    w1 = block.conv1.weight.detach().numpy()
    scale1, shift1 = _fold_bn(block.bn1, np.abs(w1).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma1, zeta1, slope1 = _rprelu_params(block.act1, q_act)
    stride1 = block.conv1.stride
    stride1 = stride1 if isinstance(stride1, int) else stride1[0]

    w2 = block.conv2.weight.detach().numpy()
    scale2, shift2 = _fold_bn(block.bn2, np.abs(w2).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma2, zeta2, slope2 = _rprelu_params(block.act2, q_act)

    data = dict(
        block=block, has_downsample=block.necesita_downsample,
        w1_bin=torch.from_numpy(np.where(w1 >= 0, 1.0, -1.0).astype(np.float32)), stride1=stride1,
        padding1=_padding(block.conv1), padding2=_padding(block.conv2),
        scale1=scale1, shift1=shift1, gamma1=gamma1, zeta1=zeta1, slope1=slope1,
        w2_bin=torch.from_numpy(np.where(w2 >= 0, 1.0, -1.0).astype(np.float32)),
        scale2=scale2, shift2=shift2, gamma2=gamma2, zeta2=zeta2, slope2=slope2,
    )
    if block.necesita_downsample:
        ds = block.downsample
        wd = ds.conv.weight.detach().numpy()
        scale_d, shift_d = _fold_bn(ds.bn, np.abs(wd).mean(axis=(1, 2, 3)), q_act, q_mult)
        data.update(
            ds_w_bin=torch.from_numpy(np.where(wd >= 0, 1.0, -1.0).astype(np.float32)),
            ds_padding=_padding(ds.conv), ds_scale=scale_d, ds_shift=shift_d,
            ds_beta_rsign=_float_to_q(ds.rsign.beta.detach().numpy().flatten(), q_act),
        )
    return data


def _simulate_block(layer, x_q, q_act, q_mult):
    if layer["has_downsample"]:
        beta_ds = layer["ds_beta_rsign"].reshape(-1, 1, 1)
        bits_ds = torch.from_numpy(np.where(x_q >= beta_ds, 1.0, -1.0).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pooled = torch.where(F.avg_pool2d(bits_ds, 2) >= 0, 1.0, -1.0)
            raw_ds = F.conv2d(pooled, layer["ds_w_bin"], stride=1, padding=layer["ds_padding"])[0].numpy()
        identity_q = _apply_downsample(raw_ds, layer["ds_scale"], layer["ds_shift"], q_act, q_mult)
    else:
        identity_q = x_q

    beta_r1 = _float_to_q(layer["block"].rsign1.beta.detach().numpy().flatten(), q_act).reshape(-1, 1, 1)
    x_bits = torch.from_numpy(np.where(x_q >= beta_r1, 1.0, -1.0).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        raw1 = F.conv2d(x_bits, layer["w1_bin"], stride=layer["stride1"], padding=layer["padding1"])[0].numpy()
    out1_q = _apply_rprelu(raw1, identity_q, layer["gamma1"], layer["zeta1"], layer["slope1"],
                                layer["scale1"], layer["shift1"], q_act, q_mult)

    beta_r2 = _float_to_q(layer["block"].rsign2.beta.detach().numpy().flatten(), q_act).reshape(-1, 1, 1)
    bits2 = torch.from_numpy(np.where(out1_q >= beta_r2, 1.0, -1.0).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        raw2 = F.conv2d(bits2, layer["w2_bin"], stride=1, padding=layer["padding2"])[0].numpy()
    return _apply_rprelu(raw2, out1_q, layer["gamma2"], layer["zeta2"], layer["slope2"],
                           layer["scale2"], layer["shift2"], q_act, q_mult)


def _classifier_detail(model, x_q, q_act, q_mult):
    """raw + logit_q per class, for ONE image already processed up to buffer_final (x_q).
    Reuses the SAME _fold_bn that exports the .h -- does not re-implement the formula."""
    from ..export.plain import _fold_bn, _rprelu_params
    w = model.linear.weight.detach().numpy()
    w_bin = np.where(w >= 0, 1, -1)
    scale, shift = _fold_bn(model.bn_final, None, q_act, q_mult)
    beta = _float_to_q(model.rsign_final.beta.detach().numpy().flatten(), q_act)

    bits = np.where(x_q.flatten() >= beta, 1, -1)
    rescale_shift = q_mult - q_act
    rounding = 1 << (rescale_shift - 1)
    result = []
    for n in range(w.shape[0]):
        raw = int((bits * w_bin[n]).sum())
        logit_q = ((raw * int(scale[n])) + rounding) >> rescale_shift
        logit_q += int(shift[n])
        result.append((raw, logit_q))
    return result


def _simulate_stem(model, x_bits, q_act, q_mult):
    """
    Simulates the stem truly IN FIXED POINT (integer XNOR+popcount +
    _apply_rprelu), like _simulate_block -- it does not run
    model.conv1/bn1/rprelu_stem in floating point. identity=0 because the
    stem has no residual connection.
    """
    w = model.conv1.weight.detach().numpy()
    w_bin = torch.from_numpy(np.where(w >= 0, 1.0, -1.0).astype(np.float32))
    with torch.no_grad():
        raw = F.conv2d(x_bits, w_bin, stride=1, padding=_padding(model.conv1))[0].numpy()
    scale, shift = _fold_bn(model.bn1, np.abs(w).mean(axis=(1, 2, 3)), q_act, q_mult)
    gamma, zeta, slope = _rprelu_params(model.rprelu_stem, q_act)
    identity_zero = np.zeros(raw.shape, dtype=np.int64)   # explicit int64 -- np.zeros_like(raw) inherited float32 from conv2d and broke the shift below
    return _apply_rprelu(raw, identity_zero, gamma, zeta, slope, scale, shift, q_act, q_mult)


def check_conv_fidelity(model, testloader, convert_input, q_act=16, q_mult=20,
                            max_images=None, verbose=True, detail_image=None):
    """
    Runs the fixed-point simulation over `testloader` and compares it against
    the model's own PyTorch forward pass.

    convert_input: the function that turns a raw image into whatever
    model.conv1 actually expects (e.g. RGB_a_24Bits from symbex_cifar10.py).
    Passed in by the caller instead of imported here, so this module stays
    independent of any specific project's model file -- the same separation
    export/plain.py already keeps.

    Returns a dict: agreement (simulator vs PyTorch -- the number that matters
    for validating the exporter), acc_simulator (also under its earlier name
    acc_simulador) and acc_pytorch (vs the true label, for reference), total.
    """
    model.eval()
    dsp_layers = [(f"{stage}_{i}", _prepare_block(block, q_act, q_mult))
                for stage in model._nombres_capas for i, block in enumerate(getattr(model, stage))]

    matches = correct_sim = correct_pt = total = 0
    t0 = time.time()
    for idx_img, (img, label) in enumerate(testloader):
        if max_images is not None and total >= max_images:
            break
        true_label = label.item()

        with torch.no_grad():
            class_pt = model(img).argmax(dim=1).item()
            x_bits_stem = convert_input(img)
        x_q = _simulate_stem(model, x_bits_stem, q_act, q_mult)

        debug = (detail_image is not None and idx_img == detail_image)
        if debug:
            print(f"\n[*] Image {idx_img} (Python simulator, block by block)")
            print(f"  stem | min={x_q.min()} max={x_q.max()} sum={x_q.sum()}")

        for name, layer in dsp_layers:
            x_q = _simulate_block(layer, x_q, q_act, q_mult)
            if debug:
                print(f"  {name} | min={x_q.min()} max={x_q.max()} sum={x_q.sum()}")

        if detail_image is not None and idx_img == detail_image:
            print(f"\n[*] Image {idx_img} | true_label={true_label} "
                  f"| buffer_final (simulator): min={x_q.min()} max={x_q.max()} sum={x_q.sum()}")
            print("  --- classifier (Python simulator, the same formula the .h exports) ---")
            for n, (raw, logit_q) in enumerate(_classifier_detail(model, x_q, q_act, q_mult)):
                print(f"    class {n} | raw={raw} | logit_q={logit_q}")

        # integer classifier, the same arithmetic as symbex_clasificador_binario_argmax (first strict maximum wins)
        logits_q = [logit_q for _, logit_q in _classifier_detail(model, x_q, q_act, q_mult)]
        class_sim = int(np.argmax(logits_q))

        total += 1
        matches += (class_sim == class_pt)
        correct_sim += (class_sim == true_label)
        correct_pt += (class_pt == true_label)

        if verbose and total % 500 == 0:
            print(f"  {total} images | Matches PyTorch: {100 * matches / total:.2f}% | "
                  f"Simulator vs label: {100 * correct_sim / total:.2f}% | "
                  f"PyTorch vs label: {100 * correct_pt / total:.2f}% | {(time.time() - t0) / 60:.1f} min")

    acc_sim = 100.0 * correct_sim / total
    return dict(agreement=100.0 * matches / total, acc_simulator=acc_sim, acc_simulador=acc_sim,
                acc_pytorch=100.0 * correct_pt / total, total=total)


def verificar_fidelidad_conv(modelo, testloader, convertir_entrada, q_act=16, q_mult=20,
                             max_imagenes=None, verbose=True, detalle_imagen=None):
    """The earlier Spanish name of check_conv_fidelity, kept as an alias (same arguments)."""
    return check_conv_fidelity(modelo, testloader, convertir_entrada, q_act, q_mult,
                               max_images=max_imagenes, verbose=verbose, detail_image=detalle_imagen)
