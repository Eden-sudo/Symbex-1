"""
================================================================================
 SYMBEX-1 SENSITIVITY SCAN  (step 1 of the pipeline: SCAN)
================================================================================
 Measures, layer by layer, how much a network suffers when its weights,
 activations and normalization are reduced to k bits. Works on the user's
 ORIGINAL model (FP32), before binarizing it.

 Usage:
     from compiler_core.sensitivity import Sensitivity
     Sensitivity(model).scan(calibration)                 # one row per layer
     Sensitivity(model).scan(calibration, detail=True)    # + the 5 detail tables per layer

 `calibration` must be REAL samples of the user's data (e.g. 32 images), not
 torch.randn: with random data the activations mean nothing.

 What the numbers mean: cosine similarity between the layer's output and its
 FP32 output, ONE layer at a time, with uniform k-bit quantization. Use it to
 compare layers with each other. It does not predict the accuracy of the whole
 network, because errors accumulate through the layers.

 Audit notes of the measurement code (kept from v3):
 1. Dynamic sizing for 2D (MLP), 3D (Transformers) and 4D (CNN) tensors.
 2. LayerNorm and BatchNorm handled natively.
 3. No double bias: F.conv2d / F.linear run without bias, the norm handler
    applies the (quantized) bias.
 4. Universal clipping at 2*sigma for weights (min-max destroys entropy with
    outliers, shown empirically on ViT for k >= 5).
 5. Stratified quantization noise (Table 5): an aggregate fidelity can look fine
    while values near zero suffer a huge relative error.

 Added in SYMBEX v3 (nothing above was removed):
 6. Norm pairing by EXECUTION order: the norm of a layer is the one that receives
    that layer's output in the forward pass (before: the next module in declaration
    order, which pairs wrongly when a model declares its BatchNorm before its conv).
    scan() warns about every layer where the two rules disagree.
 7. estimate(): the SYMBEX binarization ("bin": sign(w) * mean|w| per channel, input
    sign(x - t) * s per channel) next to k-bit options, the memory of every option
    (flash and RAM), and the decision: start at 1 bit, raise the bits of a layer only
    when it needs them (tolerance), with the hardware budget as a ceiling.
        Sensitivity(model).estimate(calibration, ram_kb=320, flash_kb=4096,
                                    budget=0.25, tolerance=0.99)              # detail=True by default
 8. estimate() keeps the input of the FIRST layer unquantized (only its weights), because training does
    the same (methods.apply_method / apply_plan leave the raw input untouched).
 9. Self-test:  python sensitivity.py

 Translated to English: the old Spanish names (estimar, memoria_capa, ...), keyword arguments and result keys
 (filas, elegida, ...) still work until every caller is translated (see RENAMES.md).
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Sensitivity:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @staticmethod
    def quantize_weight_v3(tensor, k_bits):
        """SYMBEX rule (updated): universal 2*std clipping, then k-bit uniform rounding.
        Returns (reconstructed weights, integer levels)."""
        if k_bits is None:
            return tensor.clone(), None
        levels = (2 ** k_bits) - 1

        mean = tensor.mean()
        std = tensor.std().clamp(min=1e-9)
        threshold = 2.0 * std
        core = tensor.clamp(mean - threshold, mean + threshold)

        max_val = core.abs().max().clamp(min=1e-9)
        scaled = (core / max_val) * (levels / 2.0) + (levels / 2.0)
        quant = torch.clamp(torch.round(scaled), 0, levels)
        reconstructed = 2.0 * quant - levels

        return reconstructed * (max_val / levels), quant

    @staticmethod
    def quantize_minmax(tensor, k_bits):
        if k_bits is None:
            return tensor.clone()
        levels = (2 ** k_bits) - 1
        t_min = tensor.min()
        t_max = tensor.max()
        scale = (t_max - t_min).clamp(min=1e-9)
        scaled = (tensor - t_min) / scale * levels
        quant = torch.clamp(torch.round(scaled), 0, levels)
        return quant / levels * scale + t_min

    @staticmethod
    def apply_norm_or_bias(x, target_module, norm_module, k_bits_bn=None):
        if isinstance(norm_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            eps = norm_module.eps
            weight = Sensitivity.quantize_minmax(norm_module.weight.data, k_bits_bn) if k_bits_bn else norm_module.weight.data
            bias = Sensitivity.quantize_minmax(norm_module.bias.data, k_bits_bn) if k_bits_bn else norm_module.bias.data
            scale = weight / torch.sqrt(norm_module.running_var + eps)
            shift = bias - norm_module.running_mean * scale
            view_shape = [1, -1] + [1] * (x.dim() - 2)
            return x * scale.view(*view_shape) + shift.view(*view_shape)
        elif isinstance(norm_module, nn.LayerNorm):
            weight = Sensitivity.quantize_minmax(norm_module.weight.data, k_bits_bn) if k_bits_bn else norm_module.weight.data
            bias = Sensitivity.quantize_minmax(norm_module.bias.data, k_bits_bn) if k_bits_bn else norm_module.bias.data
            return F.layer_norm(x, norm_module.normalized_shape, weight, bias, norm_module.eps)
        else:
            if getattr(target_module, 'bias', None) is not None:
                bias = Sensitivity.quantize_minmax(target_module.bias.data, k_bits_bn) if k_bits_bn else target_module.bias.data
                if isinstance(target_module, nn.Conv2d):
                    return x + bias.view(1, -1, 1, 1)
                else:
                    return x + bias
            return x

    @staticmethod
    def compute_psnr(output, reference):
        mse = ((output - reference) ** 2).mean().item()
        if mse < 1e-12:
            return float('inf')
        max_val = reference.abs().max().item()
        return 10 * np.log10((max_val ** 2) / mse)

    @staticmethod
    def compute_shannon_entropy(integer_levels, k_bits):
        if integer_levels is None:
            return None, None, None
        values, counts = torch.unique(integer_levels, return_counts=True)
        probs = counts.float() / counts.sum()
        real_entropy = -(probs * torch.log2(probs)).sum().item()
        max_entropy = k_bits
        efficiency = real_entropy / max_entropy if max_entropy > 0 else 0
        return real_entropy, max_entropy, efficiency

    @staticmethod
    def compute_stratified_noise(original, reconstructed):
        """
        Quantization noise STRATIFIED by the magnitude of the original value.

        FIX: the previous version used the mean + a tiny absolute epsilon (1e-6). With BatchNorm outputs
        (centered on zero by design), the "smallest 10% by magnitude" includes genuinely ALMOST ZERO values
        (1e-7, 1e-8...), not only "small but meaningful" ones. Dividing by those almost-null magnitudes
        explodes into meaningless numbers (ratios of millions of % were seen), and the MEAN gets
        contaminated by those few outliers.

        Fix: the median (robust to those outliers) + an epsilon relative to the real scale of the tensor
        (not an absolute constant).
        """
        original_flat = original.flatten()
        reconstructed_flat = reconstructed.flatten()
        abs_error = (original_flat - reconstructed_flat).abs()

        magnitudes = original_flat.abs()
        order = torch.argsort(magnitudes)
        n = len(order)
        n_tenth = max(1, n // 10)

        idx_small = order[:n_tenth]
        idx_large = order[-n_tenth:]

        # epsilon relative to the real scale of the tensor, not absolute
        eps = magnitudes.mean().item() * 0.01
        eps = max(eps, 1e-9)

        rel_error_small = (abs_error[idx_small] / (magnitudes[idx_small] + eps)).median().item()
        rel_error_large = (abs_error[idx_large] / (magnitudes[idx_large] + eps)).median().item()
        ratio = rel_error_small / max(rel_error_large, 1e-9)

        return rel_error_small, rel_error_large, ratio

    def run_base(self, x, target_module, weight):
        if isinstance(target_module, nn.Conv2d):
            return F.conv2d(x, weight, bias=None, stride=target_module.stride, padding=target_module.padding)
        elif isinstance(target_module, nn.Linear):
            return F.linear(x, weight, bias=None)

    def measure_combination(self, weight_fp32, images_input, target_module, norm_module, target, k_w, k_a, k_bn):
        with torch.no_grad():
            weight_q, _ = self.quantize_weight_v3(weight_fp32, k_w)
            images_q = self.quantize_minmax(images_input, k_a)
            base_out = self.run_base(images_q, target_module, weight_q)
            out_final = self.apply_norm_or_bias(base_out, target_module, norm_module, k_bits_bn=k_bn)
            fid = F.cosine_similarity(out_final.flatten(), target.flatten(), dim=0).item()
            psnr = self.compute_psnr(out_final, target)
        return fid, psnr

    def analyze_layer(self, target_module, norm_module, x_in, layer_name=None, spectrum=(1, 2, 3, 4, 5, 6, 7, 8),
                      nombre_capa=None, espectro=None):
        """The 5 main tables, with dimensionally consistent operations.
        nombre_capa / espectro: old names of layer_name / spectrum (kept until every caller is translated)."""
        layer_name = nombre_capa if nombre_capa is not None else layer_name
        spectrum = list(espectro if espectro is not None else spectrum)
        print(f"\n" + "=" * 100)
        print(f" EXTENDED SENSITIVITY TABLE -- Layer: {layer_name}")
        print("=" * 100)

        weight_fp32 = target_module.weight.data.clone()
        total_params = weight_fp32.numel()
        input_elements = x_in.numel() // x_in.size(0)

        with torch.no_grad():
            base_ref = self.run_base(x_in, target_module, weight_fp32)
            target = self.apply_norm_or_bias(base_ref, target_module, norm_module, k_bits_bn=None)

        print("\n--- TABLE 1: FIDELITY (cosine) ---")
        print(f"{'k':<4} | {'WEIGHT only':<12} | {'ACTIV only':<12} | {'NORM/bias only':<14} | {'ALL 3':<12}")
        print("-" * 65)

        results_by_k = {}
        for k in spectrum:
            with torch.no_grad():
                weight_q, weight_levels = self.quantize_weight_v3(weight_fp32, k)
                x_q = self.quantize_minmax(x_in, k)

                out_w = self.run_base(x_in, target_module, weight_q)
                out_weight_only = self.apply_norm_or_bias(out_w, target_module, norm_module, k_bits_bn=None)
                fid_weight = F.cosine_similarity(out_weight_only.flatten(), target.flatten(), dim=0).item()

                out_a = self.run_base(x_q, target_module, weight_fp32)
                out_act_only = self.apply_norm_or_bias(out_a, target_module, norm_module, k_bits_bn=None)
                fid_act = F.cosine_similarity(out_act_only.flatten(), target.flatten(), dim=0).item()

                out_bn_only = self.apply_norm_or_bias(base_ref, target_module, norm_module, k_bits_bn=k)
                fid_bn = F.cosine_similarity(out_bn_only.flatten(), target.flatten(), dim=0).item()

                out_all = self.run_base(x_q, target_module, weight_q)
                out_all3 = self.apply_norm_or_bias(out_all, target_module, norm_module, k_bits_bn=k)
                fid_all = F.cosine_similarity(out_all3.flatten(), target.flatten(), dim=0).item()

            results_by_k[k] = {'weight_levels': weight_levels, 'out_all': out_all3}
            print(f"{k:<4} | {fid_weight:<12.6f} | {fid_act:<12.6f} | {fid_bn:<14.6f} | {fid_all:<12.6f}")

        print("\n--- TABLE 2: PSNR (dB) ---")
        print(f"{'k':<4} | {'WEIGHT only':<12} | {'ACTIV only':<12} | {'NORM/bias only':<14} | {'ALL 3':<12}")
        print("-" * 65)
        for k in spectrum:
            with torch.no_grad():
                weight_q, _ = self.quantize_weight_v3(weight_fp32, k)
                x_q = self.quantize_minmax(x_in, k)

                psnr_weight = self.compute_psnr(self.apply_norm_or_bias(self.run_base(x_in, target_module, weight_q), target_module, norm_module, None), target)
                psnr_act = self.compute_psnr(self.apply_norm_or_bias(self.run_base(x_q, target_module, weight_fp32), target_module, norm_module, None), target)
                psnr_bn = self.compute_psnr(self.apply_norm_or_bias(base_ref, target_module, norm_module, k), target)
                psnr_all = self.compute_psnr(results_by_k[k]['out_all'], target)

            def psnr_str(v): return "inf" if v == float('inf') else f"{v:.2f}"
            print(f"{k:<4} | {psnr_str(psnr_weight):<12} | {psnr_str(psnr_act):<12} | {psnr_str(psnr_bn):<14} | {psnr_str(psnr_all):<12}")

        print("\n--- TABLE 3: SHANNON ENTROPY (weights) ---")
        print(f"{'k':<4} | {'Real entropy (bits)':<22} | {'Max entropy':<18} | {'Usage efficiency':<18}")
        print("-" * 68)
        for k in spectrum:
            levels = results_by_k[k]['weight_levels']
            ent_real, ent_max, efficiency = self.compute_shannon_entropy(levels, k)
            print(f"{k:<4} | {ent_real:<22.4f} | {ent_max:<18.4f} | {efficiency*100:<17.1f}%")

        print("\n--- TABLE 4: MEMORY AND OVERALL SPEED ---")
        print(f"{'k':<4} | {'Layer weights (KB)':<18} | {'Activ. RAM (KB)':<18} | {'Speed proxy k*k (NOT measured)':<18}")
        print("-" * 65)
        for k in spectrum:
            weight_kb = total_params * k / 8 / 1024
            ram_kb = input_elements * k / 8 / 1024
            speed_proxy = k * k
            print(f"{k:<4} | {weight_kb:<18.3f} | {ram_kb:<18.3f} | {speed_proxy:<18}")

        print("\n--- TABLE 5: STRATIFIED QUANTIZATION NOISE ('ALL 3' scenario) ---")
        print("(do the values near zero suffer more relative error than the large ones,")
        print(" even when the aggregate fidelity of Table 1 looks fine?)")
        print(f"{'k':<4} | {'Rel. error 10% SMALL':<22} | {'Rel. error 10% LARGE':<22} | {'Ratio':<10} | Alert")
        print("-" * 80)
        for k in spectrum:
            out_all3 = results_by_k[k]['out_all']
            err_small, err_large, ratio = self.compute_stratified_noise(target, out_all3)
            alert = "[!] small values suffer disproportionately" if ratio > 5 else "[+] even"
            print(f"{k:<4} | {err_small*100:<21.1f}% | {err_large*100:<21.1f}% | {ratio:<10.1f}| {alert}")

    # ------------------------------------------------------------------
    # SCAN: the whole network, one summary row per layer
    # ------------------------------------------------------------------
    def _capture_inputs(self, calibration):
        """Runs the model once and returns {layer name: the real input that layer received}.
        Also records (in self._execution_order / self._norm_by_layer) the order in which the layers ran
        and which norm module received each layer's output."""
        inputs = {}
        order, output_of = [], {}
        self._norm_by_layer = {}

        def store(name):
            def hook(module, inp, out):
                inputs[name] = inp[0].detach()
                order.append(name)
                output_of[id(out)] = name
            return hook

        def store_norm(norm_module):
            def hook(module, inp, out):
                layer = output_of.get(id(inp[0]))
                if layer is not None and layer not in self._norm_by_layer:
                    self._norm_by_layer[layer] = norm_module
            return hook

        modules = list(self.model.named_modules())
        hooks = [m.register_forward_hook(store(n)) for n, m in modules
                 if isinstance(m, (nn.Conv2d, nn.Linear))]
        hooks += [m.register_forward_hook(store_norm(m)) for _, m in modules
                  if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm))]
        with torch.no_grad():
            self.model(calibration)
        for h in hooks:
            h.remove()
        self._execution_order = list(dict.fromkeys(order))
        return modules, inputs

    def _norm_of(self, name, i, modules):
        """The norm of layer `name`: the one that received its output in the forward pass.
        Returns (norm, norm by the old declaration-order rule) so callers can report disagreements."""
        next_module = modules[i + 1][1] if (i + 1) < len(modules) else None
        by_declaration = next_module if isinstance(next_module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)) else None
        return self._norm_by_layer.get(name), by_declaration

    def _fid_with_weights(self, weight_q, x_in, module, norm, target, k):
        """Cosine fidelity when the weights are `weight_q` and activations + norm are quantized at k bits."""
        with torch.no_grad():
            output = self.apply_norm_or_bias(
                self.run_base(self.quantize_minmax(x_in, k), module, weight_q), module, norm, k_bits_bn=k)
            return F.cosine_similarity(output.flatten(), target.flatten(), dim=0).item()

    def _measure_layer(self, name, module, norm, x_in, spectrum, threshold):
        """Cosine fidelity of ONE layer for every k in `spectrum` (weights + activations + norm at k bits)."""
        weight = module.weight.data.clone()
        with torch.no_grad():
            base = self.run_base(x_in, module, weight)
            target = self.apply_norm_or_bias(base, module, norm, k_bits_bn=None)
        fid = {}
        for k in spectrum:
            # the better of two weight rules: 2*std clipping (good at low k) and plain min-max (good at high k,
            # where clipping the tails of a pretrained layer costs more than the rounding it saves)
            fid[k] = max(self._fid_with_weights(self.quantize_weight_v3(weight, k)[0], x_in, module, norm, target, k),
                         self._fid_with_weights(self.quantize_minmax(weight, k), x_in, module, norm, target, k))
        # CLIP: weights clipped at mean +- 2*std but NOT rounded. Low = the layer depends on its largest weights.
        mean, std = weight.mean(), weight.std().clamp(min=1e-9)
        with torch.no_grad():
            output = self.apply_norm_or_bias(
                self.run_base(x_in, module, weight.clamp(mean - 2.0 * std, mean + 2.0 * std)),
                module, norm, k_bits_bn=None)
            clip = F.cosine_similarity(output.flatten(), target.flatten(), dim=0).item()
        if isinstance(norm, (nn.BatchNorm2d, nn.BatchNorm1d)):
            norm_txt = "BN"
        elif isinstance(norm, nn.LayerNorm):
            norm_txt = "LN"
        else:
            norm_txt = "bias" if getattr(module, 'bias', None) is not None else "-"
        return dict(name=name, type=type(module).__name__,
                    shape="x".join(str(d) for d in module.weight.shape),
                    params=module.weight.numel(), norm=norm_txt, fid=fid, clip=clip,
                    min_k=next((k for k in spectrum if fid[k] >= threshold), None))

    @staticmethod
    def _print_summary(results, skipped, spectrum, threshold):
        cols = [k for k in (1, 2, 4, 8) if k in spectrum]
        kmax = max(spectrum)
        w = max([20] + [len(r["name"]) + 2 for r in results])
        width = w + 47 + 8 * (len(cols) + 1)

        def min_k_txt(r):
            return str(r["min_k"]) if r["min_k"] else ">" + str(kmax)

        print("\nSENSITIVITY SCAN  (cosine similarity to the FP32 output; one layer at a time; weights, activations"
              " and norm at k bits,\nweights with the better of 2-sigma clipping and min-max)")
        print(f"{'LAYER':<{w}}{'TYPE':<8}{'SHAPE':<16}{'PARAMS':>10}{'NORM':>6}"
              + "".join(f"{'k=' + str(k):>8}" for k in cols) + f"{'CLIP':>8}{'MIN k':>7}")
        print("-" * width)
        for r in results:
            print(f"{r['name']:<{w}}{r['type']:<8}{r['shape']:<16}{r['params']:>10,}{r['norm']:>6}"
                  + "".join(f"{r['fid'][k]:>8.4f}" for k in cols) + f"{r['clip']:>8.4f}{min_k_txt(r):>7}")
        print("-" * width)
        # most sensitive first: needs more bits; ties (or nobody reaches the threshold) -> lowest fidelity at the highest k
        worst = sorted(results, key=lambda r: (-(r["min_k"] or 99), r["fid"][kmax]))[:3]
        print(f"MIN k = smallest k with cosine >= {threshold}.  CLIP = cosine with ONLY the 2-sigma weight clipping "
              "(no rounding):\na low CLIP means the layer depends on its largest weights (outliers).")
        print("Most sensitive: " + ", ".join(
            f"{r['name']} (k={r['min_k']})" if r["min_k"] else f"{r['name']} ({r['fid'][kmax]:.2f} at k={kmax})"
            for r in worst))
        print(f"{len(results)} layers scanned" + (f", {len(skipped)} skipped (bypassed in the forward pass): "
              + ", ".join(skipped) if skipped else ""))
        print("Note: one layer in isolation, uniform k-bit. Compare layers with each other; "
              "it does not predict the accuracy of the whole network.")

    def scan(self, calibration, detail=False, max_layers=None,
             spectrum=(1, 2, 3, 4, 5, 6, 7, 8), threshold=0.99, espectro=None):
        """
        Scans every Conv2d / Linear of the model with REAL calibration samples.
        Prints one summary row per layer (and, with detail=True, the 5 detail tables of each layer).
        Returns the list of per-layer results (dicts). espectro: old name of spectrum.
        """
        if espectro is not None:
            spectrum = espectro
        modules, inputs = self._capture_inputs(calibration)

        results, skipped, pending, different = [], [], [], []
        for i, (name, module) in enumerate(modules):
            if max_layers is not None and len(results) >= max_layers:
                break
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if name not in inputs:
                skipped.append(name)
                continue
            norm, by_declaration = self._norm_of(name, i, modules)
            if norm is not by_declaration:
                different.append(name)
            results.append(self._measure_layer(name, module, norm, inputs[name], spectrum, threshold))
            pending.append((name, module, norm, inputs[name]))

        self._print_summary(results, skipped, list(spectrum), threshold)
        if different:
            print("[!] Norm paired by execution order (the norm that receives the layer's output), which differs "
                  "from the old declaration-order rule in: " + ", ".join(different))

        if detail:
            for name, module, norm, x_in in pending:
                self.analyze_layer(module, norm, x_in, layer_name=name, spectrum=list(spectrum))
        return results

    def scan_first_layers(self, calibration, n_layers=1):
        """Kept for the old scripts: scans the first `n_layers` layers and prints their 5 detail tables."""
        return self.scan(calibration, detail=True, max_layers=n_layers)

    # ------------------------------------------------------------------
    # SYMBEX binarization, memory of every option and the decision (budget as a ceiling)
    # ------------------------------------------------------------------
    @staticmethod
    def binarize_weights_symbex(weight):
        """sign(w) * alpha, alpha = mean|w| per output channel; 0 -> +1 (the convention of the C engine).
        alpha costs nothing on the chip: it is folded into the integer threshold."""
        alpha = weight.abs().mean(dim=tuple(range(1, weight.dim())), keepdim=True)
        return torch.where(weight >= 0, torch.ones_like(weight), -torch.ones_like(weight)) * alpha

    @staticmethod
    def binarize_activation_symbex(x, channel_dim=1):
        """sign(x - t) * s per channel, with t = median and s = mean|x - t| of that channel; 0 -> +1.
        On the chip the threshold and the scale come from retraining (BN + sign folded); here they are
        estimated from the calibration data, so the result is an ESTIMATE of what binarizing the input costs."""
        channel_dim = channel_dim % x.dim()
        c = x.shape[channel_dim]
        xc = x.movedim(channel_dim, 0).reshape(c, -1)
        t = xc.median(dim=1).values
        s = (xc - t[:, None]).abs().mean(dim=1).clamp(min=1e-12)
        shape = [1] * x.dim()
        shape[channel_dim] = c
        t, s = t.view(*shape), s.view(*shape)
        return torch.where(x - t >= 0, torch.ones_like(x), -torch.ones_like(x)) * s

    @staticmethod
    def _bits(option):
        return 1 if option == "bin" else int(option)

    @staticmethod
    def layer_memory(module, x_in, option, word_bits=8):
        """(flash bytes, RAM bytes) of ONE layer with `option` ("bin" or k bits).
        flash: weights packed at their bits, each output channel's row padded to a whole CPU word
               (word_bits = 8 keeps the old byte-exact count; 32 = how the ESP32 engine stores them),
               + per-output-channel parameters ("bin": one int32 threshold; k bits: scale + shift, two int32)
        RAM:   the layer's input activations for ONE sample, at the same bits."""
        bits = Sensitivity._bits(option)
        channels = module.weight.shape[0]
        per_row = module.weight.numel() // channels * bits
        row = -(-per_row // word_bits) * word_bits
        flash = (channels * row + 7) // 8 + channels * (4 if option == "bin" else 8)
        ram = (x_in[0].numel() * bits + 7) // 8
        return flash, ram

    def _fid_option(self, module, norm, x_in, target, option, quantize_input=True):
        """Cosine fidelity of ONE layer with `option`. k bits: the same measure as scan() (better of the two
        weight rules). "bin": SYMBEX binarization of weights and input; the norm is not quantized because
        BN + sign folds exactly into one integer threshold per channel.
        quantize_input=False: the input is NOT quantized (the first layer of the model: training keeps its input
        untouched, see methods.apply_method / apply_plan), only the weights (and the norm for k bits)."""
        if not quantize_input:
            weight = module.weight.data
            if option == "bin":
                weights_q, k_bn = [self.binarize_weights_symbex(weight)], None
            else:
                weights_q, k_bn = [self.quantize_weight_v3(weight, option)[0], self.quantize_minmax(weight, option)], option
            best = -1.0
            with torch.no_grad():
                for wq in weights_q:
                    output = self.apply_norm_or_bias(self.run_base(x_in, module, wq), module, norm, k_bits_bn=k_bn)
                    best = max(best, F.cosine_similarity(output.flatten(), target.flatten(), dim=0).item())
            return best
        if option != "bin":
            weight = module.weight.data
            return max(self._fid_with_weights(self.quantize_weight_v3(weight, option)[0], x_in, module, norm, target, option),
                       self._fid_with_weights(self.quantize_minmax(weight, option), x_in, module, norm, target, option))
        channel_dim = 1 if isinstance(module, nn.Conv2d) else -1
        with torch.no_grad():
            output = self.apply_norm_or_bias(
                self.run_base(self.binarize_activation_symbex(x_in, channel_dim), module,
                              self.binarize_weights_symbex(module.weight.data)),
                module, norm, k_bits_bn=None)
            return F.cosine_similarity(output.flatten(), target.flatten(), dim=0).item()

    def estimate(self, calibration, ram_kb, flash_kb, budget=0.25, tolerance=0.99,
                 options=("bin", 2, 4, 8), verbose=True, detail=True, word_bits=8,
                 presupuesto=None, tolerancia=None, opciones=None, imprimir=None, detalle=None, palabra_bits=None):
        """
        Every Conv2d / Linear, every option in `options`: fidelity (cosine, one layer at a time), flash and RAM.
        Decision per layer: the option with the FEWEST bits whose fidelity reaches `tolerance`
        (start at 1 bit; raise only when the layer needs it). The hardware budget is a CEILING:
        budget * ram_kb and budget * flash_kb. It never pushes towards more bits.
        detail=True (default): one block per layer (every option with its cosine, flash, RAM and whether it
        meets the tolerance) before the summary table, and usage bars for the budget at the end.
        verbose=False: no printing. word_bits: CPU word size (rows are padded to whole words).
        Returns a dict: rows (per layer), skipped, flash_total, ram_peak, flash_limit, ram_limit, fits, ...
        (plus the old Spanish keys until every caller is translated). Every row: name, type, shape, options
        ({option: fid, flash, ram}), chosen, meets, input_quantized, n_inputs, channels, fan_in.
        All numbers are ESTIMATES before retraining (one layer in isolation, no error accumulation).
        presupuesto / tolerancia / opciones / imprimir / detalle / palabra_bits: old names of the arguments.
        """
        budget = budget if presupuesto is None else presupuesto
        tolerance = tolerance if tolerancia is None else tolerancia
        options = options if opciones is None else opciones
        verbose = verbose if imprimir is None else imprimir
        detail = detail if detalle is None else detalle
        word_bits = word_bits if palabra_bits is None else palabra_bits
        if not (0 < budget <= 1):
            raise ValueError(f"budget must be in (0, 1], got {budget}")
        if ram_kb <= 0 or flash_kb <= 0:
            raise ValueError("ram_kb and flash_kb must be positive")
        options = sorted(options, key=self._bits)
        for op in options:
            if op != "bin" and not (isinstance(op, int) and 1 <= op <= 16):
                raise ValueError(f"unknown option {op!r}: use 'bin' or an integer number of bits (1-16)")
        if calibration.shape[0] < 8:
            print(f"[!] only {calibration.shape[0]} calibration samples: the estimate may be unstable (use 32 or more)")

        modules, inputs = self._capture_inputs(calibration)
        index = {n: i for i, (n, _) in enumerate(modules)}
        first = self._execution_order[0] if self._execution_order else None
        rows = []
        for name in self._execution_order:
            module = dict(modules)[name]
            norm, _ = self._norm_of(name, index[name], modules)
            x_in = inputs[name]
            with torch.no_grad():
                target = self.apply_norm_or_bias(self.run_base(x_in, module, module.weight.data),
                                                 module, norm, k_bits_bn=None)
            per_option = {}
            for op in options:
                flash, ram = self.layer_memory(module, x_in, op, word_bits)
                per_option[op] = dict(fid=self._fid_option(module, norm, x_in, target, op, quantize_input=name != first),
                                      flash=flash, ram=ram)
            chosen = next((op for op in options if per_option[op]["fid"] >= tolerance), options[-1])
            row = dict(name=name, type=type(module).__name__,
                       shape="x".join(str(d) for d in module.weight.shape), options=per_option,
                       chosen=chosen, meets=per_option[chosen]["fid"] >= tolerance,
                       input_quantized=name != first,
                       n_inputs=x_in[0].numel(), channels=module.weight.shape[0],
                       fan_in=module.weight.numel() // module.weight.shape[0])
            # old Spanish keys, same values (kept until every caller is translated)
            row.update(opciones=row["options"], elegida=row["chosen"], cumple=row["meets"],
                       entrada_cuantizada=row["input_quantized"], n_entradas=row["n_inputs"], canales=row["channels"])
            rows.append(row)
        skipped = [n for n, m in modules if isinstance(m, (nn.Conv2d, nn.Linear)) and n not in inputs]
        if not rows:
            raise RuntimeError("no Conv2d / Linear layer ran in the forward pass: nothing to estimate")

        flash_total = sum(r["options"][r["chosen"]]["flash"] for r in rows)
        # peak RAM estimate: a layer's input and output live together; the output is the next layer's input
        rams = [r["options"][r["chosen"]]["ram"] for r in rows]
        ram_peak = max(a + b for a, b in zip(rams, rams[1:] + [0]))
        res = dict(rows=rows, skipped=skipped, flash_total=flash_total, ram_peak=ram_peak,
                   flash_limit=budget * flash_kb * 1024, ram_limit=budget * ram_kb * 1024,
                   budget=budget, tolerance=tolerance, options=options, word_bits=word_bits)
        res["fits"] = flash_total <= res["flash_limit"] and ram_peak <= res["ram_limit"]
        res.update(filas=res["rows"], saltadas=res["skipped"], ram_pico=res["ram_peak"],
                   limite_flash=res["flash_limit"], limite_ram=res["ram_limit"], presupuesto=res["budget"],
                   tolerancia=res["tolerance"], opciones=res["options"], palabra_bits=res["word_bits"],
                   cabe=res["fits"])
        if verbose:
            if detail:
                self._print_estimate_detail(res)
            self._print_estimate(res)
            if detail:
                self._print_budget_bars(res)
        return res

    @staticmethod
    def _print_estimate_detail(res):
        """One block per layer: every option with its cosine, flash, RAM and whether it meets the tolerance."""
        txt = lambda op: "bin" if op == "bin" else f"{op} bits"
        for r in res["rows"]:
            print(f"\n--- Layer: {r['name']} ({r['type']} {r['shape']}) ---"
                  + ("" if r.get("input_quantized", True) else "   [first layer: input kept, only weights quantized]"))
            print(f"{'option':<8}| {'cosine':>8} | {'flash KB':>9} | {'RAM KB':>8} | meets")
            print("-" * 50)
            for op in res["options"]:
                o = r["options"][op]
                mark = "   <- chosen" if op == r["chosen"] else ""
                print(f"{txt(op):<8}| {o['fid']:>8.4f} | {o['flash'] / 1024:>9.2f} | {o['ram'] / 1024:>8.2f} | "
                      f"{'yes' if o['fid'] >= res['tolerance'] else 'no':<5}{mark}")
            if not r["meets"]:
                print(f"[!] no option reaches cosine {res['tolerance']}: the one with the most bits was taken")

    @staticmethod
    def _print_budget_bars(res, width=40):
        """Budget usage as bars: how much of the allowed flash and RAM the chosen options take."""
        print(f"\nBUDGET ({100 * res['budget']:.0f}% of the hardware)")
        for name, used, limit in (("flash", res["flash_total"], res["flash_limit"]),
                                  ("RAM  ", res["ram_peak"], res["ram_limit"])):
            frac = used / limit if limit else 0.0
            full = min(width, int(round(frac * width)))
            bar = "#" * full + "." * (width - full)
            print(f"{name} [{bar}] {used / 1024:8.1f} / {limit / 1024:.1f} KB  ({100 * frac:5.1f}%)"
                  + ("  <- OVER BUDGET" if frac > 1 else ""))

    @staticmethod
    def _print_estimate(res):
        ops = res["options"]
        txt = lambda op: "bin" if op == "bin" else f"{op}b"
        w = max([20] + [len(r["name"]) + 2 for r in res["rows"]])
        print("\nESTIMATE  (cosine per option, one layer at a time; bin = SYMBEX binarization; "
              f"choice = fewest bits with cosine >= {res['tolerance']})")
        print(f"{'LAYER':<{w}}{'TYPE':<8}" + "".join(f"{txt(op):>9}" for op in ops)
              + f"{'CHOICE':>8}{'FLASH KB':>10}{'RAM KB':>9}")
        print("-" * (w + 8 + 9 * len(ops) + 27))
        for r in res["rows"]:
            e = r["options"][r["chosen"]]
            print(f"{r['name']:<{w}}{r['type']:<8}" + "".join(f"{r['options'][op]['fid']:>9.4f}" for op in ops)
                  + f"{txt(r['chosen']) + ('' if r['meets'] else '!'):>8}{e['flash'] / 1024:>10.2f}{e['ram'] / 1024:>9.2f}")
        print("-" * (w + 8 + 9 * len(ops) + 27))
        print(f"flash total {res['flash_total'] / 1024:.1f} KB of {res['flash_limit'] / 1024:.1f} KB allowed  |  "
              f"RAM peak {res['ram_peak'] / 1024:.1f} KB of {res['ram_limit'] / 1024:.1f} KB allowed  "
              f"(budget {100 * res['budget']:.0f}%)  ->  {'FITS' if res['fits'] else 'DOES NOT FIT'}")
        missing = [r["name"] for r in res["rows"] if not r["meets"]]
        if missing:
            print("! = no option reached the tolerance; the most bits were taken: " + ", ".join(missing))
        if res["skipped"]:
            print("skipped (bypassed in the forward pass): " + ", ".join(res["skipped"]))
        print("ESTIMATE before retraining: one layer in isolation; retraining usually recovers part of the loss.")

    # ------------------------------------------------------------------
    # Old Spanish names, kept until every caller is translated (see RENAMES.md)
    # ------------------------------------------------------------------
    aplicar_norm_o_bias = apply_norm_or_bias
    calcular_psnr = compute_psnr
    calcular_entropia_shannon = compute_shannon_entropy
    calcular_ruido_estratificado = compute_stratified_noise
    ejecutar_base = run_base
    medir_combinacion = measure_combination
    analizar_capa = analyze_layer
    binarizar_peso_symbex = binarize_weights_symbex
    binarizar_activacion_symbex = binarize_activation_symbex
    memoria_capa = layer_memory
    estimar = estimate

    def parsear_y_rutear(self, tensor_prueba, escanear_max=1):
        """Old name of scan_first_layers."""
        return self.scan_first_layers(tensor_prueba, escanear_max)


# ----------------------------------------------------------------------
# Self-test:  python sensitivity.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    class _Net(nn.Module):
        """Small CNN that declares its BatchNorms BEFORE its convs (the case the old pairing got wrong)."""
        def __init__(self):
            super().__init__()
            self.bn1, self.c1 = nn.BatchNorm2d(16), nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.bn2, self.c2 = nn.BatchNorm2d(32), nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.c1(x)))
            x = F.relu(self.bn2(self.c2(x)))
            return self.fc(x.flatten(1))

    net = _Net()
    net.train()
    with torch.no_grad():                                   # give the BatchNorms real running statistics
        for _ in range(20):
            net(torch.rand(32, 3, 16, 16))
    # smooth synthetic images (gradients + noise), not torch.randn: only to exercise the code
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
    calib = torch.stack([torch.stack([yy * a, xx * b, (yy + xx) / 2 * c]) for a, b, c in torch.rand(32, 3)])
    calib = calib + 0.05 * torch.rand_like(calib)

    s = Sensitivity(net)
    r = s.scan(calib, spectrum=(1, 2, 4, 8))
    assert len(r) == 3 and all(set(x["fid"]) == {1, 2, 4, 8} for x in r), "scan(): unexpected result"
    assert s._norm_by_layer.get("c1") is net.bn1 and s._norm_by_layer.get("c2") is net.bn2, \
        "execution-order pairing failed"
    e = s.estimate(calib, ram_kb=320, flash_kb=4096, budget=0.25, tolerance=0.95)
    c2 = next(row for row in e["rows"] if row["name"] == "c2")
    assert c2["options"]["bin"]["flash"] == (32 * 16 * 9 + 7) // 8 + 32 * 4, "memory arithmetic (bin)"
    assert c2["options"][8]["flash"] == 32 * 16 * 9 + 32 * 8, "memory arithmetic (8 bits)"
    assert c2["options"]["bin"]["ram"] == (16 * 16 * 16 + 7) // 8, "RAM arithmetic"
    assert all(-1.0001 <= row["options"][op]["fid"] <= 1.0001 for row in e["rows"] for op in e["options"])
    # the old names still give the same thing
    e_old = s.estimar(calib, ram_kb=320, flash_kb=4096, presupuesto=0.25, tolerancia=0.95, imprimir=False)
    assert e_old["filas"][0]["elegida"] == e["rows"][0]["chosen"] and e_old["cabe"] == e["fits"], "old names"
    print("\n[OK] self-test passed: scan(), execution-order norm pairing, estimate() and its memory arithmetic, "
          "and the old names.")
