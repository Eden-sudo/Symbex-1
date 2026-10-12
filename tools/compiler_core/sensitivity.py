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
 7. estimar(): the SYMBEX binarization ("bin": sign(w) * mean|w| per channel, input
    sign(x - t) * s per channel) next to k-bit options, the memory of every option
    (flash and RAM), and the decision: start at 1 bit, raise the bits of a layer only
    when it needs them (tolerance), with the hardware budget as a ceiling.
        Sensitivity(model).estimar(calibration, ram_kb=320, flash_kb=4096,
                                   presupuesto=0.25, tolerancia=0.99)          # detalle=True by default
 8. estimar() keeps the input of the FIRST layer unquantized (only its weights), because training does
    the same (methods.apply_method / apply_plan leave the raw input untouched).
 9. Self-test:  python sensitivity.py
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
        """
        Regla SYMBEX (Actualizada): Recorte 2*std universal.
        """
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
    def aplicar_norm_o_bias(x, target_module, norm_module, k_bits_bn=None):
        if isinstance(norm_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            eps = norm_module.eps
            weight = Sensitivity.quantize_minmax(norm_module.weight.data, k_bits_bn) if k_bits_bn else norm_module.weight.data
            bias = Sensitivity.quantize_minmax(norm_module.bias.data, k_bits_bn) if k_bits_bn else norm_module.bias.data
            escala = weight / torch.sqrt(norm_module.running_var + eps)
            desplazamiento = bias - norm_module.running_mean * escala
            view_shape = [1, -1] + [1] * (x.dim() - 2)
            return x * escala.view(*view_shape) + desplazamiento.view(*view_shape)
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
    def calcular_psnr(salida, referencia):
        mse = ((salida - referencia) ** 2).mean().item()
        if mse < 1e-12:
            return float('inf')
        max_val = referencia.abs().max().item()
        return 10 * np.log10((max_val ** 2) / mse)

    @staticmethod
    def calcular_entropia_shannon(niveles_enteros, k_bits):
        if niveles_enteros is None:
            return None, None, None
        valores, conteos = torch.unique(niveles_enteros, return_counts=True)
        probs = conteos.float() / conteos.sum()
        entropia_real = -(probs * torch.log2(probs)).sum().item()
        entropia_maxima = k_bits
        eficiencia = entropia_real / entropia_maxima if entropia_maxima > 0 else 0
        return entropia_real, entropia_maxima, eficiencia

    @staticmethod
    def calcular_ruido_estratificado(original, reconstruido):
        """
        Ruido de cuantizacion ESTRATIFICADO por magnitud del valor original.

        FIX: la version anterior usaba promedio + un epsilon absoluto
        minusculo (1e-6) -- con salidas de BatchNorm (centradas en cero
        por diseno), el "10% mas chico por magnitud" incluye valores
        genuinamente CASI CERO (1e-7, 1e-8...), no solo "chicos pero
        significativos". Dividir por esas magnitudes casi nulas explota
        a numeros sin sentido (se vieron ratios de millones de %) y el
        PROMEDIO se deja contaminar por esos pocos outliers.

        Fix: mediana (robusta a esos outliers) + epsilon relativo a la
        escala real del tensor (no una constante absoluta).
        """
        original_flat = original.flatten()
        reconstruido_flat = reconstruido.flatten()
        error_abs = (original_flat - reconstruido_flat).abs()

        magnitudes = original_flat.abs()
        orden = torch.argsort(magnitudes)
        n = len(orden)
        n_decimo = max(1, n // 10)

        idx_menor = orden[:n_decimo]
        idx_mayor = orden[-n_decimo:]

        # Epsilon relativo a la escala real del tensor, no absoluto
        eps = magnitudes.mean().item() * 0.01
        eps = max(eps, 1e-9)

        error_rel_menor = (error_abs[idx_menor] / (magnitudes[idx_menor] + eps)).median().item()
        error_rel_mayor = (error_abs[idx_mayor] / (magnitudes[idx_mayor] + eps)).median().item()
        ratio = error_rel_menor / max(error_rel_mayor, 1e-9)

        return error_rel_menor, error_rel_mayor, ratio

    def ejecutar_base(self, x, target_module, peso):
        if isinstance(target_module, nn.Conv2d):
            return F.conv2d(x, peso, bias=None, stride=target_module.stride, padding=target_module.padding)
        elif isinstance(target_module, nn.Linear):
            return F.linear(x, peso, bias=None)

    def medir_combinacion(self, peso_fp32, images_input, target_module, norm_module, target, k_w, k_a, k_bn):
        with torch.no_grad():
            peso_q, _ = self.quantize_weight_v3(peso_fp32, k_w)
            images_q = self.quantize_minmax(images_input, k_a)
            base_out = self.ejecutar_base(images_q, target_module, peso_q)
            out_final = self.aplicar_norm_o_bias(base_out, target_module, norm_module, k_bits_bn=k_bn)
            fid = F.cosine_similarity(out_final.flatten(), target.flatten(), dim=0).item()
            psnr = self.calcular_psnr(out_final, target)
        return fid, psnr

    def analizar_capa(self, target_module, norm_module, x_in, nombre_capa, espectro=(1, 2, 3, 4, 5, 6, 7, 8)):
        """Las 5 tablas principales asegurando operaciones dimensionalmente puras."""
        espectro = list(espectro)
        print(f"\n" + "=" * 100)
        print(f" TABLA DE SENSIBILIDAD EXTENDIDA -- Capa: {nombre_capa}")
        print("=" * 100)

        peso_fp32 = target_module.weight.data.clone()
        params_totales = peso_fp32.numel()
        input_elementos = x_in.numel() // x_in.size(0)

        with torch.no_grad():
            base_ref = self.ejecutar_base(x_in, target_module, peso_fp32)
            target = self.aplicar_norm_o_bias(base_ref, target_module, norm_module, k_bits_bn=None)

        print("\n--- TABLA 1: FIDELIDAD (coseno) ---")
        print(f"{'k':<4} | {'Solo PESO':<12} | {'Solo ACTIV':<12} | {'Solo NORM/Bias':<14} | {'LOS 3':<12}")
        print("-" * 65)

        resultados_por_k = {}
        for k in espectro:
            with torch.no_grad():
                peso_q, niveles_peso = self.quantize_weight_v3(peso_fp32, k)
                x_q = self.quantize_minmax(x_in, k)

                out_peso = self.ejecutar_base(x_in, target_module, peso_q)
                out_solo_peso = self.aplicar_norm_o_bias(out_peso, target_module, norm_module, k_bits_bn=None)
                fid_peso = F.cosine_similarity(out_solo_peso.flatten(), target.flatten(), dim=0).item()

                out_act = self.ejecutar_base(x_q, target_module, peso_fp32)
                out_solo_act = self.aplicar_norm_o_bias(out_act, target_module, norm_module, k_bits_bn=None)
                fid_act = F.cosine_similarity(out_solo_act.flatten(), target.flatten(), dim=0).item()

                out_solo_bn = self.aplicar_norm_o_bias(base_ref, target_module, norm_module, k_bits_bn=k)
                fid_bn = F.cosine_similarity(out_solo_bn.flatten(), target.flatten(), dim=0).item()

                out_all = self.ejecutar_base(x_q, target_module, peso_q)
                out_todos = self.aplicar_norm_o_bias(out_all, target_module, norm_module, k_bits_bn=k)
                fid_todos = F.cosine_similarity(out_todos.flatten(), target.flatten(), dim=0).item()

            resultados_por_k[k] = {'niveles_peso': niveles_peso, 'out_todos': out_todos}
            print(f"{k:<4} | {fid_peso:<12.6f} | {fid_act:<12.6f} | {fid_bn:<14.6f} | {fid_todos:<12.6f}")

        print("\n--- TABLA 2: PSNR (dB) ---")
        print(f"{'k':<4} | {'Solo PESO':<12} | {'Solo ACTIV':<12} | {'Solo NORM/Bias':<14} | {'LOS 3':<12}")
        print("-" * 65)
        for k in espectro:
            with torch.no_grad():
                peso_q, _ = self.quantize_weight_v3(peso_fp32, k)
                x_q = self.quantize_minmax(x_in, k)

                psnr_peso = self.calcular_psnr(self.aplicar_norm_o_bias(self.ejecutar_base(x_in, target_module, peso_q), target_module, norm_module, None), target)
                psnr_act = self.calcular_psnr(self.aplicar_norm_o_bias(self.ejecutar_base(x_q, target_module, peso_fp32), target_module, norm_module, None), target)
                psnr_bn = self.calcular_psnr(self.aplicar_norm_o_bias(base_ref, target_module, norm_module, k), target)
                psnr_todos = self.calcular_psnr(resultados_por_k[k]['out_todos'], target)

            def psnr_str(v): return "inf" if v == float('inf') else f"{v:.2f}"
            print(f"{k:<4} | {psnr_str(psnr_peso):<12} | {psnr_str(psnr_act):<12} | {psnr_str(psnr_bn):<14} | {psnr_str(psnr_todos):<12}")

        print("\n--- TABLA 3: ENTROPIA DE SHANNON (Peso) ---")
        print(f"{'k':<4} | {'Entropia real (bits)':<22} | {'Entropia maxima':<18} | {'Eficiencia de uso':<18}")
        print("-" * 68)
        for k in espectro:
            niveles = resultados_por_k[k]['niveles_peso']
            ent_real, ent_max, eficiencia = self.calcular_entropia_shannon(niveles, k)
            print(f"{k:<4} | {ent_real:<22.4f} | {ent_max:<18.4f} | {eficiencia*100:<17.1f}%")

        print("\n--- TABLA 4: MEMORIA Y VELOCIDAD GENERAL ---")
        print(f"{'k':<4} | {'Peso capa (KB)':<18} | {'RAM activ. (KB)':<18} | {'Vel. proxy k*k (NO medida)':<18}")
        print("-" * 65)
        for k in espectro:
            peso_kb = params_totales * k / 8 / 1024
            ram_kb = input_elementos * k / 8 / 1024
            velocidad_proxy = k * k
            print(f"{k:<4} | {peso_kb:<18.3f} | {ram_kb:<18.3f} | {velocidad_proxy:<18}")

        print("\n--- TABLA 5: RUIDO DE CUANTIZACION ESTRATIFICADO (escenario 'LOS 3') ---")
        print("(¿los valores cerca de cero sufren mas error relativo que los grandes,")
        print(" aunque la fidelidad agregada de la Tabla 1 se vea bien?)")
        print(f"{'k':<4} | {'Error rel. 10% CHICOS':<22} | {'Error rel. 10% GRANDES':<22} | {'Ratio':<10} | Alerta")
        print("-" * 80)
        for k in espectro:
            out_todos = resultados_por_k[k]['out_todos']
            err_chicos, err_grandes, ratio = self.calcular_ruido_estratificado(target, out_todos)
            alerta = "[!] valores chicos sufren desproporcionado" if ratio > 5 else "[+] parejo"
            print(f"{k:<4} | {err_chicos*100:<21.1f}% | {err_grandes*100:<21.1f}% | {ratio:<10.1f}| {alerta}")

    # ------------------------------------------------------------------
    # SCAN: the whole network, one summary row per layer
    # ------------------------------------------------------------------
    def _capturar_entradas(self, calibration):
        """Runs the model once and returns {layer name: the real input that layer received}.
        Also records (in self._orden_ejecucion / self._norm_por_capa) the order in which the layers ran
        and which norm module received each layer's output."""
        entradas = {}
        orden, salida_de = [], {}
        self._norm_por_capa = {}

        def guardar(nombre):
            def hook(module, entrada, salida):
                entradas[nombre] = entrada[0].detach()
                orden.append(nombre)
                salida_de[id(salida)] = nombre
            return hook

        def guardar_norm(modulo_norm):
            def hook(module, entrada, salida):
                capa = salida_de.get(id(entrada[0]))
                if capa is not None and capa not in self._norm_por_capa:
                    self._norm_por_capa[capa] = modulo_norm
            return hook

        modulos = list(self.model.named_modules())
        hooks = [m.register_forward_hook(guardar(n)) for n, m in modulos
                 if isinstance(m, (nn.Conv2d, nn.Linear))]
        hooks += [m.register_forward_hook(guardar_norm(m)) for _, m in modulos
                  if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm))]
        with torch.no_grad():
            self.model(calibration)
        for h in hooks:
            h.remove()
        self._orden_ejecucion = list(dict.fromkeys(orden))
        return modulos, entradas

    def _norm_de(self, nombre, i, modulos):
        """The norm of layer `nombre`: the one that received its output in the forward pass.
        Returns (norm, norm by the old declaration-order rule) so callers can report disagreements."""
        siguiente = modulos[i + 1][1] if (i + 1) < len(modulos) else None
        por_declaracion = siguiente if isinstance(siguiente, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)) else None
        return self._norm_por_capa.get(nombre), por_declaracion

    def _fid_con_peso(self, peso_q, x_in, modulo, norm, target, k):
        """Cosine fidelity when the weights are `peso_q` and activations + norm are quantized at k bits."""
        with torch.no_grad():
            salida = self.aplicar_norm_o_bias(
                self.ejecutar_base(self.quantize_minmax(x_in, k), modulo, peso_q), modulo, norm, k_bits_bn=k)
            return F.cosine_similarity(salida.flatten(), target.flatten(), dim=0).item()

    def _medir_capa(self, nombre, modulo, norm, x_in, espectro, threshold):
        """Cosine fidelity of ONE layer for every k in `espectro` (weights + activations + norm at k bits)."""
        peso = modulo.weight.data.clone()
        with torch.no_grad():
            base = self.ejecutar_base(x_in, modulo, peso)
            target = self.aplicar_norm_o_bias(base, modulo, norm, k_bits_bn=None)
        fid = {}
        for k in espectro:
            # the better of two weight rules: 2*std clipping (good at low k) and plain min-max (good at high k,
            # where clipping the tails of a pretrained layer costs more than the rounding it saves)
            fid[k] = max(self._fid_con_peso(self.quantize_weight_v3(peso, k)[0], x_in, modulo, norm, target, k),
                         self._fid_con_peso(self.quantize_minmax(peso, k), x_in, modulo, norm, target, k))
        # CLIP: weights clipped at mean +- 2*std but NOT rounded. Low = the layer depends on its largest weights.
        media, std = peso.mean(), peso.std().clamp(min=1e-9)
        with torch.no_grad():
            salida = self.aplicar_norm_o_bias(
                self.ejecutar_base(x_in, modulo, peso.clamp(media - 2.0 * std, media + 2.0 * std)),
                modulo, norm, k_bits_bn=None)
            clip = F.cosine_similarity(salida.flatten(), target.flatten(), dim=0).item()
        if isinstance(norm, (nn.BatchNorm2d, nn.BatchNorm1d)):
            norm_txt = "BN"
        elif isinstance(norm, nn.LayerNorm):
            norm_txt = "LN"
        else:
            norm_txt = "bias" if getattr(modulo, 'bias', None) is not None else "-"
        return dict(name=nombre, type=type(modulo).__name__,
                    shape="x".join(str(d) for d in modulo.weight.shape),
                    params=modulo.weight.numel(), norm=norm_txt, fid=fid, clip=clip,
                    min_k=next((k for k in espectro if fid[k] >= threshold), None))

    @staticmethod
    def _imprimir_resumen(resultados, saltadas, espectro, threshold):
        cols = [k for k in (1, 2, 4, 8) if k in espectro]
        kmax = max(espectro)
        w = max([20] + [len(r["name"]) + 2 for r in resultados])
        ancho = w + 47 + 8 * (len(cols) + 1)

        def min_k_txt(r):
            return str(r["min_k"]) if r["min_k"] else ">" + str(kmax)

        print("\nSENSITIVITY SCAN  (cosine similarity to the FP32 output; one layer at a time; weights, activations"
              " and norm at k bits,\nweights with the better of 2-sigma clipping and min-max)")
        print(f"{'LAYER':<{w}}{'TYPE':<8}{'SHAPE':<16}{'PARAMS':>10}{'NORM':>6}"
              + "".join(f"{'k=' + str(k):>8}" for k in cols) + f"{'CLIP':>8}{'MIN k':>7}")
        print("-" * ancho)
        for r in resultados:
            print(f"{r['name']:<{w}}{r['type']:<8}{r['shape']:<16}{r['params']:>10,}{r['norm']:>6}"
                  + "".join(f"{r['fid'][k]:>8.4f}" for k in cols) + f"{r['clip']:>8.4f}{min_k_txt(r):>7}")
        print("-" * ancho)
        # most sensitive first: needs more bits; ties (or nobody reaches the threshold) -> lowest fidelity at the highest k
        peor = sorted(resultados, key=lambda r: (-(r["min_k"] or 99), r["fid"][kmax]))[:3]
        print(f"MIN k = smallest k with cosine >= {threshold}.  CLIP = cosine with ONLY the 2-sigma weight clipping "
              "(no rounding):\na low CLIP means the layer depends on its largest weights (outliers).")
        print("Most sensitive: " + ", ".join(
            f"{r['name']} (k={r['min_k']})" if r["min_k"] else f"{r['name']} ({r['fid'][kmax]:.2f} at k={kmax})"
            for r in peor))
        print(f"{len(resultados)} layers scanned" + (f", {len(saltadas)} skipped (bypassed in the forward pass): "
              + ", ".join(saltadas) if saltadas else ""))
        print("Note: one layer in isolation, uniform k-bit. Compare layers with each other; "
              "it does not predict the accuracy of the whole network.")

    def scan(self, calibration, detail=False, max_layers=None,
             espectro=(1, 2, 3, 4, 5, 6, 7, 8), threshold=0.99):
        """
        Scans every Conv2d / Linear of the model with REAL calibration samples.
        Prints one summary row per layer (and, with detail=True, the 5 detail tables of each layer).
        Returns the list of per-layer results (dicts).
        """
        modulos, entradas = self._capturar_entradas(calibration)

        resultados, saltadas, pendientes, distintas = [], [], [], []
        for i, (nombre, modulo) in enumerate(modulos):
            if max_layers is not None and len(resultados) >= max_layers:
                break
            if not isinstance(modulo, (nn.Conv2d, nn.Linear)):
                continue
            if nombre not in entradas:
                saltadas.append(nombre)
                continue
            norm, por_declaracion = self._norm_de(nombre, i, modulos)
            if norm is not por_declaracion:
                distintas.append(nombre)
            resultados.append(self._medir_capa(nombre, modulo, norm, entradas[nombre], espectro, threshold))
            pendientes.append((nombre, modulo, norm, entradas[nombre]))

        self._imprimir_resumen(resultados, saltadas, list(espectro), threshold)
        if distintas:
            print("[!] Norm paired by execution order (the norm that receives the layer's output), which differs "
                  "from the old declaration-order rule in: " + ", ".join(distintas))

        if detail:
            for nombre, modulo, norm, x_in in pendientes:
                self.analizar_capa(modulo, norm, x_in, nombre_capa=nombre, espectro=list(espectro))
        return resultados

    def parsear_y_rutear(self, tensor_prueba, escanear_max=1):
        """Kept for the old scripts: scans the first `escanear_max` layers and prints their 5 detail tables."""
        return self.scan(tensor_prueba, detail=True, max_layers=escanear_max)

    # ------------------------------------------------------------------
    # SYMBEX binarization, memory of every option and the decision (budget as a ceiling)
    # ------------------------------------------------------------------
    @staticmethod
    def binarizar_peso_symbex(peso):
        """sign(w) * alpha, alpha = mean|w| per output channel; 0 -> +1 (the convention of the C engine).
        alpha costs nothing on the chip: it is folded into the integer threshold."""
        alpha = peso.abs().mean(dim=tuple(range(1, peso.dim())), keepdim=True)
        return torch.where(peso >= 0, torch.ones_like(peso), -torch.ones_like(peso)) * alpha

    @staticmethod
    def binarizar_activacion_symbex(x, dim_canal=1):
        """sign(x - t) * s per channel, with t = median and s = mean|x - t| of that channel; 0 -> +1.
        On the chip the threshold and the scale come from retraining (BN + sign folded); here they are
        estimated from the calibration data, so the result is an ESTIMATE of what binarizing the input costs."""
        dim_canal = dim_canal % x.dim()
        c = x.shape[dim_canal]
        xc = x.movedim(dim_canal, 0).reshape(c, -1)
        t = xc.median(dim=1).values
        s = (xc - t[:, None]).abs().mean(dim=1).clamp(min=1e-12)
        forma = [1] * x.dim()
        forma[dim_canal] = c
        t, s = t.view(*forma), s.view(*forma)
        return torch.where(x - t >= 0, torch.ones_like(x), -torch.ones_like(x)) * s

    @staticmethod
    def _bits(opcion):
        return 1 if opcion == "bin" else int(opcion)

    @staticmethod
    def memoria_capa(modulo, x_in, opcion, palabra_bits=8):
        """(flash bytes, RAM bytes) of ONE layer with `opcion` ("bin" or k bits).
        flash: weights packed at their bits, each output channel's row padded to a whole CPU word
               (palabra_bits = 8 keeps the old byte-exact count; 32 = how the ESP32 engine stores them),
               + per-output-channel parameters ("bin": one int32 threshold; k bits: scale + shift, two int32)
        RAM:   the layer's input activations for ONE sample, at the same bits."""
        bits = Sensitivity._bits(opcion)
        canales = modulo.weight.shape[0]
        por_fila = modulo.weight.numel() // canales * bits
        fila = -(-por_fila // palabra_bits) * palabra_bits
        flash = (canales * fila + 7) // 8 + canales * (4 if opcion == "bin" else 8)
        ram = (x_in[0].numel() * bits + 7) // 8
        return flash, ram

    def _fid_opcion(self, modulo, norm, x_in, target, opcion, entrada=True):
        """Cosine fidelity of ONE layer with `opcion`. k bits: the same measure as scan() (better of the two
        weight rules). "bin": SYMBEX binarization of weights and input; the norm is not quantized because
        BN + sign folds exactly into one integer threshold per channel.
        entrada=False: the input is NOT quantized (the first layer of the model: training keeps its input
        untouched, see methods.apply_method / apply_plan), only the weights (and the norm for k bits)."""
        if not entrada:
            peso = modulo.weight.data
            if opcion == "bin":
                pesos_q, k_bn = [self.binarizar_peso_symbex(peso)], None
            else:
                pesos_q, k_bn = [self.quantize_weight_v3(peso, opcion)[0], self.quantize_minmax(peso, opcion)], opcion
            mejor = -1.0
            with torch.no_grad():
                for wq in pesos_q:
                    salida = self.aplicar_norm_o_bias(self.ejecutar_base(x_in, modulo, wq), modulo, norm, k_bits_bn=k_bn)
                    mejor = max(mejor, F.cosine_similarity(salida.flatten(), target.flatten(), dim=0).item())
            return mejor
        if opcion != "bin":
            peso = modulo.weight.data
            return max(self._fid_con_peso(self.quantize_weight_v3(peso, opcion)[0], x_in, modulo, norm, target, opcion),
                       self._fid_con_peso(self.quantize_minmax(peso, opcion), x_in, modulo, norm, target, opcion))
        dim_canal = 1 if isinstance(modulo, nn.Conv2d) else -1
        with torch.no_grad():
            salida = self.aplicar_norm_o_bias(
                self.ejecutar_base(self.binarizar_activacion_symbex(x_in, dim_canal), modulo,
                                   self.binarizar_peso_symbex(modulo.weight.data)),
                modulo, norm, k_bits_bn=None)
            return F.cosine_similarity(salida.flatten(), target.flatten(), dim=0).item()

    def estimar(self, calibration, ram_kb, flash_kb, presupuesto=0.25, tolerancia=0.99,
                opciones=("bin", 2, 4, 8), imprimir=True, detalle=True, palabra_bits=8):
        """
        Every Conv2d / Linear, every option in `opciones`: fidelity (cosine, one layer at a time), flash and RAM.
        Decision per layer: the option with the FEWEST bits whose fidelity reaches `tolerancia`
        (start at 1 bit; raise only when the layer needs it). The hardware budget is a CEILING:
        presupuesto * ram_kb and presupuesto * flash_kb. It never pushes towards more bits.
        detalle=True (default): one block per layer (every option with its cosine, flash, RAM and whether it
        meets the tolerance) before the summary table, and usage bars for the budget at the end.
        Returns a dict: filas (per layer), flash_total, ram_pico, limite_flash, limite_ram, cabe.
        All numbers are ESTIMATES before retraining (one layer in isolation, no error accumulation).
        """
        if not (0 < presupuesto <= 1):
            raise ValueError(f"presupuesto must be in (0, 1], got {presupuesto}")
        if ram_kb <= 0 or flash_kb <= 0:
            raise ValueError("ram_kb and flash_kb must be positive")
        opciones = sorted(opciones, key=self._bits)
        for op in opciones:
            if op != "bin" and not (isinstance(op, int) and 1 <= op <= 16):
                raise ValueError(f"unknown option {op!r}: use 'bin' or an integer number of bits (1-16)")
        if calibration.shape[0] < 8:
            print(f"[!] only {calibration.shape[0]} calibration samples: the estimate may be unstable (use 32 or more)")

        modulos, entradas = self._capturar_entradas(calibration)
        indice = {n: i for i, (n, _) in enumerate(modulos)}
        primera = self._orden_ejecucion[0] if self._orden_ejecucion else None
        filas, saltadas = [], []
        for nombre in self._orden_ejecucion:
            modulo = dict(modulos)[nombre]
            norm, _ = self._norm_de(nombre, indice[nombre], modulos)
            x_in = entradas[nombre]
            with torch.no_grad():
                target = self.aplicar_norm_o_bias(self.ejecutar_base(x_in, modulo, modulo.weight.data),
                                                  modulo, norm, k_bits_bn=None)
            por_opcion = {}
            for op in opciones:
                flash, ram = self.memoria_capa(modulo, x_in, op, palabra_bits)
                por_opcion[op] = dict(fid=self._fid_opcion(modulo, norm, x_in, target, op, entrada=nombre != primera),
                                      flash=flash, ram=ram)
            elegida = next((op for op in opciones if por_opcion[op]["fid"] >= tolerancia), opciones[-1])
            filas.append(dict(name=nombre, type=type(modulo).__name__,
                              shape="x".join(str(d) for d in modulo.weight.shape), opciones=por_opcion,
                              elegida=elegida, cumple=por_opcion[elegida]["fid"] >= tolerancia,
                              entrada_cuantizada=nombre != primera,
                              n_entradas=x_in[0].numel(), canales=modulo.weight.shape[0],
                              fan_in=modulo.weight.numel() // modulo.weight.shape[0]))
        saltadas = [n for n, m in modulos if isinstance(m, (nn.Conv2d, nn.Linear)) and n not in entradas]
        if not filas:
            raise RuntimeError("no Conv2d / Linear layer ran in the forward pass: nothing to estimate")

        flash_total = sum(f["opciones"][f["elegida"]]["flash"] for f in filas)
        # peak RAM estimate: a layer's input and output live together; the output is the next layer's input
        rams = [f["opciones"][f["elegida"]]["ram"] for f in filas]
        ram_pico = max(a + b for a, b in zip(rams, rams[1:] + [0]))
        res = dict(filas=filas, saltadas=saltadas, flash_total=flash_total, ram_pico=ram_pico,
                   limite_flash=presupuesto * flash_kb * 1024, limite_ram=presupuesto * ram_kb * 1024,
                   presupuesto=presupuesto, tolerancia=tolerancia, opciones=opciones, palabra_bits=palabra_bits)
        res["cabe"] = flash_total <= res["limite_flash"] and ram_pico <= res["limite_ram"]
        if imprimir:
            if detalle:
                self._imprimir_detalle_estimacion(res)
            self._imprimir_estimacion(res)
            if detalle:
                self._imprimir_barras_presupuesto(res)
        return res

    @staticmethod
    def _imprimir_detalle_estimacion(res):
        """One block per layer: every option with its cosine, flash, RAM and whether it meets the tolerance."""
        txt = lambda op: "bin" if op == "bin" else f"{op} bits"
        for f in res["filas"]:
            print(f"\n--- Layer: {f['name']} ({f['type']} {f['shape']}) ---"
                  + ("" if f.get("entrada_cuantizada", True) else "   [first layer: input kept, only weights quantized]"))
            print(f"{'option':<8}| {'cosine':>8} | {'flash KB':>9} | {'RAM KB':>8} | meets")
            print("-" * 50)
            for op in res["opciones"]:
                o = f["opciones"][op]
                marca = "   <- chosen" if op == f["elegida"] else ""
                print(f"{txt(op):<8}| {o['fid']:>8.4f} | {o['flash'] / 1024:>9.2f} | {o['ram'] / 1024:>8.2f} | "
                      f"{'yes' if o['fid'] >= res['tolerancia'] else 'no':<5}{marca}")
            if not f["cumple"]:
                print(f"[!] no option reaches cosine {res['tolerancia']}: the one with the most bits was taken")

    @staticmethod
    def _imprimir_barras_presupuesto(res, ancho=40):
        """Budget usage as bars: how much of the allowed flash and RAM the chosen options take."""
        print(f"\nBUDGET ({100 * res['presupuesto']:.0f}% of the hardware)")
        for nombre, usado, limite in (("flash", res["flash_total"], res["limite_flash"]),
                                      ("RAM  ", res["ram_pico"], res["limite_ram"])):
            frac = usado / limite if limite else 0.0
            llenas = min(ancho, int(round(frac * ancho)))
            barra = "#" * llenas + "." * (ancho - llenas)
            print(f"{nombre} [{barra}] {usado / 1024:8.1f} / {limite / 1024:.1f} KB  ({100 * frac:5.1f}%)"
                  + ("  <- OVER BUDGET" if frac > 1 else ""))

    @staticmethod
    def _imprimir_estimacion(res):
        ops = res["opciones"]
        txt = lambda op: "bin" if op == "bin" else f"{op}b"
        w = max([20] + [len(f["name"]) + 2 for f in res["filas"]])
        print("\nESTIMATE  (cosine per option, one layer at a time; bin = SYMBEX binarization; "
              f"choice = fewest bits with cosine >= {res['tolerancia']})")
        print(f"{'LAYER':<{w}}{'TYPE':<8}" + "".join(f"{txt(op):>9}" for op in ops)
              + f"{'CHOICE':>8}{'FLASH KB':>10}{'RAM KB':>9}")
        print("-" * (w + 8 + 9 * len(ops) + 27))
        for f in res["filas"]:
            e = f["opciones"][f["elegida"]]
            print(f"{f['name']:<{w}}{f['type']:<8}" + "".join(f"{f['opciones'][op]['fid']:>9.4f}" for op in ops)
                  + f"{txt(f['elegida']) + ('' if f['cumple'] else '!'):>8}{e['flash'] / 1024:>10.2f}{e['ram'] / 1024:>9.2f}")
        print("-" * (w + 8 + 9 * len(ops) + 27))
        print(f"flash total {res['flash_total'] / 1024:.1f} KB of {res['limite_flash'] / 1024:.1f} KB allowed  |  "
              f"RAM peak {res['ram_pico'] / 1024:.1f} KB of {res['limite_ram'] / 1024:.1f} KB allowed  "
              f"(budget {100 * res['presupuesto']:.0f}%)  ->  {'FITS' if res['cabe'] else 'DOES NOT FIT'}")
        sin = [f["name"] for f in res["filas"] if not f["cumple"]]
        if sin:
            print("! = no option reached the tolerance; the most bits were taken: " + ", ".join(sin))
        if res["saltadas"]:
            print("skipped (bypassed in the forward pass): " + ", ".join(res["saltadas"]))
        print("ESTIMATE before retraining: one layer in isolation; retraining usually recovers part of the loss.")


# ----------------------------------------------------------------------
# Self-test:  python sensitivity.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    class _Red(nn.Module):
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

    red = _Red()
    red.train()
    with torch.no_grad():                                   # give the BatchNorms real running statistics
        for _ in range(20):
            red(torch.rand(32, 3, 16, 16))
    # smooth synthetic images (gradients + noise), not torch.randn: only to exercise the code
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
    calib = torch.stack([torch.stack([yy * a, xx * b, (yy + xx) / 2 * c]) for a, b, c in torch.rand(32, 3)])
    calib = calib + 0.05 * torch.rand_like(calib)

    s = Sensitivity(red)
    r = s.scan(calib, espectro=(1, 2, 4, 8))
    assert len(r) == 3 and all(set(x["fid"]) == {1, 2, 4, 8} for x in r), "scan(): unexpected result"
    assert s._norm_por_capa.get("c1") is red.bn1 and s._norm_por_capa.get("c2") is red.bn2, \
        "execution-order pairing failed"
    e = s.estimar(calib, ram_kb=320, flash_kb=4096, presupuesto=0.25, tolerancia=0.95)
    c2 = next(f for f in e["filas"] if f["name"] == "c2")
    assert c2["opciones"]["bin"]["flash"] == (32 * 16 * 9 + 7) // 8 + 32 * 4, "memory arithmetic (bin)"
    assert c2["opciones"][8]["flash"] == 32 * 16 * 9 + 32 * 8, "memory arithmetic (8 bits)"
    assert c2["opciones"]["bin"]["ram"] == (16 * 16 * 16 + 7) // 8, "RAM arithmetic"
    assert all(-1.0001 <= f["opciones"][op]["fid"] <= 1.0001 for f in e["filas"] for op in e["opciones"])
    print("\n[OK] self-test passed: scan(), execution-order norm pairing, estimar() and its memory arithmetic.")
