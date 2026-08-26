"""
SYMBEX-1 Compiler v3.2 — QAT Estabilizado, Auto-M con Safeguards y Data Sheet Fidedigno.
Uso: python tools/symbex_compiler.py --epochs 150 --expansion 1 --auto_m --out_dir lib_symbex/src
"""

import os
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Fijar semillas para reproducibilidad absoluta
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =====================================================================
# 1. ACTIVACIÓN Y CAPAS QAT DE SYMBEX
# =====================================================================

class BipolarSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Empate en 0.0 va a -1.0 (Alineado con el simulador C++)
        return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1.0] = 0
        return grad_input

class BipolarStepSTE(nn.Module):
    def forward(self, x):
        return BipolarSTE.apply(x)

# --- MOTOR 1: CAPA FEED-FORWARD ---
class SymbexVotingPool(nn.Module):
    def __init__(self, in_features, out_features, expansion_factor=1, k_bits=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.M = max(1, expansion_factor)
        self.k_bits = k_bits
        self.momentum = 0.1
        
        self.weight = nn.Parameter(torch.empty(self.M, out_features, in_features))
        for m in range(self.M):
            nn.init.kaiming_uniform_(self.weight[m], a=math.sqrt(5))
            
        self.register_buffer('running_mean', torch.zeros(self.M))
        self.register_buffer('running_std', torch.ones(self.M))
        self.register_buffer('running_W_max', torch.ones(self.M))
        self.register_buffer('initialized', torch.tensor(False))
        
        self.output_scale = nn.Parameter(torch.tensor(0.05))

    def _quantize_weights(self, W, m):
        levels = (2 ** self.k_bits) - 1   
        
        if self.training:
            with torch.no_grad():
                cur_mean = torch.mean(W)
                cur_std = torch.std(W).clamp(min=1e-9)
                
                if not self.initialized:
                    self.running_mean[m].copy_(cur_mean)
                    self.running_std[m].copy_(cur_std)
                else:
                    self.running_mean[m].mul_(1 - self.momentum).add_(self.momentum * cur_mean)
                    self.running_std[m].mul_(1 - self.momentum).add_(self.momentum * cur_std)
                
        mean = self.running_mean[m].clone()
        std = self.running_std[m].clone().clamp(min=1e-9)
        
        # [MITIGACIÓN]: Piso mínimo al umbral para evitar falsos outliers en capas muertas
        threshold = torch.clamp(2.0 * std, min=1e-4)
        
        outlier_mask = (torch.abs(W - mean) > threshold).detach()
        W_core = torch.clamp(W, mean - threshold, mean + threshold)
        
        if self.training:
            with torch.no_grad():
                cur_W_max = torch.max(torch.abs(W_core)).clamp(min=1e-9)
                if not self.initialized:
                    self.running_W_max[m].copy_(cur_W_max)
                else:
                    self.running_W_max[m].mul_(1 - self.momentum).add_(self.momentum * cur_W_max)
                    
        W_max = self.running_W_max[m].clone().clamp(min=1e-9)
        
        W_scaled = (W_core / W_max) * (levels / 2.0) + (levels / 2.0)
        W_quant = torch.round(W_scaled) - W_scaled.detach() + W_scaled   
        W_quant = torch.clamp(W_quant, 0, levels)
        
        W_reconstructed = 2.0 * W_quant - levels
        
        if outlier_mask.any():
            outlier_vals = W * outlier_mask.float()
            sum_abs = torch.sum(torch.abs(outlier_vals), dim=1, keepdim=True)
            count = torch.sum(outlier_mask.float(), dim=1, keepdim=True).clamp(min=1)
            
            outlier_mag_float = sum_abs / count
            scaled_mag = (outlier_mag_float / W_max) * levels
            scaled_mag = torch.clamp(scaled_mag, 0, levels * 3.0)
            
            outlier_mag_quant = torch.round(scaled_mag) - scaled_mag.detach() + scaled_mag
            sign_msb = torch.where(W_quant >= (levels + 1)/2, 1.0, -1.0).detach()
            
            W_reconstructed = torch.where(
                outlier_mask,
                W_reconstructed + (outlier_mag_quant * sign_msb),
                W_reconstructed
            )
            
        return W_reconstructed

    def forward(self, x):
        votes = []
        for m in range(self.M):
            W_rec = self._quantize_weights(self.weight[m], m)
            votes.append(nn.functional.linear(x, W_rec))
            
        if self.training and not self.initialized:
            self.initialized.fill_(True)
            
        stacked = torch.stack(votes, dim=0)
        safe_scale = torch.clamp(self.output_scale, min=1e-4)
        return stacked.sum(dim=0) * safe_scale

# --- CLASE EN DESARROLLO (No expuesta en el CLI todavía) ---
class SymbexRecurrentPool(SymbexVotingPool):
    def __init__(self, in_features, hidden_features, expansion_factor=1, k_bits=3):
        super().__init__(in_features, hidden_features, expansion_factor, k_bits)
        self.hidden_features = hidden_features
        self.weight_hh = nn.Parameter(torch.empty(self.M, hidden_features, hidden_features))
        
        for m in range(self.M):
            nn.init.kaiming_uniform_(self.weight_hh[m], a=math.sqrt(5))
            
        self.register_buffer('running_mean_hh', torch.zeros(self.M))
        self.register_buffer('running_std_hh', torch.ones(self.M))
        self.register_buffer('running_W_max_hh', torch.ones(self.M))


# =====================================================================
# 2. ESTIMADOR DE TOPOLOGÍA Y CONSTRUCTOR
# =====================================================================

class SymbexTopologyEstimator:
    def __init__(self, k_bits=3, max_expansion=4):
        self.k_bits = k_bits
        self.max_expansion = max_expansion

def build_student(teacher, estimator, verbose=False):
    layers = []
    for name, module in teacher.named_modules():
        if isinstance(module, nn.Linear):
            M = estimator.max_expansion   
            if verbose:
                print(f"[*] Capa {name}: {module.in_features}->{module.out_features} | M: {M}")
                
            student_layer = SymbexVotingPool(module.in_features, module.out_features, M, estimator.k_bits)
            
            with torch.no_grad():
                for m in range(M):
                    noise = torch.randn_like(module.weight) * 0.01
                    student_layer.weight[m].copy_(module.weight + noise)
                    
            layers.append(student_layer)
        elif isinstance(module, BipolarStepSTE):
            layers.append(BipolarStepSTE())
    return nn.Sequential(*layers)

# =====================================================================
# 3. SIMULADOR Y EXPORTADOR
# =====================================================================

def simulate_cpp_inference(student, x_bipolar, k_bits=3):
    current = x_bipolar.detach().cpu().numpy().astype(np.float32)
    levels = (2 ** k_bits) - 1   
    
    for layer in student:
        if isinstance(layer, SymbexVotingPool):
            votes_sum = np.zeros((current.shape[0], layer.out_features), dtype=np.float32)
            for m in range(layer.M):
                W_master = layer.weight[m].detach().cpu().numpy().astype(np.float32)
                
                mean = layer.running_mean[m].item()
                std = layer.running_std[m].item()
                W_max = layer.running_W_max[m].item()
                
                threshold = max(2.0 * std, 1e-4)
                
                outlier_mask = np.abs(W_master - mean) > threshold
                W_core = np.clip(W_master, mean - threshold, mean + threshold)
                
                W_scaled = (W_core / W_max) * (levels / 2.0) + (levels / 2.0)
                W_quant = np.clip(np.round(W_scaled), 0, levels).astype(np.float32)
                
                W_reconstructed = 2.0 * W_quant - levels
                
                if outlier_mask.any():
                    sign_msb = np.where(W_quant >= (levels + 1)/2, 1.0, -1.0).astype(np.float32)
                    for n in range(layer.out_features):
                        neuron_outliers = W_master[n][outlier_mask[n]]
                        if len(neuron_outliers) > 0:
                            mag_float = np.mean(np.abs(neuron_outliers))
                            mag_quant = np.clip(np.round((mag_float / W_max) * levels), 0, levels * 3)
                            W_reconstructed[n, outlier_mask[n]] += mag_quant * sign_msb[n, outlier_mask[n]]
                            
                votes_sum += current @ W_reconstructed.T
            current = votes_sum
            
        elif isinstance(layer, BipolarStepSTE):
            current = np.where(current > 0, 1.0, -1.0).astype(np.float32)
    return current

def export_layer_to_arrays(weight_np, mean, std, W_max, k_bits=3):
    out_features, in_features = weight_np.shape
    
    threshold = max(2.0 * std, 1e-4)
    
    outlier_mask = np.abs(weight_np - mean) > threshold
    W_core = np.clip(weight_np, mean - threshold, mean + threshold)
    
    levels = (2 ** k_bits) - 1   
    W_scaled = (W_core / W_max) * (levels / 2.0) + (levels / 2.0)
    W_quant = np.clip(np.round(W_scaled), 0, levels).astype(np.uint8)

    bytes_per_neuron = (in_features + 7) // 8
    planes = [[] for _ in range(k_bits)]
    outl_array = []
    outl_magnitudes = []

    for n in range(out_features):
        neuron_outliers = weight_np[n][outlier_mask[n]]
        if len(neuron_outliers) > 0:
            mag_float = np.mean(np.abs(neuron_outliers))
            mag_quant = int(np.clip(np.round((mag_float / W_max) * levels), 0, levels * 3))
        else:
            mag_quant = 0
            
        outl_magnitudes.append(mag_quant)

        for b in range(bytes_per_neuron):
            plane_bytes = [0] * k_bits
            outl_byte = 0
            for bit_idx in range(8):
                weight_idx = b * 8 + bit_idx
                if weight_idx < in_features:
                    val = int(W_quant[n, weight_idx])
                    for k in range(k_bits):
                        if (val >> (k_bits - 1 - k)) & 1:
                            plane_bytes[k] |= (1 << (7 - bit_idx))
                    if outlier_mask[n, weight_idx]:
                        outl_byte |= (1 << (7 - bit_idx))
            
            for k in range(k_bits):
                planes[k].append(plane_bytes[k])
            outl_array.append(outl_byte)

    return {
        "planes": planes,
        "outliers": outl_array,
        "outlier_magnitudes": outl_magnitudes,
        "in_features": in_features,
        "out_features": out_features,
        "params": in_features * out_features
    }

def export_model_to_symbex_h(student, filepath, k_bits=3):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    MAX_K_BITS = 4
    
    with open(filepath, "w") as f:
        f.write("#ifndef SYMBEX_WEIGHTS_H\n#define SYMBEX_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n")
        f.write("#include \"SymbexNetwork.h\"\n\n")
        f.write("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
        
        layer_instances = []
        total_bytes = 0
        total_params = 0
        layer_counter = 0
        
        for layer in student:
            if isinstance(layer, SymbexVotingPool):
                in_f = layer.in_features
                out_f = layer.out_features
                M = layer.M
                
                subs_names = []
                
                for m in range(M):
                    w = layer.weight[m].detach().cpu().numpy()
                    mean = layer.running_mean[m].item()
                    std = layer.running_std[m].item()
                    w_max = layer.running_W_max[m].item()
                    
                    exp = export_layer_to_arrays(w, mean, std, w_max, k_bits)
                    
                    sub_bytes = 0
                    bit_names = []
                    
                    for k, plane in enumerate(exp["planes"]):
                        name = f"layer{layer_counter}_m{m}_bit{k}"
                        bit_names.append(name)
                        f.write(f"static const uint8_t {name}[{len(plane)}] PROGMEM = {{\n   ")
                        f.write(", ".join(f"0x{v:02X}" for v in plane))
                        f.write("\n};\n\n")
                        sub_bytes += len(plane)
                    
                    outl_name = f"layer{layer_counter}_m{m}_outliers"
                    f.write(f"static const uint8_t {outl_name}[{len(exp['outliers'])}] PROGMEM = {{")
                    f.write(", ".join(f"0x{v:02X}" for v in exp["outliers"]))
                    f.write("};\n\n")
                    sub_bytes += len(exp["outliers"])
                    
                    mag_name = f"layer{layer_counter}_m{m}_outlier_mag"
                    f.write(f"static const int8_t {mag_name}[{len(exp['outlier_magnitudes'])}] PROGMEM = {{")
                    f.write(", ".join(map(str, exp["outlier_magnitudes"])))
                    f.write("};\n\n")
                    
                    while len(bit_names) < MAX_K_BITS:
                        bit_names.append("NULL")
                        
                    planes_str = "{" + ", ".join(bit_names) + "}"
                    subs_names.append(f"    {{ {planes_str}, {outl_name}, {mag_name} }}")
                    
                    total_bytes += sub_bytes
                    total_params += (in_f * out_f)
                    
                f.write(f"// --- ESTRUCTURA DE LA CAPA {layer_counter} ---\n")
                f.write(f"static const SymbexSubLayer layer{layer_counter}_subs[{M}] = {{\n")
                f.write(",\n".join(subs_names))
                f.write("\n};\n")
                
                f.write(f"static SymbexLayer symbex_layer_{layer_counter}({in_f}, {out_f}, {M}, {k_bits}, layer{layer_counter}_subs);\n\n")
                
                layer_instances.append(f"symbex_layer_{layer_counter}")
                layer_counter += 1
        
        f.write("// --- RED ARMADA AUTOMÁTICAMENTE ---\n")
        f.write("static SymbexNetwork symbex_net;\n\n")
        f.write("static inline void symbex_init() {\n")
        for inst in layer_instances:
            f.write(f"    symbex_net.add_layer(&{inst});\n")
        f.write("}\n\n")
        
        f.write("#endif // SYMBEX_WEIGHTS_H\n")
        
    return total_bytes, total_params

# =====================================================================
# 4. RUTINA PRINCIPAL
# =====================================================================

def main(args):
    # [MITIGACIÓN]: Falla temprano y con un mensaje claro si el usuario pide algo que rompe C++
    assert 1 <= args.k_bits <= 4, f"[!] ERROR: k_bits debe estar entre 1 y 4 (límite físico MAX_K_BITS del motor C++). Recibido: {args.k_bits}"

    print(f"[*] Cargando Dataset (Clases 0-{args.classes - 1})...")
    digits = load_digits()
    mask = digits.target < args.classes
    X_all = np.where(digits.data[mask] > 8, 1.0, -1.0)
    y_all = digits.target[mask]

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    print("\n[*] 1. Entrenando Profesor (FP32)...")
    teacher = nn.Sequential(
        nn.Linear(X_all.shape[1], args.hidden, bias=False),
        BipolarStepSTE(),   
        nn.Linear(args.hidden, args.classes, bias=False),
    )
    
    opt = torch.optim.Adam(teacher.parameters(), lr=0.005)
    ce_crit = nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad()
        loss = ce_crit(teacher(X_train), y_train)
        loss.backward()
        opt.step()

    teacher.eval()
    with torch.no_grad():
        acc_test = (torch.argmax(teacher(X_test), 1) == y_test).float().mean().item() * 100
    print(f"[+] Precisión FP32 (Datos invisibles): {acc_test:.2f}%")

    current_m = 1 if args.auto_m else args.expansion
    success = False
    
    while current_m <= 8 and not success:
        print(f"\n[*] 2. Destilando Estudiante (K={args.k_bits}, M={current_m})...")
        estimator = SymbexTopologyEstimator(k_bits=args.k_bits, max_expansion=current_m)
        student = build_student(teacher, estimator, verbose=False)
        
        s_opt = torch.optim.Adam(student.parameters(), lr=0.001)
        T, alpha = 4.0, 0.85
        
        for epoch in range(args.epochs):
            student.train()
            s_opt.zero_grad()
            with torch.no_grad():
                t_out = teacher(X_train)
            s_out = student(X_train)
            
            loss_kl = nn.KLDivLoss(reduction='batchmean')(
                nn.functional.log_softmax(s_out / T, dim=1),
                nn.functional.softmax(t_out / T, dim=1)
            ) * (T * T)
            loss_ce = ce_crit(s_out, y_train)
            loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            s_opt.step()

        student.eval()
        with torch.no_grad():
            s_acc_test = (torch.argmax(student(X_test), 1) == y_test).float().mean().item() * 100
            
        print("[*] 3. Validando fidelidad Bit-a-Bit (PyTorch vs Numpy)...")
        student.eval()
        with torch.no_grad():
            torch_out = student(X_test).cpu().numpy().astype(np.float32)
        sim_out = simulate_cpp_inference(student, X_test, args.k_bits)
        agreement = (np.argmax(torch_out, axis=1) == np.argmax(sim_out, axis=1)).mean()
        
        print(f"    - Precisión QAT : {s_acc_test:.2f}%")
        print(f"    - Fidelidad     : {agreement*100:.2f}%")
        
        if agreement >= 0.98:
            success = True
        else:
            if args.auto_m:
                print(f"[!] Fidelidad baja. Incrementando M a {current_m + 1}...")
                current_m += 1
            else:
                print("[!] ERROR: El simulador no reproduce la red. Usa --auto_m o sube --expansion.")
                break

    if success:
        export_path = os.path.join(args.out_dir, "symbex_weights.h")
        total_bytes, total_params = export_model_to_symbex_h(student, export_path, k_bits=args.k_bits)
        
        fp32_bytes = total_params * 4.0
        symbex_kb = total_bytes / 1024.0
        symbex_mb = symbex_kb / 1024.0
        compression_ratio = ((fp32_bytes - total_bytes) / fp32_bytes) * 100
        
        print("\n==================================================")
        print(" SYMBEX-1 DATA SHEET: COMPRESSION REPORT")
        print("==================================================")
        print(" MODEL METRICS")
        # [MITIGACIÓN]: Ahora el Task Type es honesto y fijo.
        print("   - Task Type        : Feed-Forward (Static)")
        print(f"   - Total Parameters : {total_params:,}")
        print(f"   - FP32 Disk Space  : {fp32_bytes / (1024*1024):.4f} MB")
        print(f"   - SYMBEX Disk Space: {symbex_mb:.4f} MB ({symbex_kb:.2f} KB)")
        print(f"   - Compression Ratio: {compression_ratio:.4f}%")
        print(" PERFORMANCE")
        print(f"   - FP32 Accuracy    : {acc_test:.2f}%")
        print(f"   - SYMBEX Accuracy  : {s_acc_test:.2f}%")
        print(f"   - Bit-level Fidelity: {agreement*100:.2f}%")
        print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SYMBEX-1 Compiler v3.2")
    parser.add_argument("--classes", type=int, default=8, help="Clases a predecir")
    parser.add_argument("--hidden", type=int, default=128, help="Neuronas de capa oculta")
    parser.add_argument("--k_bits", type=int, default=2, help="Resolución de Bit-Slicing")
    parser.add_argument("--expansion", type=int, default=1, help="Expansión M inicial")
    parser.add_argument("--epochs", type=int, default=150, help="Épocas de destilación")
    parser.add_argument("--out_dir", type=str, default="lib_symbex/src", help="Ruta de exportación .h")
    
    # [MITIGACIÓN]: Se elimina la flag --autoregressive para no crear falsas expectativas en el CLI.
    parser.add_argument("--auto_m", action="store_true", help="Ajusta automáticamente M hasta lograr 98%% de fidelidad.")
    
    args = parser.parse_args()
    main(args)
