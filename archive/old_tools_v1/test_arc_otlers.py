"""
PRUEBA DE CONCEPTO: Binarized Gated Masking con Data Sheet de Hardware + Exportador a C++
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# =====================================================================
# 1. LAS FUNCIONES DE CUANTIZACIÓN
# =====================================================================
class BipolarSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
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

class BinaryGateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input > 0, torch.ones_like(input), torch.zeros_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0
        return grad_input

# =====================================================================
# 2. LA NUEVA CAPA: SYMBEX GATED POOL
# =====================================================================
class SymbexGatedPool(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.core_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        nn.init.kaiming_uniform_(self.core_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        
        self.output_scale = nn.Parameter(torch.tensor(0.05))
        self.last_mask = None

    def forward(self, x):
        w_core_bin = BipolarSTE.apply(self.core_weight)
        w_gate_bin = BipolarSTE.apply(self.gate_weight)
        
        gate_logits = F.linear(x, w_gate_bin)
        mask = BinaryGateSTE.apply(gate_logits)
        self.last_mask = mask.detach()
        
        core_out = F.linear(x, w_core_bin)
        out_final = core_out * mask
        
        return out_final * self.output_scale

# =====================================================================
# 3. EXPORTADOR A ARDUINO (.h)
# =====================================================================
def export_gated_model_to_h(student, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("#ifndef SYMBEX_WEIGHTS_H\n#define SYMBEX_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n#include <stddef.h>\n")
        f.write("#include \"SymbexNetwork.h\"\n\n")
        f.write("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
        
        # Arrays vacíos (Dummies) para no romper la estructura original de SymbexSubLayer
        f.write("static const uint8_t dummy_outl[1] PROGMEM = {0};\n")
        f.write("static const int8_t dummy_mag[1] PROGMEM = {0};\n\n")
        
        layer_instances = []
        layer_counter = 0
        
        for layer in student:
            if isinstance(layer, SymbexGatedPool):
                out_f, in_f = layer.gate_weight.shape
                bytes_per_neuron = (in_f + 7) // 8
                
                # Binarización física para el microcontrolador (1 o 0 lógico)
                gate_bin = (layer.gate_weight > 0).detach().cpu().numpy()
                core_bin = (layer.core_weight > 0).detach().cpu().numpy()
                
                gate_plane, core_plane = [], []
                
                for n in range(out_f):
                    for b in range(bytes_per_neuron):
                        g_byte, c_byte = 0, 0
                        for bit_idx in range(8):
                            weight_idx = b * 8 + bit_idx
                            if weight_idx < in_f:
                                if gate_bin[n, weight_idx]: g_byte |= (1 << (7 - bit_idx))
                                if core_bin[n, weight_idx]: c_byte |= (1 << (7 - bit_idx))
                        gate_plane.append(g_byte)
                        core_plane.append(c_byte)
                        
                n_gate = f"layer{layer_counter}_gate"
                n_core = f"layer{layer_counter}_core"
                
                f.write(f"static const uint8_t {n_gate}[{len(gate_plane)}] PROGMEM = {{")
                f.write(", ".join(f"0x{v:02X}" for v in gate_plane) + "};\n\n")
                
                f.write(f"static const uint8_t {n_core}[{len(core_plane)}] PROGMEM = {{")
                f.write(", ".join(f"0x{v:02X}" for v in core_plane) + "};\n\n")
                
                # Armado del nodo. Plano 0: Director, Plano 1: Músculo
                f.write(f"static const SymbexSubLayer layer{layer_counter}_subs[1] = {{\n")
                f.write(f"    {{ {{{n_gate}, {n_core}, NULL, NULL}}, dummy_outl, dummy_mag }}\n")
                f.write("};\n\n")
                
                # Constructor: M=1, K_bits=2 (Le indicamos que lea dos planos físicos)
                f.write(f"static SymbexLayer symbex_layer_{layer_counter}({in_f}, {out_f}, 1, 2, layer{layer_counter}_subs);\n\n")
                
                layer_instances.append(f"symbex_layer_{layer_counter}")
                layer_counter += 1
                
        f.write("// --- RED ARMADA AUTOMÁTICAMENTE ---\n")
        f.write("static SymbexNetwork symbex_net;\n")
        f.write("static inline void symbex_init() {\n")
        for inst in layer_instances:
            f.write(f"    symbex_net.add_layer(&{inst});\n")
        f.write("}\n\n#endif // SYMBEX_WEIGHTS_H\n")


# =====================================================================
# 4. CONSTRUCTOR DEL ESTUDIANTE Y RUTINA PRINCIPAL
# =====================================================================
def build_gated_student(teacher):
    layers = []
    for name, module in teacher.named_modules():
        if isinstance(module, nn.Linear):
            layers.append(SymbexGatedPool(module.in_features, module.out_features))
        elif isinstance(module, BipolarStepSTE):
            layers.append(BipolarStepSTE())
    return nn.Sequential(*layers)

def main():
    print("[*] 1. Cargando Datos (Dígitos UCI)...")
    digits = load_digits()
    X_all = np.where(digits.data > 8, 1.0, -1.0)
    y_all = digits.target
    
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    print("[*] 2. Entrenando Profesor FP32...")
    hidden = 128
    teacher = nn.Sequential(
        nn.Linear(64, hidden, bias=False),
        BipolarStepSTE(),
        nn.Linear(hidden, 10, bias=False)
    )
    
    opt_t = torch.optim.Adam(teacher.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    for _ in range(200):
        opt_t.zero_grad()
        crit(teacher(X_train), y_train).backward()
        opt_t.step()

    print("[*] 3. Destilando Estudiante con Gated Masking...")
    student = build_gated_student(teacher)
    opt_s = torch.optim.Adam(student.parameters(), lr=0.002)
    
    T, alpha = 4.0, 0.85
    epochs = 200
    
    for epoch in range(epochs):
        student.train()
        opt_s.zero_grad()
        
        with torch.no_grad(): t_out = teacher(X_train)
        s_out = student(X_train)
        
        loss_kl = nn.KLDivLoss(reduction='batchmean')(
            nn.functional.log_softmax(s_out / T, dim=1),
            nn.functional.softmax(t_out / T, dim=1)
        ) * (T * T)
        loss_ce = crit(s_out, y_train)
        loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
        
        sparsity_loss = 0.0
        for layer in student:
            if isinstance(layer, SymbexGatedPool):
                sparsity_loss += layer.last_mask.float().mean()
        
        loss += 1.5 * sparsity_loss 
        loss.backward()
        opt_s.step()

    print("[*] 4. Exportando Weights a C++ (examples/SymbexGatedBenchmark/symbex_gated_weights.h)...")
    # Apuntamos el exportador directamente a la carpeta de tu nuevo sketch
    export_path = os.path.join("lib_symbex", "examples", "SymbexGatedBenchmark", "symbex_gated_weights.h")
    export_gated_model_to_h(student, export_path)
    print("[+] Archivo de cabecera generado con éxito. Listo para compilar en Arduino.")

if __name__ == "__main__":
    main()
