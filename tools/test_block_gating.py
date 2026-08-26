"""
SYMBEX-1 V2 — Block-Gated 1-Bit Engine
Incluye: Colchón de Capacidad (Ensanchamiento) y Relación de Convergencia (Épocas 1:2)
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# CONFIGURACIÓN MAESTRA (LA CIENCIA DE LA BITÁCORA)
# ---------------------------------------------------------
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 512   # El "Colchón": 4x neuronas 1-bit para asimilar FP32
BLOCK_SIZE = 32        # Tamaño de clúster de activación

TEACHER_EPOCHS = 150
STUDENT_EPOCHS = 300   # Regla empírica 1:2 para asimilar cuantización

# Fijar semillas para reproducibilidad absoluta
torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# 1. ACTIVACIÓN 1-BIT PURO (BIPOLAR STE)
# =========================================================
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

# =========================================================
# 2. CAPA V2: BLOCK-GATED (TOPOLOGÍA DINÁMICA)
# =========================================================
class SymbexBlockGatedPoolV2(nn.Module):
    def __init__(self, in_features, out_features, block_size=32, active_ratio=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = max(1, out_features // block_size)
        self.k_active = max(1, int(self.num_blocks * active_ratio))
        self.active_ratio = active_ratio

        # El Director (Gate)
        self.gate_weight = nn.Parameter(torch.empty(self.num_blocks, in_features))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        # El Músculo (Core)
        self.core_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.core_weight, a=math.sqrt(5))
        
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        w_gate_bin = BipolarSTE.apply(self.gate_weight)
        w_core_bin = BipolarSTE.apply(self.core_weight)

        # 1. Puntuar Bloques (Director)
        gate_scores = nn.functional.linear(x, w_gate_bin)
        
        # [TIE-BREAKER DETERMINISTA UNIVERSAL]
        # Favorece siempre al índice menor en caso de empate exacto, 
        # idéntico en train y eval.
        tie_break = -torch.arange(self.num_blocks, dtype=gate_scores.dtype, device=x.device) * 1e-4
        gate_scores = gate_scores + tie_break

        # 2. Ordenamiento Top-K
        _, topk_indices = torch.topk(gate_scores, self.k_active, dim=-1)

        # 3. Máscara y Expansión
        mask = torch.zeros_like(gate_scores)
        mask.scatter_(1, topk_indices, 1.0)
        mask_expanded = mask.repeat_interleave(self.block_size, dim=1)

        # 4. Inferencia Real (Músculo) y Early Exit simulado
        core_scores = nn.functional.linear(x, w_core_bin)
        out = core_scores * mask_expanded
        
        return out * self.scale

# =========================================================
# 3. EXPORTADOR A C++ (.h)
# =========================================================
def export_v2_model_to_h(student, filepath, block_size, k_active):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    layer0 = student[0]
    w_gate = torch.where(layer0.gate_weight > 0, 1, 0).byte().cpu().numpy()
    w_core = torch.where(layer0.core_weight > 0, 1, 0).byte().cpu().numpy()
    
    in_f = layer0.in_features
    out_f = layer0.out_features
    num_blocks = layer0.num_blocks
    
    def pack_bits(weight_matrix):
        rows, cols = weight_matrix.shape
        packed = []
        for r in range(rows):
            row_bytes = []
            for b_idx in range(0, cols, 8):
                byte_val = 0
                for bit in range(8):
                    if b_idx + bit < cols:
                        byte_val |= (weight_matrix[r, b_idx + bit] << (7 - bit))
                row_bytes.append(byte_val)
            packed.append(row_bytes)
        return np.array(packed, dtype=np.uint8)

    gate_packed = pack_bits(w_gate)
    core_packed = pack_bits(w_core)
    
    # Capa final de salida (dense 1-bit)
    layer1 = student[2]
    w_out_core = torch.where(layer1.core_weight > 0, 1, 0).byte().cpu().numpy()
    out_packed = pack_bits(w_out_core)
    final_classes = layer1.out_features
    
    with open(filepath, 'w') as f:
        f.write("// ==================================================\n")
        f.write("// SYMBEX-1 V2 (BLOCK-GATED 1-BIT) PESOS EXPORTADOS\n")
        f.write("// ==================================================\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
        
        f.write(f"#define IN_FEATURES_BITS {in_f}\n")
        f.write(f"#define OUT_FEATURES_BITS {out_f}\n")
        f.write(f"#define BLOCK_SIZE_BITS {block_size}\n")
        f.write(f"#define GATE_NUM_BLOCKS {num_blocks}\n")
        f.write(f"#define GATE_K_ACTIVE {k_active}\n")
        f.write(f"#define FINAL_CLASSES {final_classes}\n\n")
        
        f.write(f"const uint8_t gate_weights_bin[{num_blocks}][{(in_f+7)//8}] PROGMEM = {{\n")
        for r in gate_packed:
            f.write("  {" + ", ".join(f"0x{v:02X}" for v in r) + "},\n")
        f.write("};\n\n")
        
        f.write(f"const uint8_t core_weights_bin[{out_f}][{(in_f+7)//8}] PROGMEM = {{\n")
        for r in core_packed:
            f.write("  {" + ", ".join(f"0x{v:02X}" for v in r) + "},\n")
        f.write("};\n\n")

        f.write(f"const uint8_t out_weights_bin[{final_classes}][{(out_f+7)//8}] PROGMEM = {{\n")
        for r in out_packed:
            f.write("  {" + ", ".join(f"0x{v:02X}" for v in r) + "},\n")
        f.write("};\n\n")

# =========================================================
# 4. FLUJO PRINCIPAL
# =========================================================
def main():
    print("[*] 1. Preparando entorno y cargando Digits Dataset...")
    digits = load_digits()
    X_all = np.where(digits.data > 8, 1.0, -1.0).astype(np.float32)
    y_all = digits.target.astype(np.int64)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np)
    X_test = torch.tensor(X_test_np)
    y_test = torch.tensor(y_test_np)

    print(f"[*] Entrenando Profesor FP32 (Ocultas: {TEACHER_HIDDEN} | Épocas: {TEACHER_EPOCHS})...")
    teacher = nn.Sequential(
        nn.Linear(64, TEACHER_HIDDEN, bias=False),
        BipolarStepSTE(),
        nn.Linear(TEACHER_HIDDEN, 10, bias=False)
    )
    
    opt_t = torch.optim.Adam(teacher.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    
    for _ in range(TEACHER_EPOCHS):
        opt_t.zero_grad()
        loss = crit(teacher(X_train), y_train)
        loss.backward()
        opt_t.step()

    teacher.eval()
    with torch.no_grad():
        acc_fp32 = (torch.argmax(teacher(X_test), 1) == y_test).float().mean().item() * 100
    print(f"[+] Precisión FP32 base: {acc_fp32:.2f}%\n")

    print(f"[*] Destilando Estudiantes 1-Bit (Colchón Ocultas: {STUDENT_HIDDEN} | Épocas: {STUDENT_EPOCHS})...")
    print(f"{'Activo':<8} | {'Precisión':<10} | {'Sparsity Real':<14} | {'Ahorro MACs Neto':<16}")
    print("-" * 65)

    active_ratios = [1.0, 0.75, 0.5, 0.25, 0.1]
    best_student_to_export = None
    best_k_active = 0

    T, alpha = 4.0, 0.85

    for ratio in active_ratios:
        # Construimos el estudiante con el "Colchón de Aprendizaje" ensanchado
        student = nn.Sequential(
            SymbexBlockGatedPoolV2(64, STUDENT_HIDDEN, block_size=BLOCK_SIZE, active_ratio=ratio),
            BipolarStepSTE(),
            SymbexBlockGatedPoolV2(STUDENT_HIDDEN, 10, block_size=10, active_ratio=1.0)
        )
        
        s_opt = torch.optim.Adam(student.parameters(), lr=0.001)

        # Regla empírica: El doble de tiempo para asimilar la cuantización
        for epoch in range(STUDENT_EPOCHS):
            student.train()
            s_opt.zero_grad()
            with torch.no_grad():
                t_out = teacher(X_train)
            s_out = student(X_train)
            
            loss_kl = nn.KLDivLoss(reduction='batchmean')(
                nn.functional.log_softmax(s_out / T, dim=1),
                nn.functional.softmax(t_out / T, dim=1)
            ) * (T * T)
            loss_ce = crit(s_out, y_train)
            loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
            
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            s_opt.step()

        student.eval()
        with torch.no_grad():
            s_acc = (torch.argmax(student(X_test), 1) == y_test).float().mean().item() * 100
            
        sparsity = (1.0 - ratio) * 100
        # Cálculo del overhead del gate: (num_blocks * in_features) vs (out_features * in_features)
        overhead = (student[0].num_blocks * 64) / (STUDENT_HIDDEN * 64) * 100
        net_savings = sparsity - overhead
        
        print(f"{ratio:<8} | {s_acc:>8.2f}% | {sparsity:>12.2f}% | {net_savings:>+14.2f}%")

        if ratio == 1.0:
            best_student_to_export = student
            best_k_active = student[0].k_active

    if best_student_to_export is not None:
        print("\n[*] Exportando Modelo V2 (Ratio 1.0 - Colchón Máximo) a C++...")
        # Exporta directamente a la ruta donde compilarás
        export_path = "lib_symbex/examples/SymbexGatedBenchmarkESP32/symbex_gated_weights.h"
        export_v2_model_to_h(best_student_to_export, export_path, BLOCK_SIZE, best_k_active)
        print(f"[+] Archivo generado en: {export_path}")

if __name__ == "__main__":
    main()
