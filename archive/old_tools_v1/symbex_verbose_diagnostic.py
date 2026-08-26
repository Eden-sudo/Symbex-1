"""
SYMBEX-1 — Diagnóstico verbose sobre el modelo de dígitos (64->128->8).

No modifica symbex_compiler.py: importa sus clases y le agrega instrumentación
para responder dos preguntas concretas en cada paso del auto_m:

  1. ¿Qué está pasando estadísticamente dentro de cada capa al cuantizar?
     (mean, std, W_max, threshold, cuántos outliers, qué magnitud tienen)
  2. ¿Dónde exactamente difieren Torch y el simulador NumPy, caso por caso?
     (no solo el % de acuerdo -- qué muestras fallan y por qué margen)

Uso: python tools/symbex_verbose_diagnostic.py --k_bits 2 --auto_m
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from symbex_compiler import (
    BipolarStepSTE,
    SymbexVotingPool,
    SymbexTopologyEstimator,
    build_student,
    simulate_cpp_inference,
)


# ============================================================
# 1. DIAGNÓSTICO POR CAPA -- qué pasa al cuantizar cada sub-capa
# ============================================================
def print_layer_diagnostics(student, k_bits):
    print("\n" + "=" * 60)
    print(" DIAGNÓSTICO POR CAPA (post-entrenamiento)")
    print("=" * 60)

    levels = (2 ** k_bits) - 1
    layer_idx = 0

    for layer in student:
        if not isinstance(layer, SymbexVotingPool):
            continue

        print(f"\n--- Capa {layer_idx}: {layer.in_features} -> {layer.out_features} "
              f"(M={layer.M}, k_bits={k_bits}) ---")

        for m in range(layer.M):
            W = layer.weight[m].detach()
            mean = layer.running_mean[m].item()
            std = layer.running_std[m].item()
            w_max = layer.running_W_max[m].item()
            threshold = max(2.0 * std, 1e-4)

            outlier_mask = (W - mean).abs() > threshold
            n_outliers = outlier_mask.sum().item()
            n_total = W.numel()
            pct_outliers = 100.0 * n_outliers / n_total

            if n_outliers > 0:
                outlier_vals = W[outlier_mask].abs()
                mag_min, mag_mean, mag_max = (
                    outlier_vals.min().item(),
                    outlier_vals.mean().item(),
                    outlier_vals.max().item(),
                )
            else:
                mag_min = mag_mean = mag_max = 0.0

            # Distribución de niveles cuantizados (para ver si se están usando
            # todos los niveles disponibles o si colapsó a unos pocos)
            W_core = W.clamp(mean - threshold, mean + threshold)
            W_scaled = (W_core / max(w_max, 1e-9)) * (levels / 2.0) + (levels / 2.0)
            W_quant = W_scaled.round().clamp(0, levels).long()
            level_counts = torch.bincount(W_quant.flatten(), minlength=levels + 1)

            print(f"  [m={m}] mean={mean:+.4f}  std={std:.4f}  W_max={w_max:.4f}  "
                  f"threshold={threshold:.4f}")
            print(f"         outliers: {n_outliers}/{n_total} ({pct_outliers:.2f}%)  "
                  f"magnitud[min/mean/max]={mag_min:.3f}/{mag_mean:.3f}/{mag_max:.3f}")
            print(f"         niveles usados (0..{levels}): "
                  f"{level_counts.tolist()}")

            # --- Fidelidad de reconstrucción: ¿cuánta "forma" del peso
            #     original sobrevivió a la cuantización? ---
            W_reconstructed = 2.0 * W_quant.float() - levels
            if outlier_mask.any():
                sign_msb = torch.where(W_quant >= (levels + 1) / 2, 1.0, -1.0)
                outl_vals = W * outlier_mask.float()
                sum_abs = outl_vals.abs().sum(dim=1, keepdim=True)
                count = outlier_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                mag = torch.clamp((sum_abs / count / max(w_max, 1e-9)) * levels, 0, levels * 3.0).round()
                W_reconstructed = torch.where(outlier_mask, W_reconstructed + mag * sign_msb, W_reconstructed)

            cos_sim = torch.nn.functional.cosine_similarity(
                W.flatten(), W_reconstructed.flatten(), dim=0
            ).item()
            rel_frob_err = (torch.norm(W - W_reconstructed * (w_max / levels))
                             / torch.norm(W).clamp(min=1e-9)).item()
            print(f"         fidelidad de reconstrucción: similitud coseno={cos_sim:.4f}  "
                  f"error relativo (Frobenius)={rel_frob_err:.4f}")

        layer_idx += 1


def print_capacity_relation(teacher, student, acc_fp32, acc_symbex, k_bits, total_bytes=None):
    print("\n" + "=" * 60)
    print(" RELACIÓN CON EL MODELO ORIGINAL (no hay un ratio único --")
    print(" esto triangula la relación desde varios ángulos)")
    print("=" * 60)

    total_params = sum(p.numel() for p in teacher.parameters())
    fp32_bits = total_params * 32
    theoretical_ratio = 32.0 / k_bits

    print(f"\nParámetros totales (misma cantidad en ambos modelos): {total_params:,}")
    print(f"Bits nominales por peso -- FP32: 32  |  SYMBEX: {k_bits}  "
          f"(ratio teórico: {theoretical_ratio:.1f}x, SIN contar overhead de outliers)")

    print(f"\nPrecisión FP32   : {acc_fp32:.2f}%")
    print(f"Precisión SYMBEX : {acc_symbex:.2f}%")
    print(f"Precisión retenida: {100.0 * acc_symbex / acc_fp32:.2f}% de la del profesor")

    print("\nLectura honesta: la compresión de bits (arriba) es un hecho de "
          "almacenamiento -- se mide sola. La 'equivalencia de capacidad' "
          "(cuántas neuronas binarizadas hacen falta para igualar UNA neurona "
          "FP32) NO tiene una fórmula cerrada: depende de cuánta redundancia "
          "tenía esa capa específica. La similitud coseno de reconstrucción "
          "por capa (arriba) es el proxy más directo que tenemos hoy -- una "
          "capa con coseno alto perdió poca 'forma', una con coseno bajo "
          "perdió mucha, y ahí es donde más rendiría una expansión de ancho.")


# ============================================================
# 2. COMPARACIÓN CASO POR CASO -- Torch vs NumPy, no solo el %
# ============================================================
def print_case_by_case(student, X_test, y_test, k_bits, top_n_mismatches=10):
    print("\n" + "=" * 60)
    print(" COMPARACIÓN CASO POR CASO (Torch vs Simulador NumPy)")
    print("=" * 60)

    student.eval()
    with torch.no_grad():
        torch_out = student(X_test).cpu().numpy().astype(np.float32)
    sim_out = simulate_cpp_inference(student, X_test, k_bits)

    torch_pred = np.argmax(torch_out, axis=1)
    sim_pred = np.argmax(sim_out, axis=1)
    y_true = y_test.numpy()

    def top2_margin(logits_row):
        sorted_vals = np.sort(logits_row)[::-1]
        return sorted_vals[0] - sorted_vals[1]

    torch_margins = np.array([top2_margin(row) for row in torch_out])
    sim_margins = np.array([top2_margin(row) for row in sim_out])

    mismatches = np.where(torch_pred != sim_pred)[0]
    agreement = 1.0 - len(mismatches) / len(y_true)

    print(f"\nMuestras totales: {len(y_true)}")
    print(f"Fidelidad bit-a-bit (Torch==NumPy): {agreement*100:.2f}%")
    print(f"Margen top-2 promedio -- Torch: {torch_margins.mean():.3f} | "
          f"NumPy: {sim_margins.mean():.3f}")

    if len(mismatches) == 0:
        print("\n[+] Sin discrepancias entre Torch y el simulador.")
        return

    print(f"\n{len(mismatches)} discrepancias encontradas. "
          f"Mostrando hasta {top_n_mismatches}, ordenadas por margen más chico "
          f"(los empates más ajustados, causa más probable del desacuerdo):")

    # Ordenar por el margen más chico entre Torch y sim (el caso "más al borde")
    order = mismatches[np.argsort(np.minimum(torch_margins[mismatches],
                                              sim_margins[mismatches]))]

    print(f"\n{'idx':>5} {'real':>5} {'torch':>6} {'sim':>4} "
          f"{'marg_torch':>11} {'marg_sim':>9}")
    for i in order[:top_n_mismatches]:
        print(f"{i:5d} {y_true[i]:5d} {torch_pred[i]:6d} {sim_pred[i]:4d} "
              f"{torch_margins[i]:11.3f} {sim_margins[i]:9.3f}")


# ============================================================
# 3. RUTINA PRINCIPAL
# ============================================================
def main(args):
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

    print("[*] Entrenando Profesor (FP32)...")
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

    current_m = 1 if args.auto_m else args.expansion
    success = False

    while current_m <= 8 and not success:
        print(f"\n[*] Destilando Estudiante (K={args.k_bits}, M={current_m})...")
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

        # --- Instrumentación 1: qué pasa dentro de cada capa ---
        print_layer_diagnostics(student, args.k_bits)

        # --- Instrumentación 2: dónde difieren Torch y NumPy, caso por caso ---
        print_case_by_case(student, X_test, y_test, args.k_bits)

        student.eval()
        with torch.no_grad():
            torch_out = student(X_test).cpu().numpy().astype(np.float32)
            s_acc_test = (torch.argmax(student(X_test), 1) == y_test).float().mean().item() * 100
        sim_out = simulate_cpp_inference(student, X_test, args.k_bits)
        agreement = (np.argmax(torch_out, axis=1) == np.argmax(sim_out, axis=1)).mean()

        # --- Instrumentación 3: relación de capacidad con el modelo original ---
        with torch.no_grad():
            acc_fp32 = (torch.argmax(teacher(X_test), 1) == y_test).float().mean().item() * 100
        print_capacity_relation(teacher, student, acc_fp32, s_acc_test, args.k_bits)

        if agreement >= 0.98:
            success = True
        elif args.auto_m:
            print(f"\n[!] Fidelidad baja. Incrementando M a {current_m + 1}...")
            current_m += 1
        else:
            print("\n[!] Fidelidad insuficiente. Usa --auto_m o sube --expansion.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SYMBEX-1 Diagnóstico Verbose")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--k_bits", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--auto_m", action="store_true")
    args = parser.parse_args()
    main(args)
