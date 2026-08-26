"""
SYMBEX-1 — Diagnóstico verbose sobre MNIST (784 -> hidden -> 10), MLP puro.

Mismo patrón que symbex_verbose_diagnostic.py, pero con un profesor entrenado
sobre MNIST completo (más difícil que dígitos UCI) para forzar degradación
real y poder ver dónde aprieta la capacidad, capa por capa.

IMPORTANTE: el teacher debe ser 100% nn.Linear (sin convoluciones), porque
build_student() en symbex_compiler.py solo convierte capas Linear -- cualquier
capa conv se saltearía silenciosamente y rompería la arquitectura del
estudiante.

Uso: python tools/symbex_mnist_diagnostic.py --hidden 32 --k_bits 1 --auto_m

Primera corrida: descarga MNIST vía sklearn (~50MB, se cachea localmente,
solo tarda la primera vez).
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from symbex_compiler import (
    BipolarStepSTE,
    SymbexVotingPool,
    SymbexTopologyEstimator,
    build_student,
    simulate_cpp_inference,
)
from symbex_verbose_diagnostic import (
    print_layer_diagnostics,
    print_case_by_case,
    print_capacity_relation,
)


def load_mnist():
    print("[*] Descargando/leyendo MNIST (cache local si ya existe)...")
    t0 = time.time()
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    print(f"    Listo en {time.time()-t0:.1f}s -- {mnist.data.shape[0]} muestras, "
          f"{mnist.data.shape[1]} features")

    X_all = np.where(mnist.data > 127, 1.0, -1.0).astype(np.float32)  # binarizado, mismo criterio que digits
    y_all = mnist.target.astype(np.int64)
    return X_all, y_all


def main(args):
    X_all, y_all = load_mnist()

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    print(f"[*] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("[*] Entrenando Profesor (FP32, MLP Expandido)...")
    IN_FEATURES = X_all.shape[1]  # 784
    CLASSES = 10
    EXPANDED_DIM = 128  # El amortiguador de entropía
    
    teacher = nn.Sequential(
        # Capa 0: Extracción (Ancla)
        nn.Linear(IN_FEATURES, args.hidden, bias=False),
        BipolarStepSTE(),
        # Capa 1: Expansión (El nuevo amortiguador)
        nn.Linear(args.hidden, EXPANDED_DIM, bias=False),
        BipolarStepSTE(),
        # Capa 2: Proyección final (El nuevo cuello de botella)
        nn.Linear(EXPANDED_DIM, CLASSES, bias=False),
    )
    opt = torch.optim.Adam(teacher.parameters(), lr=0.005)
    ce_crit = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(args.teacher_epochs):
        opt.zero_grad()
        loss = ce_crit(teacher(X_train), y_train)
        loss.backward()
        opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Teacher epoch {epoch+1}/{args.teacher_epochs} | loss={loss.item():.4f}")
    print(f"    Entrenamiento del profesor: {time.time()-t0:.1f}s")

    teacher.eval()
    with torch.no_grad():
        acc_fp32 = (torch.argmax(teacher(X_test), 1) == y_test).float().mean().item() * 100
    print(f"[+] Precisión FP32 (Datos invisibles): {acc_fp32:.2f}%")

    current_m = 1 if args.auto_m else args.expansion
    success = False

    while current_m <= 8 and not success:
        print(f"\n[*] Destilando Estudiante (K={args.k_bits}, M={current_m}, hidden={args.hidden})...")
        estimator = SymbexTopologyEstimator(k_bits=args.k_bits, max_expansion=current_m)
        student = build_student(teacher, estimator, verbose=False)

        s_opt = torch.optim.Adam(student.parameters(), lr=0.001)
        T, alpha = 4.0, 0.85

        t0 = time.time()
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
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{args.epochs} | loss={loss.item():.4f}")
        print(f"    Destilación: {time.time()-t0:.1f}s")

        print_layer_diagnostics(student, args.k_bits)
        print_case_by_case(student, X_test, y_test, args.k_bits)

        student.eval()
        with torch.no_grad():
            torch_out = student(X_test).cpu().numpy().astype(np.float32)
            s_acc_test = (torch.argmax(student(X_test), 1) == y_test).float().mean().item() * 100
        sim_out = simulate_cpp_inference(student, X_test, args.k_bits)
        agreement = (np.argmax(torch_out, axis=1) == np.argmax(sim_out, axis=1)).mean()

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
    parser = argparse.ArgumentParser(description="SYMBEX-1 Diagnóstico Verbose sobre MNIST")
    parser.add_argument("--hidden", type=int, default=32, help="Neuronas de capa oculta")
    parser.add_argument("--k_bits", type=int, default=1)
    parser.add_argument("--expansion", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=800, help="Épocas de destilación del estudiante")
    parser.add_argument("--teacher_epochs", type=int, default=400, help="Épocas de entrenamiento del profesor")
    parser.add_argument("--auto_m", action="store_true")
    args = parser.parse_args()
    main(args)
