"""
SYMBEX-1 Universal Compiler (V1 & V2).
CLI entry point for training, validating, and exporting Edge-AI models.
"""
import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Internal imports from the compiler core
from compiler_core.models.core import BipolarStepSTE
from compiler_core.models.bitslice import SymbexVotingPool
from compiler_core.models.gated import SymbexBlockGatedPool
from compiler_core import trainer, validator, exporter

def set_seeds(seed=42):
    """Locks all random engines to guarantee reproducible firmware output."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_student_v1(teacher, expansion, k_bits):
    """Constructs a V1 (Bitslice) architecture derived from the Teacher."""
    layers = []
    for module in teacher.modules():
        if isinstance(module, nn.Linear):
            student_layer = SymbexVotingPool(
                module.in_features, module.out_features, 
                expansion_factor=expansion, k_bits=k_bits
            )
            # Warm start: copy teacher weights with noise
            with torch.no_grad():
                for m in range(student_layer.M):
                    noise = torch.randn_like(module.weight) * 0.01
                    student_layer.weight[m].copy_(module.weight + noise)
            layers.append(student_layer)
        elif isinstance(module, BipolarStepSTE):
            layers.append(BipolarStepSTE())
    return nn.Sequential(*layers)

def build_student_v2(in_features, hidden_features, classes, block_size):
    """Constructs a V2 (Block-Gated) architecture with dynamic early exit."""
    return nn.Sequential(
        SymbexBlockGatedPool(in_features, hidden_features, block_size=block_size, active_ratio=1.0),
        BipolarStepSTE(),
        SymbexBlockGatedPool(hidden_features, classes, block_size=10, active_ratio=1.0)
    )

def main():
    parser = argparse.ArgumentParser(description="SYMBEX-1 Compiler (V1 & V2)")
    parser.add_argument("--mode", choices=["bitslice", "gated"], required=True, help="Compilation target architecture")
    parser.add_argument("--classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--t_epochs", type=int, default=150, help="Teacher training epochs")
    parser.add_argument("--s_epochs", type=int, default=300, help="Student distillation epochs")
    parser.add_argument("--out_dir", type=str, default="lib_symbex/examples/", help="Base output directory")
    
    # V1 Specific
    parser.add_argument("--k_bits", type=int, default=1, help="[V1] Resolution bits")
    parser.add_argument("--expansion", type=int, default=1, help="[V1] Voting Pool Expansion (M)")
    
    # V2 Specific
    parser.add_argument("--student_hidden", type=int, default=512, help="[V2] Student Hidden Buffer Size")
    parser.add_argument("--block_size", type=int, default=32, help="[V2] Gate clustering size")
    
    args = parser.parse_args()
    set_seeds()

    # 1. Dataset Preparation
    print("[*] 1. Loading Digits Dataset...")
    digits = load_digits()
    mask = digits.target < args.classes
    X_all = np.where(digits.data[mask] > 8, 1.0, -1.0).astype(np.float32)
    y_all = digits.target[mask].astype(np.int64)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np)
    X_test = torch.tensor(X_test_np)
    y_test = torch.tensor(y_test_np)
    in_features = X_all.shape[1]

    # 2. Teacher Training
    print(f"\n[*] 2. Training FP32 Master Teacher ({args.t_epochs} epochs)...")
    teacher = nn.Sequential(
        nn.Linear(in_features, 128, bias=False),
        BipolarStepSTE(),
        nn.Linear(128, args.classes, bias=False)
    )
    teacher = trainer.train_teacher(teacher, X_train, y_train, epochs=args.t_epochs)
    acc_fp32 = trainer.evaluate_accuracy(teacher, X_test, y_test)
    print(f"[+] FP32 Baseline Accuracy: {acc_fp32:.2f}%")

    # 3. Student Topology Generation & Distillation
    print(f"\n[*] 3. Distilling 1-Bit Student [{args.mode.upper()}] ({args.s_epochs} epochs)...")
    
    if args.mode == "bitslice":
        student = build_student_v1(teacher, args.expansion, args.k_bits)
    else:
        student = build_student_v2(in_features, args.student_hidden, args.classes, args.block_size)

    # Physical limits check before heavy training
    validator.validate_hardware_limits(student, args.mode)

    student = trainer.distill_student(student, teacher, X_train, y_train, epochs=args.s_epochs)
    acc_symbex = trainer.evaluate_accuracy(student, X_test, y_test)
    print(f"[+] Quantized Accuracy: {acc_symbex:.2f}%")

    # 4. Strict Hardware Fidelity Check
    print("\n[*] 4. Simulating C++ Hardware Physics in Numpy...")
    fidelity = validator.verify_fidelity(student, X_test, X_test_np, args.mode, k_bits=args.k_bits)
    print(f"    - Hardware Math Fidelity: {fidelity:.2f}%")
    
    if fidelity < 98.0:
        print("[!] FATAL: Simulation mismatch. Aborting export.")
        return

    # 5. Export to Firmware
    print("\n[*] 5. Generating Firmware C++ Headers...")
    if args.mode == "bitslice":
        out_path = os.path.join(args.out_dir, "SymbexBitsliceBenchmark/symbex_weights.h")
        bytes_used, params = exporter.export_model_v1_to_h(student, out_path, k_bits=args.k_bits)
    else:
        out_path = os.path.join(args.out_dir, "SymbexGatedBenchmark/symbex_gated_weights.h")
        # Ensure k_active is extracted dynamically from the BlockGated layer
        k_active = student[0].k_active
        bytes_used, params = exporter.export_model_v2_to_h(student, out_path, args.block_size, k_active)

    print(f"[+] Success. Model firmly exported to: {out_path}")
    print(f"    - Parameters: {params} | Flash Cost: {bytes_used/1024:.2f} KB")

if __name__ == "__main__":
    main()
