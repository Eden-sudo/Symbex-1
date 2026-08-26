"""
CHEQUEO DE FIDELIDAD BIT-A-BIT (PyTorch vs Emulación C++)
Identifica exactamente en qué imágenes diverge el hardware del simulador.
"""

import sys
import os

# Forzamos a Python a mirar dentro de la carpeta 'tools'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Importamos las clases desde tu archivo de entrenamiento real
import test_block_gating
from test_block_gating import build_student, BipolarStepSTE, SymbexBlockGatedPoolV2, SymbexPlainPool

# [!] EL HACK PARA PYTORCH: Engañamos a PyTorch para que sepa que las clases 
# que guardó en __main__ ahora están importadas aquí, para evitar el ModuleNotFoundError.
sys.modules['__main__'].SymbexBlockGatedPoolV2 = SymbexBlockGatedPoolV2
sys.modules['__main__'].SymbexPlainPool = SymbexPlainPool
sys.modules['__main__'].BipolarStepSTE = BipolarStepSTE

def emular_cpp_popcount(x_bin, w_bin, in_features):
    """Emula el (2 * popcount(xnor) - 8) a nivel de byte de C++"""
    score = 0
    bytes_len = in_features // 8
    for b in range(bytes_len):
        x_byte = x_bin[b*8 : (b+1)*8]
        w_byte = w_bin[b*8 : (b+1)*8]
        # XNOR a nivel lógico (ambos 1 o ambos 0)
        matches = np.sum(x_byte == w_byte)
        score += (2 * matches - 8)
    return score

def main():
    print("[*] Cargando datos y aislando el dataset de prueba...")
    digits = load_digits()
    X_all = np.where(digits.data > 8, 1, 0) # 0 y 1 lógico
    y_all = digits.target
    _, X_test_np, _, y_test_np = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    # Convertimos los 0 lógicos de X_test_np a -1 para PyTorch
    X_test_torch = torch.where(X_test_tensor > 0, 1.0, -1.0)

    # 1. Cargar el modelo con ruta absoluta
    model_path = os.path.join(os.getcwd(), 'student_05.pt')
    try:
        student = torch.load(model_path, weights_only=False)
        print("[+] Modelo PyTorch cargado correctamente.")
    except Exception as e:
        print(f"[!] Error real al cargar '{model_path}':\n{e}")
        return

    student.eval()
    layer0 = student[0] # BlockGated
    layer2 = student[2] # PlainPool

    mismatches = 0

    print("[*] Iniciando comparación Bit-a-Bit (PyTorch vs Emulación C++)...")
    with torch.no_grad():
        for i in range(len(X_test_np)):
            x_input = X_test_torch[i].unsqueeze(0)
            x_bin_logic = X_test_np[i] # Arreglo de 0s y 1s

            # --- SALIDA PYTORCH ---
            out_torch = student(x_input)
            pred_torch = torch.argmax(out_torch, dim=1).item()

            # --- EMULACIÓN C++ ---
            # 1. Evaluar Gate
            gate_w_logic = (layer0.gate_weight > 0).cpu().numpy().astype(int)
            core_w_logic = (layer0.core_weight > 0).cpu().numpy().astype(int)
            
            gate_scores = []
            for b in range(layer0.num_blocks):
                g_score = emular_cpp_popcount(x_bin_logic, gate_w_logic[b], layer0.in_features)
                gate_scores.append((g_score, b))
            
            # Ordenamiento estable (como el Insertion Sort de C++)
            # Si hay empate, se mantiene el orden original (índices menores ganan primero)
            gate_scores.sort(key=lambda x:(-x[0], x[1]))
            active_blocks = [idx for score, idx in gate_scores[:layer0.k_active]]

            # 2. Evaluar Core (Capa Oculta)
            hidden_out = np.zeros(layer0.out_features, dtype=int)
            for n in range(layer0.out_features):
                block_id = n // layer0.block_size
                if block_id not in active_blocks:
                    continue # Early exit emulado
                
                c_score = emular_cpp_popcount(x_bin_logic, core_w_logic[n], layer0.in_features)
                if c_score > 0:
                    hidden_out[n] = 1

            # 3. Evaluar Salida (Capa Plana)
            out_w_logic = (layer2.weight > 0).cpu().numpy().astype(int)
            final_scores = []
            for n in range(layer2.out_features):
                f_score = emular_cpp_popcount(hidden_out, out_w_logic[n], layer2.in_features)
                final_scores.append(f_score)
            
            pred_cpp = np.argmax(final_scores)

            # --- COMPARACIÓN ---
            if pred_torch != pred_cpp:
                mismatches += 1
                print(f"[!] Mismatch en img {i:03d}: PyTorch={pred_torch} | C++_Emu={pred_cpp}")
                print(f"    Puntajes Gate C++: {gate_scores}")
                print(f"    Bloques Activos C++: {active_blocks}")
    
    print(f"\n[*] Total de Mismatches: {mismatches} / {len(X_test_np)}")
    if mismatches == 0:
        print("[+] La matemática de C++ es 100% fiel al simulador PyTorch.")
    else:
        print("[-] Hay divergencia. El bug está en el Top-K o en la escala de cuantización.")

if __name__ == "__main__":
    main()
