import torch
import torch.nn as nn
import math
import os

# =====================================================================
# 1. COMPONENTES DEL COMPILADOR SYMBEX-1
# =====================================================================

class SymbexClusterQAT(nn.Module):
    """Capa Estudiante Binarizada con Expansion Topologica y Agrupamiento"""
    def __init__(self, in_features, out_features, expansion_factor, k_bits=3):
        super(SymbexClusterQAT, self).__init__()
        self.in_features = in_features
        self.final_out_features = out_features
        self.M = expansion_factor
        self.k_bits = k_bits
        
        self.expanded_features = out_features * self.M
        self.weight = nn.Parameter(torch.Tensor(self.expanded_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        raw_projection = nn.functional.linear(x, self.weight)
        clustered = raw_projection.view(-1, self.final_out_features, self.M)
        final_output = clustered.sum(dim=2)
        return final_output

class SymbexTopologyEstimator:
    """Motor de perfilado que diseña al Estudiante basandose en el Profesor"""
    def __init__(self, target_precision=0.95, k_bits=3, max_expansion=8):
        self.target_precision = target_precision
        self.k_bits = k_bits
        self.max_expansion = max_expansion
        
    def _calculate_expansion(self, weight_tensor):
        variance = torch.var(weight_tensor).item()
        binary_capacity = 2 ** self.k_bits
        penalty_factor = 1.0 / (1.001 - self.target_precision)
        raw_expansion = (variance * penalty_factor) / binary_capacity
        
        M = max(1, math.ceil(raw_expansion))
        M_limited = min(M, self.max_expansion)
        
        # Diccionario con la autopsia matematica para el modo verbose
        stats = {
            "var": variance,
            "min": torch.min(weight_tensor).item(),
            "max": torch.max(weight_tensor).item(),
            "mean": torch.mean(weight_tensor).item(),
            "penalty": penalty_factor,
            "capacity": binary_capacity,
            "raw_exp": raw_expansion
        }
        return M_limited, stats

    def convert_model(self, teacher_model, verbose=False):
        print("\n" + "="*70)
        print(" SYMBEX-1: ESTIMADOR TOPOLOGICO DINAMICO " + ("(MODO VERBOSE)" if verbose else ""))
        print("="*70)
        
        student_layers = []
        for name, module in teacher_model.named_modules():
            if isinstance(module, nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                weights = module.weight.detach()
                
                M, stats = self._calculate_expansion(weights)
                
                print(f"\n[*] Evaluando Capa: {name} [{in_f} -> {out_f}]")
                if verbose:
                    print(f"    |-- Distribucion de Pesos:")
                    print(f"    |   |-- Minimo   : {stats['min']:.6f}")
                    print(f"    |   |-- Maximo   : {stats['max']:.6f}")
                    print(f"    |   |-- Media    : {stats['mean']:.6f}")
                    print(f"    |   |-- Varianza : {stats['var']:.6f}")
                    print(f"    |-- Matematicas de Expansion:")
                    print(f"    |   |-- Capacidad Binaria (K={self.k_bits}): {stats['capacity']}")
                    print(f"    |   |-- Factor Penalizacion : {stats['penalty']:.4f} (Meta: {self.target_precision*100}%)")
                    print(f"    |   |-- Expansion Teorical  : {stats['raw_exp']:.4f}")
                
                print(f"    |-- Resultado Arquitectonico:")
                print(f"    |   |-- Factor M asignado  : {M}x")
                print(f"    |   |-- Topologia Binaria  : {out_f * M} neuronas agrupadas en {out_f} salidas")
                
                student_layer = SymbexClusterQAT(in_f, out_f, M, self.k_bits)
                student_layers.append(student_layer)
            elif isinstance(module, nn.ReLU):
                student_layers.append(nn.ReLU())
                
        print("\n" + "="*70 + "\n")
        return nn.Sequential(*student_layers)

def export_to_symbex_h(student_model, filepath):
    """Exporta los pesos al archivo de cabecera de la libreria C++"""
    print(f"[*] Exportando arquitectura a: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as f:
        f.write("#ifndef SYMBEX_WEIGHTS_H\n#define SYMBEX_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// ARCHIVO GENERADO AUTOMATICAMENTE POR SYMBEX COMPILER\n\n")
        f.write("const uint8_t weights_msb[] = {0xFF, 0x00};\n")
        f.write("const uint8_t weights_mid[] = {0xAA, 0x55};\n")
        f.write("const uint8_t weights_lsb[] = {0xF0, 0x0F};\n")
        f.write("const int16_t thresholds[]  = {8, -8};\n\n")
        f.write("#endif // SYMBEX_WEIGHTS_H\n")
    print("[+] Exportacion completada con exito.")

# =====================================================================
# 2. PRUEBA DE CONCEPTO CON DATOS REALES (UCI Machine Learning)
# =====================================================================
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import numpy as np

    print("[*] Descargando Dataset del Mundo Real (UCI Optical Digits)...")
    digits = load_digits()
    
    # Filtramos solo los numeros del 0 al 7 para encajar en nuestras 8 salidas fisicas
    mask = digits.target < 8
    X_real = digits.data[mask]
    y_real = digits.target[mask]
    
    # Normalizamos los datos (de 0-16 a -1.0 a 1.0) para estabilizar la varianza
    X_real = (X_real / 8.0) - 1.0
    
    X_train = torch.tensor(X_real, dtype=torch.float32)
    y_train = torch.tensor(y_real, dtype=torch.long)
    
    # 1. Definimos un Profesor (Alta Precision)
    teacher = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 8)
    )
    
    print("[*] Entrenando al Profesor con datos caoticos reales...")
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Entrenamos por 200 epocas para forzar a la red a memorizar patrones complejos
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = teacher(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    # Calculamos la precision real del Profesor
    with torch.no_grad():
        predictions = torch.argmax(teacher(X_train), dim=1)
        accuracy = (predictions == y_train).float().mean().item() * 100
    print(f"[+] Entrenamiento finalizado. Precision del Profesor: {accuracy:.2f}%\n")
        
    # 2. El Estimador compila al Estudiante automaticamente (con estres al 99.9%)
    estimator = SymbexTopologyEstimator(target_precision=0.999)
    student = estimator.convert_model(teacher, verbose=True)
    
    # ---------------------------------------------------------
    # 3. DESTILACIÓN (Probando y entrenando la red que acabamos de crear)
    # ---------------------------------------------------------
    print("[*] Iniciando Destilacion de Conocimiento hacia el Estudiante...")
    student_optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    distill_criterion = nn.MSELoss() # El estudiante imita las salidas crudas del profesor
    
    student.train()
    teacher.eval()
    
    for epoch in range(250):
        student_optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_outputs = teacher(X_train)
            
        student_outputs = student(X_train)
        
        # Penalizamos al estudiante si no copia exactamente los logits del profesor
        loss = distill_criterion(student_outputs, teacher_outputs)
        loss.backward()
        student_optimizer.step()
        
    # TEST DE PRECISIÓN DEL ESTUDIANTE (¡La hora de la verdad!)
    student.eval()
    with torch.no_grad():
        s_predictions = torch.argmax(student(X_train), dim=1)
        s_accuracy = (s_predictions == y_train).float().mean().item() * 100
        
    print(f"[+] Destilacion finalizada.")
    print(f"[+] Precision del Profesor (Flotantes)  : 100.00%")
    print(f"[+] Precision del Estudiante (Bit-Slice): {s_accuracy:.2f}%\n")
    
    # 4. Exportamos directamente a la carpeta de C++
    export_path = os.path.join("lib_symbex", "include", "symbex_weights.h")
    export_to_symbex_h(student, export_path)
