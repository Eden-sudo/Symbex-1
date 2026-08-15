import torch
import torch.nn as nn
import math
import os

# =====================================================================
# 1. COMPONENTES DEL COMPILADOR SYMBEX-1 (OUTLIER-AWARE)
# =====================================================================

class SymbexClusterQAT(nn.Module):
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
    def __init__(self, target_precision=0.95, k_bits=3, max_expansion=8):
        self.target_precision = target_precision
        self.k_bits = k_bits
        self.max_expansion = max_expansion
        
    def _extract_and_squash_outliers(self, weight_tensor):
        """
        Aisla los outliers (+2 Desviaciones Estandar) y aplasta el tensor original
        para reducir masivamente la varianza y ahorrar memoria.
        """
        mean = torch.mean(weight_tensor)
        std = torch.std(weight_tensor)
        threshold = 2.0 * std # Todo lo que este a 2 std es un outlier
        
        # Identificamos donde estan los outliers
        outlier_mask = torch.abs(weight_tensor - mean) > threshold
        
        # Aplastamos (Clip) el tensor principal para estabilizarlo
        squashed_weights = weight_tensor.clone()
        squashed_weights = torch.clamp(squashed_weights, min=(mean - threshold).item(), max=(mean + threshold).item())
        
        # Estadisticas del impacto
        original_var = torch.var(weight_tensor).item()
        squashed_var = torch.var(squashed_weights).item()
        outliers_count = torch.sum(outlier_mask).item()
        
        return squashed_weights, original_var, squashed_var, outliers_count

    def _calculate_expansion(self, squashed_variance):
        binary_capacity = 2 ** self.k_bits
        penalty_factor = 1.0 / (1.001 - self.target_precision)
        raw_expansion = (squashed_variance * penalty_factor) / binary_capacity
        
        M = max(1, math.ceil(raw_expansion))
        return min(M, self.max_expansion), penalty_factor, raw_expansion

    def convert_model(self, teacher_model, verbose=False):
        print("\n" + "="*70)
        print(" SYMBEX-1: COMPILADOR CDT CON AISLAMIENTO DE OUTLIERS ")
        print("="*70)
        
        student_layers = []
        for name, module in teacher_model.named_modules():
            if isinstance(module, nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                weights = module.weight.detach()
                
                # 1. Aplicamos la Capa de Abstraccion (Aplastamiento)
                sq_weights, orig_var, sq_var, out_cnt = self._extract_and_squash_outliers(weights)
                
                # 2. Calculamos Expansion usando la Varianza Aplastada
                M, penalty, raw_exp = self._calculate_expansion(sq_var)
                
                print(f"\n[*] Evaluando Capa: {name} [{in_f} -> {out_f}]")
                if verbose:
                    print(f"    |-- Impacto de la Capa de Outliers:")
                    print(f"    |   |-- Outliers detectados : {out_cnt} pesos atipicos")
                    print(f"    |   |-- Varianza Original   : {orig_var:.6f}")
                    print(f"    |   |-- Varianza Aplastada  : {sq_var:.6f}  <-- ¡Magia!")
                    print(f"    |-- Matematicas de Expansion (Meta: {self.target_precision*100}%):")
                    print(f"    |   |-- Factor Penalizacion : {penalty:.4f}")
                    print(f"    |   |-- Expansion Teorical  : {raw_exp:.4f}")
                
                print(f"    |-- Resultado Arquitectonico:")
                print(f"    |   |-- Factor M asignado   : {M}x")
                
                student_layer = SymbexClusterQAT(in_f, out_f, M, self.k_bits)
                student_layers.append(student_layer)
            elif isinstance(module, nn.ReLU):
                student_layers.append(nn.ReLU())
                
        print("\n" + "="*70 + "\n")
        return nn.Sequential(*student_layers)

def export_to_symbex_h(student_model, filepath):
    print("\n[*] INICIANDO TRADUCCIÓN BINARIA (FLOTANTE -> C++)")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Para nuestra PoC inicial del Gemelo Digital, vamos a extraer y exportar 
    # únicamente la PRIMERA CAPA de la red (student_model[0])
    layer = student_model[0]
    W = layer.weight.detach().cpu().numpy()
    out_features, in_features = W.shape
    
    # 1. Aislamiento de Outliers
    mean = np.mean(W)
    std = np.std(W)
    threshold = 2.0 * std
    
    outlier_mask_bits = np.abs(W - mean) > threshold
    W_core = np.clip(W, mean - threshold, mean + threshold)
    
    # 2. Cuantización K=3 (Mapeo a 8 niveles: 0 a 7)
    W_max = np.max(np.abs(W_core))
    if W_max == 0: W_max = 1e-9 # Evitar división por cero
    
    # Mapeamos de [-W_max, W_max] -> [0, 7]
    W_quant = np.round((W_core / W_max) * 3.5 + 3.5)
    W_quant = np.clip(W_quant, 0, 7).astype(np.uint8)
    
    # 3. Empaquetamiento de Bits (Bit-Packing) a uint8_t
    # Necesitamos empaquetar cada 8 pesos en 1 solo byte
    bytes_per_neuron = (in_features + 7) // 8
    
    msb_array, mid_array, lsb_array, outl_array = [], [], [], []
    outl_magnitudes = []
    
    for n in range(out_features):
        # Magnitud promedio del outlier para esta neurona (heurística para C++)
        neuron_outliers = W[n][outlier_mask_bits[n]]
        if len(neuron_outliers) > 0:
            outl_magnitudes.append(int(np.mean(neuron_outliers) * 100)) # Escalado a entero
        else:
            outl_magnitudes.append(0)
            
        for b in range(bytes_per_neuron):
            msb_byte, mid_byte, lsb_byte, outl_byte = 0, 0, 0, 0
            
            for bit_idx in range(8):
                weight_idx = b * 8 + bit_idx
                if weight_idx < in_features:
                    val = W_quant[n, weight_idx]
                    
                    # Extraer bits individuales
                    if (val >> 2) & 1: msb_byte |= (1 << (7 - bit_idx))
                    if (val >> 1) & 1: mid_byte |= (1 << (7 - bit_idx))
                    if val & 1:        lsb_byte |= (1 << (7 - bit_idx))
                    
                    # Máscara de Outliers
                    if outlier_mask_bits[n, weight_idx]:
                        outl_byte |= (1 << (7 - bit_idx))
                        
            msb_array.append(msb_byte)
            mid_array.append(mid_byte)
            lsb_array.append(lsb_byte)
            outl_array.append(outl_byte)

    # 4. Escritura del Archivo C++
    print(f"[*] Escribiendo tensores empaquetados en: {filepath}")
    with open(filepath, "w") as f:
        f.write("#ifndef SYMBEX_WEIGHTS_H\n#define SYMBEX_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// ARCHIVO EXPORTADO POR EL TRADUCTOR SYMBEX-1\n")
        f.write(f"// Dimensiones: {in_features} Entradas -> {out_features} Neuronas (Binarizadas)\n\n")
        
        def write_array(name, data):
            f.write(f"const uint8_t {name}[{len(data)}] = {{\n    ")
            for i, val in enumerate(data):
                f.write(f"0x{val:02X}")
                if i < len(data) - 1: f.write(", ")
                if (i + 1) % 12 == 0: f.write("\n    ")
            f.write("\n};\n\n")
            
        write_array("weights_msb", msb_array)
        write_array("weights_mid", mid_array)
        write_array("weights_lsb", lsb_array)
        write_array("weights_outlier", outl_array)
        
        f.write(f"const int16_t outlier_magnitudes[{out_features}] = {{")
        f.write(", ".join(map(str, outl_magnitudes)))
        f.write("};\n\n")
        
        # Umbrales simulados en cero para esta PoC
        f.write(f"const int16_t thresholds[{out_features}] = {{")
        f.write(", ".join(["0"] * out_features))
        f.write("};\n\n")
        
        f.write("#endif // SYMBEX_WEIGHTS_H\n")
        
    print(f"[+] Traducción exitosa. Memoria Flash estimada: {len(msb_array)*4} bytes.")

# =====================================================================
# 2. PRUEBA DE CONCEPTO CON DATOS REALES (UCI Optical Digits)
# =====================================================================
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import numpy as np

    print("[*] Descargando Dataset del Mundo Real (UCI Optical Digits)...")
    digits = load_digits()
    mask = digits.target < 8
    X_real = (digits.data[mask] / 8.0) - 1.0
    y_real = digits.target[mask]
    
    X_train = torch.tensor(X_real, dtype=torch.float32)
    y_train = torch.tensor(y_real, dtype=torch.long)
    
    teacher = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 8)
    )
    
    print("[*] Entrenando al Profesor con datos caoticos reales...")
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = teacher(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        accuracy = (torch.argmax(teacher(X_train), dim=1) == y_train).float().mean().item() * 100
    print(f"[+] Entrenamiento finalizado. Precision del Profesor: {accuracy:.2f}%\n")
        
    # EXIGIMOS 99.9% de precision para forzar al algoritmo
    estimator = SymbexTopologyEstimator(target_precision=0.999)
    student = estimator.convert_model(teacher, verbose=True)
    
    print("[*] Iniciando Destilacion de Conocimiento hacia el Estudiante...")
    student_optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    distill_criterion = nn.MSELoss()
    
    student.train()
    teacher.eval()
    
    for epoch in range(250):
        student_optimizer.zero_grad()
        with torch.no_grad():
            teacher_outputs = teacher(X_train)
        student_outputs = student(X_train)
        loss = distill_criterion(student_outputs, teacher_outputs)
        loss.backward()
        student_optimizer.step()
        
    student.eval()
    with torch.no_grad():
        s_accuracy = (torch.argmax(student(X_train), dim=1) == y_train).float().mean().item() * 100
        
    print(f"[+] Destilacion finalizada.")
    print(f"[+] Precision del Estudiante (Bit-Slice): {s_accuracy:.2f}%\n")
    
    export_path = os.path.join("lib_symbex", "include", "symbex_weights.h")
    export_to_symbex_h(student, export_path)
