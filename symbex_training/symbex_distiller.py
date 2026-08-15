import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. COMPONENTES BASE (STE y Capa Simulada QAT)
# =====================================================================
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return torch.sign(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

binarize = BinarizeSTE.apply

class SymbexLayerQAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(SymbexLayerQAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Umbrales que también aprenderán durante el entrenamiento
        self.thresholds = nn.Parameter(torch.zeros(out_features))
        self.scales = [4.0, 2.0, 1.0]

    def forward(self, input_data):
        bin_input = binarize(input_data)
        
        w_msb = binarize(self.weight)
        residuo_1 = self.weight - (w_msb * self.scales[0])
        
        w_mid = binarize(residuo_1)
        residuo_2 = residuo_1 - (w_mid * self.scales[1])
        
        w_lsb = binarize(residuo_2)
        
        simulated_weight = (w_msb * self.scales[0]) + (w_mid * self.scales[1]) + (w_lsb * self.scales[2])
        return F.linear(bin_input, simulated_weight)

# =====================================================================
# 2. EL EXPORTADOR A C++ (La magia del hardware)
# =====================================================================
def export_to_symbex_h(qat_layer, filename="symbex_weights.h"):
    print(f"[*] Exportando pesos a {filename}...")
    
    # Extraemos las 3 matrices matemáticas
    w_msb = binarize(qat_layer.weight).detach().numpy()
    residuo_1 = qat_layer.weight.detach().numpy() - (w_msb * 4.0)
    w_mid = binarize(torch.tensor(residuo_1)).numpy()
    residuo_2 = residuo_1 - (w_mid * 2.0)
    w_lsb = binarize(torch.tensor(residuo_2)).numpy()
    
    thresholds = qat_layer.thresholds.detach().numpy().astype(int)
    
    # Función interna para empaquetar -1/+1 a bits puros (0 y 1) y luego a Hex
    def pack_to_hex_array(weight_matrix):
        hex_array = []
        for neuron_weights in weight_matrix:
            # Convertimos -1 a 0 lógicos
            bits = [1 if w > 0 else 0 for w in neuron_weights]
            # Empaquetamos en bytes de 8 en 8
            for i in range(0, len(bits), 8):
                byte = 0
                for bit_idx in range(8):
                    if i + bit_idx < len(bits):
                        byte |= (bits[i + bit_idx] << bit_idx)
                hex_array.append(f"0x{byte:02X}")
        return hex_array

    msb_hex = pack_to_hex_array(w_msb)
    mid_hex = pack_to_hex_array(w_mid)
    lsb_hex = pack_to_hex_array(w_lsb)
    
    # Escribimos el archivo .h
    with open(filename, 'w') as f:
        f.write("/*\n * ARCHIVO AUTO-GENERADO POR SYMBEX-1 DISTILLER\n")
        f.write(f" * Topologia: Entradas={qat_layer.in_features}, Neuronas={qat_layer.out_features}\n */\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write(f"const uint8_t weights_msb[{len(msb_hex)}] = {{ {', '.join(msb_hex)} }};\n")
        f.write(f"const uint8_t weights_mid[{len(mid_hex)}] = {{ {', '.join(mid_hex)} }};\n")
        f.write(f"const uint8_t weights_lsb[{len(lsb_hex)}] = {{ {', '.join(lsb_hex)} }};\n\n")
        
        th_str = [str(t) for t in thresholds]
        f.write(f"const int16_t thresholds[{len(th_str)}] = {{ {', '.join(th_str)} }};\n")
        
    print("[+] Exportacion completada con exito.")

# =====================================================================
# 3. EL BUCLE DE DESTILACIÓN (El Profesor enseñando al Estudiante)
# =====================================================================
def train_distillation(teacher_model, student_model, dataloader, epochs=50):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)
    # Función de pérdida: Error Cuadrático Medio para imitar las salidas
    criterion = nn.MSELoss() 
    
    teacher_model.eval() # El profesor no aprende, solo enseña
    student_model.train()
    
    print("[*] Iniciando Destilacion de Conocimiento...")
    for epoch in range(epochs):
        total_loss = 0
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            
            # El Profesor da la respuesta maestra (alta precisión)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                
            # El Estudiante intenta predecir sufriendo el Bit-Slicing
            student_outputs = student_model(inputs)
            
            # Forzamos al estudiante a acercarse a la respuesta del profesor
            loss = criterion(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"    - Epoch {epoch}: Perdida = {total_loss:.4f}")
            
    print("[+] Destilacion terminada. Estudiante optimizado.")

# =====================================================================
# 4. EL INSPECTOR Y VALIDADOR (Radiografía de la red)
# =====================================================================
def inspect_and_validate(teacher_model, student_model, test_dataloader):
    print("\n" + "="*50)
    print(" 🔍 INICIANDO INSPECCIÓN Y VALIDACIÓN PROFUNDA")
    print("="*50)
    
    teacher_model.eval()
    student_model.eval()
    
    # ---------------------------------------------------------
    # RADIOGRAFÍA 1: Fidelidad de Salida (Precisión)
    # ---------------------------------------------------------
    total_samples = 0
    exact_matches = 0
    total_error = 0.0
    
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            t_out = teacher_model(inputs)
            s_out = student_model(inputs)
            
            # Error promedio en las salidas (MAE - Mean Absolute Error)
            total_error += torch.nn.functional.l1_loss(s_out, t_out, reduction='sum').item()
            
            # Si las salidas del estudiante son idénticas al profesor (aprox)
            # Para clasificación, solemos buscar el índice con el valor máximo
            t_preds = torch.argmax(t_out, dim=1)
            s_preds = torch.argmax(s_out, dim=1)
            
            exact_matches += (t_preds == s_preds).sum().item()
            total_samples += inputs.size(0)
            
    match_percentage = (exact_matches / total_samples) * 100
    avg_error = total_error / total_samples
    
    print(f"\n[1] TEST DE PREDICCIÓN (Con datos nunca antes vistos):")
    print(f"    - Similitud de Decisiones: {match_percentage:.2f}% (El estudiante imita al profesor)")
    print(f"    - Error Absoluto Promedio de Salida: {avg_error:.4f}")

    # ---------------------------------------------------------
    # RADIOGRAFÍA 2: Inspección Interna de los Pesos
    # ---------------------------------------------------------
    print(f"\n[2] INSPECCIÓN DEL BIT-SLICING (K=3):")
    
    # Extraemos el peso maestro en punto flotante
    w_master = student_model.weight.detach()
    
    # Simulamos el corte en 3 capas
    w_msb = binarize(w_master)
    r1 = w_master - (w_msb * 4.0)
    
    w_mid = binarize(r1)
    r2 = r1 - (w_mid * 2.0)
    
    w_lsb = binarize(r2)
    
    # Reconstruimos usando la matemática del hardware
    w_reconstructed = (w_msb * 4.0) + (w_mid * 2.0) + (w_lsb * 1.0)
    
    # Calculamos la "Amnesia Matemática"
    weight_mae = torch.abs(w_master - w_reconstructed).mean().item()
    max_error = torch.abs(w_master - w_reconstructed).max().item()
    
    print(f"    - Error de cuantización promedio por peso: {weight_mae:.4f}")
    print(f"    - Pérdida máxima en un solo peso: {max_error:.4f}")
    
    # ---------------------------------------------------------
    # RADIOGRAFÍA 3: Densidad de la Red
    # ---------------------------------------------------------
    # Vemos qué porcentaje de bits están encendidos (+1) vs apagados (-1)
    def count_ones(matrix):
        return (matrix > 0).float().mean().item() * 100
        
    print(f"\n[3] DENSIDAD LÓGICA (Porcentaje de 1s lógicos):")
    print(f"    - MSB (Impacto Fuerte): {count_ones(w_msb):.2f}%")
    print(f"    - MID (Impacto Medio) : {count_ones(w_mid):.2f}%")
    print(f"    - LSB (Ajuste Fino)   : {count_ones(w_lsb):.2f}%")
    print("="*50 + "\n")

# =====================================================================
# EJEMPLO DE USO (Flujo Real con Patrones)
# =====================================================================
if __name__ == "__main__":
    # 1. El Profesor: Red profunda tradicional
    teacher = nn.Sequential(
        nn.Linear(64, 1024),
        nn.ReLU(),
        nn.Linear(1024, 8)
    )
    
    # 2. El Estudiante: Red binarizada super ligera adaptada al hardware
    student = SymbexLayerQAT(in_features=64, out_features=8)
    
    print("[*] Generando un dataset con patrones matematicos reales...")
    # Creamos 2000 muestras aleatorias
    x_data = torch.randn(2000, 64)
    
    # CREAMOS UN PATRÓN: Multiplicamos por una matriz fija para que haya una lógica oculta
    matriz_secreta = torch.randn(64, 8) 
    y_data = torch.matmul(x_data, matriz_secreta) # El patrón a descubrir
    
    # Separamos en Entrenamiento (1500) y Test (500)
    train_dataset = torch.utils.data.TensorDataset(x_data[:1500], y_data[:1500])
    test_dataset = torch.utils.data.TensorDataset(x_data[1500:], y_data[1500:])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # ---------------------------------------------------------
    # NUEVO PASO: Entrenar al Profesor primero
    # ---------------------------------------------------------
    print("[*] Entrenando al Profesor (Alta Precision)...")
    prof_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
    prof_criterion = nn.MSELoss()
    
    for epoch in range(30):
        for inputs, targets in train_dataloader:
            prof_optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = prof_criterion(outputs, targets)
            loss.backward()
            prof_optimizer.step()
            
    print("[+] Profesor entrenado con exito. Listo para enseñar.")
    
    # 4. Fase de Destilacion (El estudiante aprende del profesor experto)
    train_distillation(teacher, student, train_dataloader, epochs=50)
    
    # 6. Radiografia de la Red
    inspect_and_validate(teacher, student, test_dataloader)
    
    # 7. Exportacion a C++
    export_to_symbex_h(student, "symbex_weights.h")
