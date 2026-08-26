"""
VALIDADOR SERIAL UNIVERSAL (SYMBEX V1 y V2)
Soporta el formato viejo (X,YYY) y el nuevo (Prediccion: X | Latencia: YYY us)
"""
import serial
import time
import re
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def main():
    print("[*] 1. Preparando munición para validación masiva...")
    digits = load_digits()
    X_all = np.where(digits.data > 8, 1, 0)
    y_all = digits.target
    
    # 1. Usar el random_state=42 para tener exactamente el mismo set de prueba oficial
    _, X_test_np, _, y_test_np = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # 2. Barajar el ORDEN de las muestras de forma completamente aleatoria en cada ejecución
    shuffle_idx = np.random.permutation(len(X_test_np))
    X_test_np = X_test_np[shuffle_idx]
    y_test_np = y_test_np[shuffle_idx]

    try:
        ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=2)
        time.sleep(2) # Dar tiempo a que el puerto se abra
    except Exception as e:
        print(f"[!] Error abriendo puerto /dev/ttyUSB0: {e}")
        return

    print(f"[*] Blanco: {len(X_test_np)} imágenes listas para enviar.\n")
    print("[*] Iniciando fuego a discreción...\n")
    
    correct = 0
    total = 0
    latencies = []

    for i in range(len(X_test_np)):
        # 1. Empaquetar imagen a 8 bytes
        img_bytes = bytearray(8)
        for b in range(8):
            byte_val = 0
            for bit in range(8):
                if X_test_np[i][b * 8 + bit]:
                    byte_val |= (1 << (7 - bit))
            img_bytes[b] = byte_val
        
        # 2. Enviar y leer
        ser.write(img_bytes)
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        # 3. Parseador Universal
        pred = -1
        lat = -1
        
        if ',' in line and "Prediccion" not in line:
            # Formato V1: "5,526"
            parts = line.split(',')
            if len(parts) == 2:
                pred = int(parts[0])
                lat = int(parts[1])
        elif "Prediccion:" in line:
            # Formato V2: "Prediccion: 5 | Latencia: 586 us"
            m = re.search(r'Prediccion:\s*(\d+).*?Latencia:\s*(\d+)', line)
            if m:
                pred = int(m.group(1))
                lat = int(m.group(2))
        
        if pred == -1:
            print(f"[!] Basura serial en img {i:03d}: {line}")
            continue
            
        total += 1
        latencies.append(lat)
        expected = y_test_np[i]
        
        if pred == expected:
            correct += 1
            acc = correct / total * 100
            print(f"   -> Progreso: {total:03d}/{len(X_test_np)} | Precisión: {acc:.2f}% | Lat. actual: {lat} us")
        else:
            print(f"[!] FALLA en img {i:03d}: Esperaba {expected}, HW predijo {pred} | Latencia: {lat} us")

    print("\n==================================================")
    print(" REPORTE DE VALIDACIÓN EN HARDWARE (UNIVERSAL)")
    print("==================================================")
    if total > 0:
        print(f" - Muestras procesadas : {total}")
        print(f" - Precisión Real      : {correct/total*100:.2f}% ({correct}/{total})")
        print(f" - Latencia Promedio   : {int(np.mean(latencies))} us")
        print(f" - Latencia Mínima     : {np.min(latencies)} us")
        print(f" - Latencia Máxima     : {np.max(latencies)} us")
    print("==================================================")

if __name__ == '__main__':
    main()
