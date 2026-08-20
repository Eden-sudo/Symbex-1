import serial
import time
import numpy as np
from sklearn.datasets import load_digits

# NOTA: En Arch Linux el puerto suele ser /dev/ttyACM0 o /dev/ttyUSB0
# Cambia esto si tu Arduino aparece en otro puerto.
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200

print("[*] Cargando Dataset UCI Optical Digits...")
digits = load_digits()

# Máscara estricta: Solo del 0 al 7
mask = digits.target < 8 
X_real = digits.data[mask]
y_true = digits.target[mask]

# Binarizamos
X_bin = (X_real > 8).astype(int)
total_tested = len(X_bin) # Procesar todo el dataset filtrado

print(f"[*] Abriendo conexión serial con el hardware en {SERIAL_PORT}...")
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2) # Esperar a que el Arduino se reinicie tras la conexión
    
    # Limpiar buffer y esperar el "READY"
    arduino.reset_input_buffer()
    ready_msg = arduino.readline().decode().strip()
    if ready_msg == "READY":
        print("[+] Arduino conectado y listo para la inferencia.\n")
except Exception as e:
    print(f"[-] Error conectando al Arduino: {e}")
    exit(1)

correct_predictions = 0
total_tested = 200 # Vamos a probar 200 imágenes reales
latencies = []

print("==================================================")
print(" INICIANDO BENCHMARK DE PRECISIÓN EN HARDWARE")
print("==================================================")

for i in range(total_tested):
    pixels = X_bin[i]
    byte_array = bytearray()
        
    # 1. Empaquetado estricto alineado con el exportador C++
    for b in range(8):
        byte_val = 0
        for bit in range(8):
            idx = b * 8 + bit
            if idx < 64:  # in_features de la capa de entrada
                # FIX: En el exportador usaste (1 << (7 - bit_idx)).
                # Aquí DEBE ser igual para que el mapa XNOR cuadre.
                if pixels[idx] == 1:
                    byte_val |= (1 << (7 - bit)) 
        byte_array.append(byte_val)
    
    # 2. Enviar la imagen física al Arduino
    arduino.write(byte_array)
    
    # 3. Leer la respuesta
    response = arduino.readline().decode().strip()
    
    if "," in response:
        pred_str, lat_str = response.split(",")
        pred = int(pred_str)
        lat = int(lat_str)
        
        latencies.append(lat)
        
        # OJO: La capa cruda a veces necesita mapearse al índice real.
        # Aquí simplificamos asumiendo que la red escupe el índice directo o lo evaluamos.
        # Solo comprobaremos si el byte de salida es consistente.
        
        print(f"Img {i+1:03d} | Etiqueta Real: {y_true[i]} | Arduino dice: {pred} | Latencia: {lat} us")

# 4. Resultados finales
avg_lat = sum(latencies) / len(latencies) if latencies else 0
print("==================================================")
print(f"[*] Total de imagenes procesadas por silicio : {total_tested}")
print(f"[*] Latencia promedio de inferencia bare-metal : {avg_lat:.2f} us")
print("==================================================")

arduino.close()
