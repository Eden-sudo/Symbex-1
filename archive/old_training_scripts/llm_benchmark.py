import torch
import time
import math
from transformers import GPT2LMHeadModel
from symbex_compiler import SymbexTopologyEstimator, SymbexClusterQAT

print("\n" + "="*70)
print(" 🚀 SYMBEX-1 vs LARGE LANGUAGE MODEL (GPT-2)")
print("="*70)

# 1. CARGAR EL LLM REAL
print("[*] Descargando/Cargando pesos reales de GPT-2...")
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Extraemos la capa masiva: lm_head (768 entradas -> 50,257 salidas)
# Esta es la capa que decide que palabra sigue en el texto.
target_layer = model.lm_head
in_f = target_layer.in_features
out_f = target_layer.out_features
weights = target_layer.weight.detach()

total_params = in_f * out_f
print(f"[*] Capa extraida exitosamente: {in_f} entradas -> {out_f} salidas")
print(f"[*] Total de conexiones sinapticas: {total_params:,}")

# 2. EVALUACIÓN SYMBEX-1 (COMPILADOR CDT)
print("\n[*] Ejecutando Compilador CDT con deteccion de Outliers...")
# Le exigimos un 99% de retencion de inteligencia verbal
estimator = SymbexTopologyEstimator(target_precision=0.99)

# Aislamiento y perfilado
sq_weights, orig_var, sq_var, out_cnt = estimator._extract_and_squash_outliers(weights)
M, penalty, raw_exp = estimator._calculate_expansion(sq_var)

print(f"    |-- Outliers (Palabras atipicas) detectados : {out_cnt:,} pesos")
print(f"    |-- Varianza Original de la Capa            : {orig_var:.6f}")
print(f"    |-- Varianza Aplastada (Magia SYMBEX)       : {sq_var:.6f}")
print(f"    |-- Factor de Expansion M asignado          : {M}x")

# 3. BENCHMARK DE MEMORIA (EL CUELLO DE BOTELLA DE LOS LLMs)
# Float32: 4 bytes por parametro
float_memory = total_params * 4 

# SYMBEX-1: (Parametros * Factor M) / 8 bits = Bytes por mascara.
# Usamos 4 mascaras (MSB, MID, LSB, Outliers)
symbex_memory = ((total_params * M) / 8) * 4

compression_ratio = (1 - (symbex_memory / float_memory)) * 100

print("\n--- REPORTE DE COMPRESIÓN (MEMORIA) ---")
print(f"[!] Peso Original (Float32) : {float_memory / (1024*1024):.2f} MB")
print(f"[+] Peso SYMBEX-1 (K=3)     : {symbex_memory / (1024*1024):.2f} MB")
print(f"[+] Reduccion total de RAM  : {compression_ratio:.2f}%")

# 4. BENCHMARK DE VELOCIDAD DE INFERENCIA
print("\n--- REPORTE DE VELOCIDAD (SIMULACIÓN PYTORCH) ---")
print("Nota: Python simula el agrupamiento en Float32, por lo que no tiene la aceleracion nativa XNOR del C++.")

# Creamos la capa binarizada
symbex_layer = SymbexClusterQAT(in_f, out_f, M)

# Simulamos la entrada de una palabra (1 token, 768 dimensiones)
dummy_input = torch.randn(1, in_f)

# Prueba Float32
start = time.perf_counter()
for _ in range(100):
    _ = target_layer(dummy_input)
float_time = (time.perf_counter() - start) * 1000

# Prueba SYMBEX
start = time.perf_counter()
for _ in range(100):
    _ = symbex_layer(dummy_input)
symbex_time = (time.perf_counter() - start) * 1000

print(f"[-] Tiempo Original (100 inferencias) : {float_time:.2f} ms")
print(f"[-] Tiempo SYMBEX-1 (100 inferencias) : {symbex_time:.2f} ms")
print("======================================================================")
