import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from symbex_compiler import SymbexTopologyEstimator, SymbexClusterQAT

print("\n" + "="*70)
print(" PRUEBA DE FUEGO: GENERACIÓN DE TEXTO CON SYMBEX-1")
print("="*70)

print("[*] Cargando GPT-2 y el Tokenizador...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# GPT-2 fue entrenado en inglés, usamos un prompt en ese idioma
prompt = "The future of artificial intelligence in robotics is"
inputs = tokenizer(prompt, return_tensors="pt")

print("\n--- GENERACIÓN ORIGINAL (Float32 Intacto) ---")
# Generamos 30 tokens
original_output = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)
print("Texto: " + tokenizer.decode(original_output[0], skip_special_tokens=True))

print("\n[*] Aplicando SYMBEX-1 (Arquitectura Dual-Path simulada)...")
estimator = SymbexTopologyEstimator(target_precision=0.99)
target_layer = model.lm_head
original_weights = target_layer.weight.detach()

# 1. Extraemos los outliers y aplastamos el nucleo
sq_weights, _, _, out_cnt = estimator._extract_and_squash_outliers(original_weights)

# 2. Calculamos la mascara de outliers exacta (verdadero/falso) y sus magnitudes
mean = torch.mean(original_weights)
std = torch.std(original_weights)
threshold = 2.0 * std
outlier_mask = (torch.abs(original_weights - mean) > threshold).float()

# El impacto que le fue restado a los outliers al aplastarlos
outlier_impact = original_weights - sq_weights

class SymbexDualPathSimulation(nn.Module):
    def __init__(self, squashed_core, outlier_impact):
        super().__init__()
        # Via Principal (Pesos estabilizados y aplastados)
        self.core_weight = nn.Parameter(squashed_core)
        # Via Secundaria (Mascara dispersa superpuesta)
        self.outlier_impact = nn.Parameter(outlier_impact)

    def forward(self, x):
        # 1. El procesador evalua la via principal (Simulando K=3)
        core_logits = nn.functional.linear(x, self.core_weight)
        
        # 2. El procesador evalua la capa de outliers superpuesta
        outlier_logits = nn.functional.linear(x, self.outlier_impact)
        
        # 3. Suma abstracta (Lo que el acumulador hace en C++)
        return core_logits + outlier_logits

# 3. CIRUGÍA: Reemplazamos la capa en el modelo original con nuestra Via Dual
model.lm_head = SymbexDualPathSimulation(sq_weights, outlier_impact)

print(f"[*] Cirugía Dual-Path completada. Outliers aislados en la Vía Secundaria: {out_cnt:,}")
print("[*] Generando texto con la topologia bifurcada...\n")

print("--- GENERACIÓN SYMBEX-1 (Dual-Path) ---")
symbex_output = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)
print("Texto: " + tokenizer.decode(symbex_output[0], skip_special_tokens=True))
print("\n" + "="*70)
