import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Función de Binarización con STE (Straight-Through Estimator)
# ---------------------------------------------------------
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # Hacia adelante (Inferencia Simulada): 
        # Convertimos los números decimales a +1 o -1 (1 lógico y 0 lógico)
        # Esto simula exactamente cómo verá los datos el XNOR en C++
        return torch.sign(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        # Hacia atrás (Entrenamiento):
        # El STE hace trampa. Ignora que binarizamos los datos y deja 
        # pasar el gradiente intacto para que los pesos decimales puedan ajustarse.
        return grad_output

# Instanciamos la función para usarla fácilmente
binarize = BinarizeSTE.apply

# ---------------------------------------------------------
# 2. Capa Simuladora de Bit-Slicing (K=3) para QAT
# ---------------------------------------------------------
class SymbexLayerQAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(SymbexLayerQAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Estos son los pesos "Maestros" en punto flotante.
        # Aquí se acumulan los gradientes sutiles de la GPU.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Factores de escala para el Bit-Slicing simulado
        # MSB vale 4, MID vale 2, LSB vale 1 (Simulando los << en C++)
        self.scales = [4.0, 2.0, 1.0]

    def forward(self, input_data):
        # 1. Binarizamos la entrada (Simulando los sensores de 1 bit)
        bin_input = binarize(input_data)

        # 2. Extraemos las 3 hojas semitransparentes del peso maestro
        # (Esta es una simplificación matemática para el QAT)

        # Capa MSB (Los trazos fuertes)
        w_msb = binarize(self.weight)
        residuo_1 = self.weight - (w_msb * self.scales[0])

        # Capa MID (Los detalles medios)
        w_mid = binarize(residuo_1)
        residuo_2 = residuo_1 - (w_mid * self.scales[1])

        # Capa LSB (El ajuste fino)
        w_lsb = binarize(residuo_2)

        # 3. Reconstruimos el peso usando los multiplicadores de hardware
        # Esto es equivalente al (sum_msb << 2) + (sum_mid << 1) + sum_lsb de C++
        simulated_hardware_weight = (w_msb * self.scales[0]) + \
                                    (w_mid * self.scales[1]) + \
                                    (w_lsb * self.scales[2])

        # 4. Ejecutamos la multiplicación matricial final
        return F.linear(bin_input, simulated_hardware_weight)
