"""
Code generation module. Transforms PyTorch tensors into static C/C++ header files.
"""
import os
import torch
import numpy as np
from .models.bitslice import SymbexVotingPool

# =====================================================================
# V1: BITSLICE EXPORTER
# =====================================================================

def _export_layer_to_arrays_v1(weight_np, mean, std, W_max, k_bits=3):
    """
    Internal helper for V1 to slice float weights into bit-planes and extract outliers.
    """
    out_features, in_features = weight_np.shape
    threshold = max(2.0 * std, 1e-4)
    
    outlier_mask = np.abs(weight_np - mean) > threshold
    W_core = np.clip(weight_np, mean - threshold, mean + threshold)
    
    levels = (2 ** k_bits) - 1   
    W_scaled = (W_core / W_max) * (levels / 2.0) + (levels / 2.0)
    W_quant = np.clip(np.round(W_scaled), 0, levels).astype(np.uint8)

    bytes_per_neuron = (in_features + 7) // 8
    planes = [[] for _ in range(k_bits)]
    outl_array = []
    outl_magnitudes = []

    for n in range(out_features):
        neuron_outliers = weight_np[n][outlier_mask[n]]
        if len(neuron_outliers) > 0:
            mag_float = np.mean(np.abs(neuron_outliers))
            mag_quant = int(np.clip(np.round((mag_float / W_max) * levels), 0, levels * 3))
        else:
            mag_quant = 0
            
        outl_magnitudes.append(mag_quant)

        for b in range(bytes_per_neuron):
            plane_bytes = [0] * k_bits
            outl_byte = 0
            for bit_idx in range(8):
                weight_idx = b * 8 + bit_idx
                if weight_idx < in_features:
                    val = int(W_quant[n, weight_idx])
                    for k in range(k_bits):
                        if (val >> (k_bits - 1 - k)) & 1:
                            plane_bytes[k] |= (1 << (7 - bit_idx))
                    if outlier_mask[n, weight_idx]:
                        outl_byte |= (1 << (7 - bit_idx))
            
            for k in range(k_bits):
                planes[k].append(plane_bytes[k])
            outl_array.append(outl_byte)

    return {
        "planes": planes,
        "outliers": outl_array,
        "outlier_magnitudes": outl_magnitudes,
        "in_features": in_features,
        "out_features": out_features,
        "params": in_features * out_features
    }

def export_model_v1_to_h(student, filepath, k_bits=3):
    """
    Exports a Bitslice (V1) PyTorch model to a C++ header file.
    Includes legacy structure generation (SymbexLayer).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    MAX_K_BITS = 4
    
    with open(filepath, "w") as f:
        f.write("#ifndef SYMBEX_WEIGHTS_H\n#define SYMBEX_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n")
        f.write("#include \"SymbexNetwork.h\"\n\n")
        f.write("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
        
        layer_instances = []
        total_bytes = 0
        total_params = 0
        layer_counter = 0
        
        for layer in student:
            if isinstance(layer, SymbexVotingPool):
                in_f = layer.in_features
                out_f = layer.out_features
                M = layer.M
                
                subs_names = []
                
                for m in range(M):
                    w = layer.weight[m].detach().cpu().numpy()
                    mean = layer.running_mean[m].item()
                    std = layer.running_std[m].item()
                    w_max = layer.running_W_max[m].item()
                    
                    exp = _export_layer_to_arrays_v1(w, mean, std, w_max, k_bits)
                    
                    sub_bytes = 0
                    bit_names = []
                    
                    for k, plane in enumerate(exp["planes"]):
                        name = f"layer{layer_counter}_m{m}_bit{k}"
                        bit_names.append(name)
                        f.write(f"static const uint8_t {name}[{len(plane)}] PROGMEM = {{\n   ")
                        f.write(", ".join(f"0x{v:02X}" for v in plane))
                        f.write("\n};\n\n")
                        sub_bytes += len(plane)
                    
                    outl_name = f"layer{layer_counter}_m{m}_outliers"
                    f.write(f"static const uint8_t {outl_name}[{len(exp['outliers'])}] PROGMEM = {{")
                    f.write(", ".join(f"0x{v:02X}" for v in exp["outliers"]))
                    f.write("};\n\n")
                    sub_bytes += len(exp["outliers"])
                    
                    mag_name = f"layer{layer_counter}_m{m}_outlier_mag"
                    f.write(f"static const int8_t {mag_name}[{len(exp['outlier_magnitudes'])}] PROGMEM = {{")
                    f.write(", ".join(map(str, exp["outlier_magnitudes"])))
                    f.write("};\n\n")
                    
                    while len(bit_names) < MAX_K_BITS:
                        bit_names.append("NULL")
                        
                    planes_str = "{" + ", ".join(bit_names) + "}"
                    subs_names.append(f"    {{ {planes_str}, {outl_name}, {mag_name} }}")
                    
                    total_bytes += sub_bytes
                    total_params += (in_f * out_f)
                    
                f.write(f"// --- ESTRUCTURA DE LA CAPA {layer_counter} ---\n")
                f.write(f"static const SymbexSubLayer layer{layer_counter}_subs[{M}] = {{\n")
                f.write(",\n".join(subs_names))
                f.write("\n};\n")
                
                f.write(f"static SymbexLayer symbex_layer_{layer_counter}({in_f}, {out_f}, {M}, {k_bits}, layer{layer_counter}_subs);\n\n")
                
                layer_instances.append(f"symbex_layer_{layer_counter}")
                layer_counter += 1
        
        f.write("// --- RED ARMADA AUTOMÁTICAMENTE ---\n")
        f.write("static SymbexNetwork symbex_net;\n\n")
        f.write("static inline void symbex_init() {\n")
        for inst in layer_instances:
            f.write(f"    symbex_net.add_layer(&{inst});\n")
        f.write("}\n\n")
        
        f.write("#endif // SYMBEX_WEIGHTS_H\n")
        
    return total_bytes, total_params


# =====================================================================
# V2: BLOCK-GATED EXPORTER
# =====================================================================

def export_model_v2_to_h(student, filepath, block_size, k_active):
    """
    Exports a Block-Gated (V2) PyTorch model to a C/C++ header file.
    Uses clean arrays (AoS compatible) without POO structure injection.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    layer0 = student[0]
    w_gate = torch.where(layer0.gate_weight > 0, 1, 0).byte().cpu().numpy()
    w_core = torch.where(layer0.core_weight > 0, 1, 0).byte().cpu().numpy()
    
    in_f = layer0.in_features
    out_f = layer0.out_features
    num_blocks = layer0.num_blocks
    
    def _pack_bits(weight_matrix):
        rows, cols = weight_matrix.shape
        packed = []
        for r in range(rows):
            row_bytes = []
            for b_idx in range(0, cols, 8):
                byte_val = 0
                for bit in range(8):
                    if b_idx + bit < cols:
                        byte_val |= (weight_matrix[r, b_idx + bit] << (7 - bit))
                row_bytes.append(byte_val)
            packed.append(row_bytes)
        return np.array(packed, dtype=np.uint8)

    gate_packed = _pack_bits(w_gate)
    core_packed = _pack_bits(w_core)
    
    layer1 = student[2]
    w_out_core = torch.where(layer1.core_weight > 0, 1, 0).byte().cpu().numpy()
    out_packed = _pack_bits(w_out_core)
    final_classes = layer1.out_features
    
    total_bytes = gate_packed.nbytes + core_packed.nbytes + out_packed.nbytes
    total_params = (num_blocks * in_f) + (out_f * in_f) + (final_classes * out_f)
    
    with open(filepath, 'w') as f:
        f.write("// ==================================================\n")
        f.write("// SYMBEX-1 V2 (BLOCK-GATED 1-BIT) EXPORTED WEIGHTS\n")
        f.write("// ==================================================\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
        
        f.write(f"#define IN_FEATURES_BITS {in_f}\n")
        f.write(f"#define OUT_FEATURES_BITS {out_f}\n")
        f.write(f"#define BLOCK_SIZE_BITS {block_size}\n")
        f.write(f"#define GATE_NUM_BLOCKS {num_blocks}\n")
        f.write(f"#define GATE_K_ACTIVE {k_active}\n")
        f.write(f"#define FINAL_CLASSES {final_classes}\n\n")
        
        f.write(f"const uint8_t gate_weights_bin[{num_blocks}][{(in_f+7)//8}] PROGMEM = {{\n")
        for r in gate_packed:
            f.write("  {" + ", ".join(f"0x{v:02X}" for v in r) + "},\n")
        f.write("};\n\n")
        
        f.write(f"const uint8_t core_weights_bin[{out_f}][{(in_f+7)//8}] PROGMEM = {{\n")
        for r in core_packed:
            f.write("  {" + ", ".join(f"0x{v:02X}" for v in r) + "},\n")
        f.write("};\n\n")

        f.write(f"const uint8_t out_weights_bin[{final_classes}][{(out_f+7)//8}] PROGMEM = {{\n")
        for r in out_packed:
            f.write("  {" + ", ".join(f"0x{v:02X}" for v in r) + "},\n")
        f.write("};\n\n")
        
    return total_bytes, total_params
