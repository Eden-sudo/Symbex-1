"""Exporter for the Block-Gated (V2) vector method. The header writer is the original one, unchanged."""
import os
import numpy as np
import torch

from .report import print_report


def _write_header_v2(student, filepath, block_size, k_active):
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


def _row(name, shape):
    out, inp = shape
    return dict(name=name, shape=f"{inp}->{out}", n_weights=out * inp, bits=1,
                weights_bytes=out * ((inp + 7) // 8), params_bytes=0)


def export_model_v2_to_h(student, filepath, block_size, k_active, detail=False):
    """Writes the header, prints the input/output report. Returns (total_bytes, total_params)."""
    result = _write_header_v2(student, filepath, block_size, k_active)
    rows = [_row("GATE_0", tuple(student[0].gate_weight.shape)),
            _row("CORE_0", tuple(student[0].core_weight.shape)),
            _row("OUT", tuple(student[2].core_weight.shape))]
    # the output layer's own gate (2.gate_weight) is never read at inference
    print_report(student, rows, filepath, detail=detail, skipped=("2.gate_weight",))
    return result
