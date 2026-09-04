"""
Hardware constraint validation and bit-level fidelity verification.
Prevents the export of models that violate C++ physical memory limits or mathematical logic.
"""
import torch
import numpy as np
from .simulator import simulate_bitslice_cpp, simulate_block_gated_cpp

def validate_hardware_limits(student, mode):
    """
    Enforces strict C++ physical limitations before allowing compilation.

    Args:
        student (nn.Sequential): PyTorch student model.
        mode (str): Topology mode ("bitslice" or "gated").
        
    Raises:
        AssertionError: If the model architecture exceeds static C++ bounds.
    """
    if mode == "gated":
        layer0 = student[0]
        layer_name = type(layer0).__name__
        
        if "SymbexBlockGatedPool" in layer_name:
            # Check maximum static block allocation limit in C++ (SYMBEX_MAX_BLOCKS)
            assert layer0.num_blocks <= 64, \
                f"[!] Hardware Error: num_blocks ({layer0.num_blocks}) exceeds C++ limit (64)."
            
            # Check memory safety limits (ESP32 RAM bounds for bits arrays)
            assert layer0.out_features <= 4096, \
                f"[!] Warning: Extreme hidden size ({layer0.out_features}) may overflow MCU stack."

    elif mode == "bitslice":
        for layer in student:
            if type(layer).__name__ == "SymbexVotingPool":
                # Ensure the bit-slicing depth aligns with C++ switch statements
                assert 1 <= layer.k_bits <= 8, \
                    f"[!] Hardware Error: k_bits ({layer.k_bits}) exceeds physical register limits (1-8)."
                
                # Protect against excessive Flash usage from extreme expansion (M)
                assert layer.M <= 16, \
                    f"[!] Warning: Expansion factor M={layer.M} is exceptionally high. Risk of Flash overflow."

def verify_fidelity(student, X_test_tensor, X_test_np, mode, k_bits=1):
    """
    Cross-checks PyTorch tensor execution against the Numpy hardware simulator.

    Args:
        student (nn.Sequential): PyTorch student model.
        X_test_tensor (torch.Tensor): FP32 test dataset.
        X_test_np (np.ndarray): FP32 test dataset as a Numpy array.
        mode (str): Topology mode ("bitslice" or "gated").
        k_bits (int, optional): Quantization bits (for bitslice).

    Returns:
        float: Percentage of exact agreement between PyTorch and the C++ simulation.
    """
    student.eval()
    
    with torch.no_grad():
        torch_out = student(X_test_tensor)
        torch_preds = torch.argmax(torch_out, dim=1).cpu().numpy()

    if mode == "gated":
        sim_preds = simulate_block_gated_cpp(student, X_test_np)
        
    elif mode == "bitslice":
        sim_out = simulate_bitslice_cpp(student, X_test_tensor, k_bits=k_bits)
        sim_preds = np.argmax(sim_out, axis=1)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

    agreement = np.mean(torch_preds == sim_preds) * 100.0
    return agreement
