"""
Hardware-accurate Numpy simulators for SYMBEX-1 bit-level verification.
Ensures PyTorch graphs match C++ bitwise execution exactly.
"""
import torch
import numpy as np

def simulate_bitslice_cpp(student, x_bipolar, k_bits=3):
    """
    Simulates the Multi-Bit (V1) C++ inference engine using pure Numpy.
    Reconstructs quantized matrices and applies hardware-equivalent outlier compensation.

    Args:
        student (nn.Sequential): PyTorch student model.
        x_bipolar (torch.Tensor or np.ndarray): Bipolar input array (-1.0 and 1.0).
        k_bits (int): Bit resolution of the simulated architecture.

    Returns:
        np.ndarray: Final neuron scores matching C++ int32_t buffers.
    """
    current = x_bipolar.detach().cpu().numpy().astype(np.float32) if hasattr(x_bipolar, 'detach') else np.array(x_bipolar, dtype=np.float32)
    levels = (2 ** k_bits) - 1   
    
    for layer in student:
        layer_name = type(layer).__name__
        
        if layer_name == "SymbexVotingPool":
            votes_sum = np.zeros((current.shape[0], layer.out_features), dtype=np.float32)
            
            for m in range(layer.M):
                W_master = layer.weight[m].detach().cpu().numpy().astype(np.float32)
                
                mean = layer.running_mean[m].item()
                std = layer.running_std[m].item()
                W_max = layer.running_W_max[m].item()
                
                threshold = max(2.0 * std, 1e-4)
                outlier_mask = np.abs(W_master - mean) > threshold
                W_core = np.clip(W_master, mean - threshold, mean + threshold)
                
                W_scaled = (W_core / W_max) * (levels / 2.0) + (levels / 2.0)
                W_quant = np.clip(np.round(W_scaled), 0, levels).astype(np.float32)
                
                W_reconstructed = 2.0 * W_quant - levels
                
                if outlier_mask.any():
                    sign_msb = np.where(W_quant >= (levels + 1)/2, 1.0, -1.0).astype(np.float32)
                    for n in range(layer.out_features):
                        neuron_outliers = W_master[n][outlier_mask[n]]
                        if len(neuron_outliers) > 0:
                            mag_float = np.mean(np.abs(neuron_outliers))
                            mag_quant = np.clip(np.round((mag_float / W_max) * levels), 0, levels * 3)
                            W_reconstructed[n, outlier_mask[n]] += mag_quant * sign_msb[n, outlier_mask[n]]
                            
                votes_sum += current @ W_reconstructed.T
            current = votes_sum
            
        elif layer_name == "BipolarStepSTE":
            current = np.where(current > 0, 1.0, -1.0).astype(np.float32)
            
    return current

def simulate_block_gated_cpp(student, X_test_np):
    """
    Simulates the Block-Gated (V2) C++ inference engine using pure Numpy.
    Executes binary XNOR mapping, deterministic sorting, and early-exit mechanisms.

    Args:
        student (nn.Sequential): PyTorch student model containing SymbexBlockGatedPoolV2 layers.
        X_test_np (np.ndarray): Floating point test dataset (will be thresholded to 1/0).

    Returns:
        np.ndarray: Predicted class indices matching C++ output exactly.
    """
    X_bits = np.where(X_test_np > 0, 1, 0).astype(np.uint8)
    
    layer0 = student[0]
    layer1 = student[2]
    
    # Weight packaging to binary format
    w_gate = torch.where(layer0.gate_weight > 0, 1, 0).cpu().numpy().astype(np.uint8)
    w_core = torch.where(layer0.core_weight > 0, 1, 0).cpu().numpy().astype(np.uint8)
    w_out  = torch.where(layer1.core_weight > 0, 1, 0).cpu().numpy().astype(np.uint8)
    
    num_blocks = layer0.num_blocks
    block_size = layer0.block_size
    k_active = layer0.k_active
    out_features = layer0.out_features
    
    predictions = []
    
    for sample in X_bits:
        # Phase 1: Gate Evaluation
        gate_scores = []
        for b in range(num_blocks):
            match = (sample == w_gate[b]).astype(int)
            score = np.sum(match)
            gate_scores.append(score)
            
        gate_scores = np.array(gate_scores, dtype=np.float32)
        
        # Tie-breaker (simulates C++ insertion sort stability)
        for b in range(num_blocks):
            gate_scores[b] -= b * 1e-4
            
        topk_indices = np.argsort(gate_scores)[::-1][:k_active]
        active_blocks = np.zeros(num_blocks, dtype=bool)
        active_blocks[topk_indices] = True
        
        # Phase 2: Core Inference (active blocks only)
        hidden_out_bits = np.zeros(out_features, dtype=np.uint8)
        for n in range(out_features):
            b_idx = n // block_size
            if not active_blocks[b_idx]:
                continue
                
            match_core = (sample == w_core[n]).astype(int)
            score_core = np.sum(match_core)
            
            # Hardware C++ Threshold (score > in_features / 2)
            if score_core > (layer0.in_features / 2):
                hidden_out_bits[n] = 1
                
        # Phase 3: Output Argmax
        best_class = 0
        max_score = -1
        final_classes = layer1.out_features
        
        for c in range(final_classes):
            match_out = (hidden_out_bits == w_out[c]).astype(int)
            score_out = np.sum(match_out)
            if score_out > max_score:
                max_score = score_out
                best_class = c
                
        predictions.append(best_class)
        
    return np.array(predictions)
