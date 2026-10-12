"""
Multi-Bit (Bitslice) topology definition with dynamic expansion (Voting Pool).
"""
import math
import torch
import torch.nn as nn

class SymbexVotingPool(nn.Module):
    """
    Multi-Bit Quantization Aware Training (QAT) Layer.
    
    Expands a single logical layer into 'M' physical parallel planes (Voting Pool)
    to recover capacity lost during aggressive bit-slicing quantization.
    """
    def __init__(self, in_features, out_features, expansion_factor=1, k_bits=3):
        """
        Initializes the Bitslice layer parameters.

        Args:
            in_features (int): Number of input characteristics.
            out_features (int): Number of total output neurons.
            expansion_factor (int): Multiplier 'M' for parallel voting planes.
            k_bits (int): Quantization resolution (1 to 8 bits).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.M = max(1, expansion_factor)
        self.k_bits = k_bits
        self.momentum = 0.1
        
        self.weight = nn.Parameter(torch.empty(self.M, out_features, in_features))
        for m in range(self.M):
            nn.init.kaiming_uniform_(self.weight[m], a=math.sqrt(5))
            
        self.register_buffer('running_mean', torch.zeros(self.M))
        self.register_buffer('running_std', torch.ones(self.M))
        self.register_buffer('running_W_max', torch.ones(self.M))
        self.register_buffer('initialized', torch.tensor(False))
        
        self.output_scale = nn.Parameter(torch.tensor(0.05))

    def _quantize_weights(self, W, m):
        """
        Applies dynamic thresholding and quantization mapping to the weight matrix.
        """
        levels = (2 ** self.k_bits) - 1   
        
        if self.training:
            with torch.no_grad():
                cur_mean = torch.mean(W)
                cur_std = torch.std(W).clamp(min=1e-9)
                
                if not self.initialized:
                    self.running_mean[m].copy_(cur_mean)
                    self.running_std[m].copy_(cur_std)
                else:
                    self.running_mean[m].mul_(1 - self.momentum).add_(self.momentum * cur_mean)
                    self.running_std[m].mul_(1 - self.momentum).add_(self.momentum * cur_std)
                
        mean = self.running_mean[m].clone()
        std = self.running_std[m].clone().clamp(min=1e-9)
        
        threshold = torch.clamp(2.0 * std, min=1e-4)
        outlier_mask = (torch.abs(W - mean) > threshold).detach()
        W_core = torch.clamp(W, mean - threshold, mean + threshold)
        
        if self.training:
            with torch.no_grad():
                cur_W_max = torch.max(torch.abs(W_core)).clamp(min=1e-9)
                if not self.initialized:
                    self.running_W_max[m].copy_(cur_W_max)
                else:
                    self.running_W_max[m].mul_(1 - self.momentum).add_(self.momentum * cur_W_max)
                    
        W_max = self.running_W_max[m].clone().clamp(min=1e-9)
        
        W_scaled = (W_core / W_max) * (levels / 2.0) + (levels / 2.0)
        W_quant = torch.round(W_scaled) - W_scaled.detach() + W_scaled   
        W_quant = torch.clamp(W_quant, 0, levels)
        
        W_reconstructed = 2.0 * W_quant - levels
        
        # Outlier compensation routing
        if outlier_mask.any():
            outlier_vals = W * outlier_mask.float()
            sum_abs = torch.sum(torch.abs(outlier_vals), dim=1, keepdim=True)
            count = torch.sum(outlier_mask.float(), dim=1, keepdim=True).clamp(min=1)
            
            outlier_mag_float = sum_abs / count
            scaled_mag = (outlier_mag_float / W_max) * levels
            scaled_mag = torch.clamp(scaled_mag, 0, levels * 3.0)
            
            outlier_mag_quant = torch.round(scaled_mag) - scaled_mag.detach() + scaled_mag
            sign_msb = torch.where(W_quant >= (levels + 1)/2, 1.0, -1.0).detach()
            
            W_reconstructed = torch.where(
                outlier_mask,
                W_reconstructed + (outlier_mag_quant * sign_msb),
                W_reconstructed
            )
            
        return W_reconstructed

    def forward(self, x):
        """
        Forward pass executing the Voting Pool ensemble mechanism.
        """
        votes = []
        for m in range(self.M):
            W_rec = self._quantize_weights(self.weight[m], m)
            votes.append(nn.functional.linear(x, W_rec))
            
        if self.training and not self.initialized:
            self.initialized.fill_(True)
            
        stacked = torch.stack(votes, dim=0)
        safe_scale = torch.clamp(self.output_scale, min=1e-4)
        
        return stacked.sum(dim=0) * safe_scale
