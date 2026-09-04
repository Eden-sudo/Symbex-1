"""
Block-Gated topology definition for conditional 1-bit inference.
"""
import math
import torch
import torch.nn as nn
from .core import BipolarSTE

class SymbexBlockGatedPool(nn.Module):
    """
    Block-Gated 1-bit Linear Layer.
    
    Separates inference into two sequential phases:
    1. Gate Evaluation: Scores blocks of neurons and selects the top-k most active blocks.
    2. Core Inference: Executes the full 1-bit dot product only for the selected blocks.
    """
    def __init__(self, in_features, out_features, block_size=32, active_ratio=1.0):
        """
        Initializes the Block-Gated layer parameters.

        Args:
            in_features (int): Number of input characteristics.
            out_features (int): Number of total output neurons.
            block_size (int): Number of neurons grouped per evaluation block.
            active_ratio (float): Fraction of blocks to remain active (0.0 to 1.0).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = max(1, out_features // block_size)
        self.k_active = max(1, int(self.num_blocks * active_ratio))
        self.active_ratio = active_ratio

        # Gate Weights: Matrix for block selection
        self.gate_weight = nn.Parameter(torch.empty(self.num_blocks, in_features))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        # Core Weights: Matrix for final neuron evaluation
        self.core_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.core_weight, a=math.sqrt(5))
        
        # Trainable scaling factor
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Forward pass emulating the C++ hardware execution flow natively in PyTorch.
        """
        w_gate_bin = BipolarSTE.apply(self.gate_weight)
        w_core_bin = BipolarSTE.apply(self.core_weight)

        # 1. Gate Evaluation
        gate_scores = nn.functional.linear(x, w_gate_bin)
        
        # Deterministic tie-breaker (aligns with C++ insertion sort stability)
        tie_break = -torch.arange(self.num_blocks, dtype=gate_scores.dtype, device=x.device) * 1e-4
        gate_scores = gate_scores + tie_break

        # 2. Top-K Block Selection
        _, topk_indices = torch.topk(gate_scores, self.k_active, dim=-1)

        # 3. Create Expansion Mask
        mask = torch.zeros_like(gate_scores)
        mask.scatter_(1, topk_indices, 1.0)
        mask_expanded = mask.repeat_interleave(self.block_size, dim=1)

        # 4. Core Inference execution
        core_scores = nn.functional.linear(x, w_core_bin)
        out = core_scores * mask_expanded
        
        return out * self.scale
