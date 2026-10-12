"""
Block-Gated topology definition for conditional 1-bit inference (linear layers only).

Fixes in SYMBEX v3 (the forward pass, and therefore inference and export, is unchanged):
  - the gates now LEARN: the top-k mask is hard in the forward pass but lets a gradient through
    (straight-through, see _mascara); before, gate_weight never received any gradient.
    compuertas_aprenden=False restores the old behavior exactly.
  - out_features no longer has to be a multiple of block_size (the last block may be shorter).
  - scale is kept positive in the forward pass (a negative scale would flip the signs of the next
    binarized layer in PyTorch but not on the chip, where scale is not exported).
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
    def __init__(self, in_features, out_features, block_size=32, active_ratio=1.0, compuertas_aprenden=True):
        """
        Args:
            in_features (int): Number of input characteristics.
            out_features (int): Number of total output neurons.
            block_size (int): Number of neurons grouped per evaluation block.
            active_ratio (float): Fraction of blocks to remain active (0.0 to 1.0).
            compuertas_aprenden (bool): let a gradient reach the gates (default True; False = old behavior).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = max(1, math.ceil(out_features / block_size))     # last block may be shorter
        self.k_active = max(1, int(self.num_blocks * active_ratio))
        self.active_ratio = active_ratio
        self.compuertas_aprenden = compuertas_aprenden

        # Gate Weights: Matrix for block selection
        self.gate_weight = nn.Parameter(torch.empty(self.num_blocks, in_features))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        # Core Weights: Matrix for final neuron evaluation
        self.core_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.core_weight, a=math.sqrt(5))

        # Trainable scaling factor
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _mascara(self, gate_scores):
        """Hard top-k mask per sample. With compuertas_aprenden, a soft version (sigmoid around the k-th
        score) carries the gradient back to the gates; its value is subtracted, so the forward is unchanged."""
        _, topk_indices = torch.topk(gate_scores, self.k_active, dim=-1)
        dura = torch.zeros_like(gate_scores)
        dura.scatter_(1, topk_indices, 1.0)
        if not (self.compuertas_aprenden and self.training):
            return dura
        kth = gate_scores.topk(self.k_active, dim=-1).values[:, -1:].detach()
        tau = gate_scores.detach().std(dim=-1, keepdim=True).clamp(min=1e-6)
        suave = torch.sigmoid((gate_scores - kth) / tau)
        return dura + suave - suave.detach()

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

        # 2-3. Top-K Block Selection and Expansion Mask (cut to out_features: the last block may be shorter)
        mask_expanded = self._mascara(gate_scores).repeat_interleave(self.block_size, dim=1)[:, :self.out_features]

        # 4. Core Inference execution
        core_scores = nn.functional.linear(x, w_core_bin)
        out = core_scores * mask_expanded

        return out * torch.clamp(self.scale, min=1e-4)
