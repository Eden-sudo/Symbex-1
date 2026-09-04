"""
Core mathematical operations and activation functions for the SYMBEX-1 compiler.
"""
import torch
import torch.nn as nn

class BipolarSTE(torch.autograd.Function):
    """
    Bipolar Step function with Straight-Through Estimator (STE) for backpropagation.
    
    Maps inputs to {-1.0, 1.0} during the forward pass and passes gradients 
    transparently if the input magnitude is within the [-1.0, 1.0] range.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # 0.0 maps to -1.0 to align strictly with the C++ XNOR hardware logic
        return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Cancel gradients for values outside the clipping range
        grad_input[x.abs() > 1.0] = 0
        return grad_input

class BipolarStepSTE(nn.Module):
    """
    PyTorch Module wrapper for the BipolarSTE autograd function.
    """
    def forward(self, x):
        return BipolarSTE.apply(x)
