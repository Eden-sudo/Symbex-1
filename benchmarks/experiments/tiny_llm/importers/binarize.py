"""Initialize W1A8 fine-tuning from imported float weights (not a trained model)."""
import copy

import torch

LINEARS = ("q", "k", "v", "o", "w1", "w3", "w2")


def binarize_model(model):
    """Return a separate trainable model with materialized sign x row scale.

    Exactly train.py's binary convention: zero is +1, scale is mean(abs(row))
    clamped at 1e-5. Norms and embeddings/classifier stay float. QLinear.kind
    switches on binary weights and int8 activations for subsequent fine-tuning.
    Latent weights remain float Parameters for the straight-through estimator;
    intsim.integerize supplies the packed-export signs and integer multipliers.
    """
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("cannot binarize non-finite model weights")
    result = copy.deepcopy(model)
    with torch.no_grad():
        for block in result.blocks:
            for name in LINEARS:
                linear = getattr(block, name)
                if linear.kind != "fp32" or linear.had_blk:
                    raise ValueError("binary initialization requires unrotated fp32 block linears")
                w = linear.weight
                scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
                w.copy_(torch.where(w >= 0, 1.0, -1.0) * scale)
                linear.kind, linear.acts = "binary", 8
    return result
