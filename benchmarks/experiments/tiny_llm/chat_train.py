"""Opt-in causal chat loss; leaves train.py and its training recipe unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


def causal_loss(logits, token_ids, loss_mask=None):
    """Mean next-token CE, optionally over True target positions only.

    logits: [batch, time, vocab]; IDs/mask: [batch, time]. The mask is aligned
    to tokens (chat_data), NOT prediction positions. Position zero is never a
    target. Unselected labels may be padding sentinels; selected IDs must be
    valid vocabulary IDs. Reject zero targets rather than producing NaN.
    """
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or token_ids.shape[1] < 2:
        raise ValueError("expected aligned batch/time dimensions and at least two tokens")
    if token_ids.dtype != torch.long or token_ids.device != logits.device:
        raise ValueError("token IDs must be int64 on the logits device")
    if loss_mask is None:
        selected = torch.ones_like(token_ids[:, 1:], dtype=torch.bool)
    else:
        if (loss_mask.shape != token_ids.shape or loss_mask.dtype != torch.bool
                or loss_mask.device != logits.device):
            raise ValueError("mask must be aligned boolean data on the logits device")
        selected = loss_mask[:, 1:]
    if not selected.any():
        raise ValueError("batch has no selected next-token targets")
    targets = token_ids[:, 1:][selected]
    if ((targets < 0) | (targets >= logits.shape[-1])).any():
        raise ValueError("selected target outside vocabulary")
    return F.cross_entropy(logits[:, :-1][selected], targets)


class ChatTrainingModel(nn.Module):
    """Wrap a train.TinyLM instance; no CLI/default recipe changes.

    forward returns (logits, loss). Omitting loss_mask trains all next tokens.
    Save model.state_dict() (the wrapped model) for existing export consumers.
    All input IDs must be valid even where the loss mask is False.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids, loss_mask=None):
        logits = self.model(token_ids)
        return logits, causal_loss(logits, token_ids, loss_mask)
