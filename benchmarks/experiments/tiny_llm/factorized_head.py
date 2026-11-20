"""Opt-in floating low-rank head; no compatibility with the legacy integer export."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

spec = importlib.util.spec_from_file_location("head_train", Path(__file__).with_name("train.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class FactorizedHead(nn.Module):
    """Compute (..., dim) @ A(dim, rank) @ B(rank, vocab), without bias."""

    def __init__(self, dim, vocab, rank):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (dim, vocab, rank)):
            raise ValueError("dim, vocab and rank must be positive integers")
        if rank > min(dim, vocab):
            raise ValueError("rank must not exceed dim or vocab")
        self.A = nn.Parameter(torch.empty(dim, rank))
        self.B = nn.Parameter(torch.empty(rank, vocab))
        nn.init.normal_(self.A, std=dim ** -0.5)
        nn.init.normal_(self.B, std=0.02)

    def forward(self, h):
        return (h @ self.A) @ self.B


class FactorizedTinyLM(train.TinyLM):
    """Explicit alternative model with an untied trainable output head.

    Instantiate before creating the optimizer. Reconstruct with the same rank
    before loading state_dict. The embedding remains dense and independent;
    this is not a low-rank conversion preserving pretrained logits.
    """

    def __init__(self, *args, rank, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_kind != "fp32":
            raise ValueError("factorized head requires head='fp32'")
        self.output_head = FactorizedHead(self.emb.embedding_dim, self.emb.num_embeddings, rank)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, self.cos[:t], self.sin[:t])
        return self.output_head(self.norm(x))
