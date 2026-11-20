"""Lossless float mapping into Eden's TinyLM; GQA is expanded to MHA."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_local(relative):
    """Load only a sibling source file, without changing sys.path or globals."""
    path = Path(__file__).resolve().parent / relative
    spec = importlib.util.spec_from_file_location("tiny_import_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_local("../train.py")


class UntiedTinyLM(T.TinyLM):
    """Float-only adapter; stock train/intsim run loading requires tied heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = torch.nn.Linear(self.emb.embedding_dim, self.emb.num_embeddings, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[:idx.shape[1]], self.sin[:idx.shape[1]])
        return F.linear(self.norm(x), self.classifier.weight)


def to_model(config, tensors):
    """Copy v0 weights to a CPU float model, preserving its logits.

    run.c recomputes RoPE with base 10000, ignoring the legacy stored tables.
    Adjacent rotary pairs already match train.py; no Q/K permutation is needed.
    Repeating whole KV heads preserves GQA but increases parameter/storage cost.
    """
    c = config
    cls = T.TinyLM if c["shared_classifier"] else UntiedTinyLM
    model = cls(c["vocab_size"], c["dim"], c["n_layers"], c["n_heads"],
                c["hidden_dim"], c["seq_len"], "fp32")
    state = {"emb.weight": tensors["token_embedding_table"],
             "norm.weight": tensors["rms_final_weight"]}
    if not c["shared_classifier"]:
        state["classifier.weight"] = tensors["wcls"]
    names = dict(n1="rms_att_weight", n2="rms_ffn_weight", q="wq", k="wk",
                 v="wv", o="wo", w1="w1", w2="w2", w3="w3")
    hd = c["dim"] // c["n_heads"]
    for i in range(c["n_layers"]):
        for dst, src in names.items():
            w = tensors[src][i]
            if dst in ("k", "v"):
                w = np.repeat(w.reshape(c["n_kv_heads"], hd, c["dim"]),
                              c["n_heads"] // c["n_kv_heads"], axis=0).reshape(c["dim"], c["dim"])
            state[f"blocks.{i}.{dst}.weight"] = w
    if not all(np.isfinite(w).all() for w in state.values()):
        raise ValueError("cannot map non-finite model weights")
    model.load_state_dict({k: torch.from_numpy(np.array(w, dtype=np.float32, copy=True))
                           for k, w in state.items()}, strict=True)
    return model.eval()
