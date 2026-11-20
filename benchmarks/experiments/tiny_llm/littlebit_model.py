"""Explicit, opt-in conversion of tiny_llm block projections before optimizer creation."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("littlebit", Path(__file__).with_name("littlebit.py"))
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


class FactorAct8Linear(lb.FactorLinear):
    def forward(self, x):
        return super().forward(lb._train.quant_act8(x))


def replace_linears(model, *, rank, names=("w1", "w3"), residual=False, itq_iters=None, seed=0):
    """Replace selected projections in all blocks, using their effective weights.

    Call explicitly after loading weights and before constructing an optimizer.
    Unsupported activation rotations are rejected before any model mutation.
    """
    names = tuple(names)
    if len(set(names)) != len(names) or not set(names) <= {"q", "k", "v", "o", "w1", "w3", "w2"}:
        raise ValueError("names must be unique block projection names")
    replacements = []
    for block in model.blocks:
        for name in names:
            old = getattr(block, name)
            if not isinstance(old, lb.nn.Linear) or old.bias is not None:
                raise ValueError("only bias-free linear projections are supported")
            if getattr(old, "had_blk", 0) or getattr(old, "acts", 8) != 8:
                raise ValueError("conversion requires unrotated 8-bit activations")
            weight = old.weight.detach()
            kind = getattr(old, "kind", "fp32")
            if kind != "fp32":
                weight = lb._train.quant_weight(weight, kind)
            new = FactorAct8Linear.from_dense(weight, rank, residual, itq_iters=itq_iters, seed=seed)
            new.train(old.training)
            replacements.append((block, name, new))
    for block, name, new in replacements:
        setattr(block, name, new)
    return model
