import sys

import pytest
import torch

from tiny_import_helpers import checkpoint, load

mapping = load("importers/mapping")
reader = load("importers/llama2c")
binary = load("importers/binarize")


def test_binary_values_and_integer_multipliers(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    path = tmp_path / "toy.bin"
    checkpoint(path)
    model = mapping.to_model(*reader.read_checkpoint(path))
    with torch.no_grad():
        model.blocks[0].q.weight[0].zero_()
        model.blocks[0].q.weight[1] = torch.tensor([-4., -2., 0., 2., 4., 0., 1., -1.])
    original = {k: v.clone() for k, v in model.state_dict().items()}
    result = binary.binarize_model(model)
    monkeypatch.setitem(sys.modules, "train", mapping.T)
    intsim = load("intsim")
    integer = intsim.integerize(result, "binary")
    for i, block in enumerate(result.blocks):
        for name in binary.LINEARS:
            key = f"blocks.{i}.{name}.weight"
            w = original[key]
            scale = w.abs().mean(1, keepdim=True).clamp(min=1e-5)
            signs = torch.where(w >= 0, 1., -1.)
            linear = getattr(block, name)
            torch.testing.assert_close(linear.weight, signs * scale, rtol=0, atol=0)
            torch.testing.assert_close(linear.weight, mapping.T.quant_weight(w, "binary"))
            assert linear.kind == "binary" and linear.acts == 8
            assert linear.weight.requires_grad and linear.weight.shape == w.shape
            torch.testing.assert_close(integer["blocks"][i][name]["w"], signs.long())
            expected_m = torch.round(scale[:, 0].double() / 127 * 2**30).long()
            torch.testing.assert_close(integer["blocks"][i][name]["m"], expected_m, rtol=0, atol=0)
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        if not any(key.endswith(f".{n}.weight") for n in binary.LINEARS):
            torch.testing.assert_close(result.state_dict()[key], value, rtol=0, atol=0)
    result(torch.tensor([[1, 2]])).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in result.parameters())


def test_nonfinite_rejected(tmp_path):
    path = tmp_path / "toy.bin"
    checkpoint(path)
    model = mapping.to_model(*reader.read_checkpoint(path))
    with torch.no_grad():
        model.emb.weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        binary.binarize_model(model)
