"""CPU contract tests for the standard training helpers."""
import importlib.util
import pickle
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(scope="module")
def standard():
    name = "standard_for_contract_tests"
    path = Path(__file__).resolve().parents[1] / "tools/compiler_core/train/standard.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("momentum", [None, 0.5])
def test_fix_batchnorm_only_changes_default_momentum(standard, momentum):
    other = nn.Identity()
    other.momentum = 0.1
    model = nn.Sequential(nn.BatchNorm1d(1), nn.Sequential(nn.BatchNorm2d(1), nn.BatchNorm3d(1)),
                          nn.BatchNorm1d(1, momentum=0.2), nn.BatchNorm2d(1, momentum=0.0),
                          nn.BatchNorm3d(1, momentum=None), other)
    args = () if momentum is None else (momentum,)
    assert standard.fix_batchnorm(model, *args) == 3
    assert [model[0].momentum, model[1][0].momentum, model[1][1].momentum] == [momentum or 0.3] * 3
    assert [model[2].momentum, model[3].momentum, model[4].momentum, other.momentum] == [0.2, 0.0, None, 0.1]
    assert standard.fix_batchnorm(model, *args) == 0
    assert standard.fix_batchnorm(nn.Linear(1, 1), *args) == 0


def _check_saved(path, model, marker):
    if marker is None:
        assert not path.exists()
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state) == {"weight"}
        assert torch.equal(state["weight"], torch.full_like(model.weight, marker))


def _run_epochs(checkpoints, path, accuracies, expected):
    model = nn.Linear(1, 1, bias=False)
    assert not path.exists()
    for epoch, (accuracy, marker) in enumerate(zip(accuracies, expected)):
        with torch.no_grad():
            model.weight.fill_(epoch + 1)
        checkpoints.after_epoch(model, epoch, len(accuracies), accuracy)
        _check_saved(path, model, marker)


@pytest.mark.parametrize("save", ["last", "ultima"])
@pytest.mark.parametrize("every,expected", [
    (2, [None, 2, 2, 4, 5]),
    (None, [None, None, None, None, 5]),
    (1, [1, 2, 3, 4, 5]),
    (10, [None, None, None, None, 5]),
])
def test_checkpoints_last_schedule(standard, tmp_path, save, every, expected):
    path = tmp_path / "checkpoint.pt"
    checkpoints = standard.Checkpoints(path, save=save, save_every=every, verbose=False)
    _run_epochs(checkpoints, path, [5, 4, 3, 2, 1], expected)


def test_checkpoints_default_single_epoch(standard, tmp_path):
    path = tmp_path / "checkpoint.pt"
    checkpoints = standard.Checkpoints(path, verbose=False)
    _run_epochs(checkpoints, path, [0], [1])


@pytest.mark.parametrize("save", ["best", "mejor"])
@pytest.mark.parametrize("every", [None, 2])
def test_checkpoints_best_only_strict_improvements(standard, tmp_path, save, every):
    path = tmp_path / "checkpoint.pt"
    checkpoints = standard.Checkpoints(path, save=save, save_every=every, verbose=False)
    _run_epochs(checkpoints, path, [0, 50, 50, 40, 45, 60, 55], [1, 2, 2, 2, 2, 6, 6])


@pytest.mark.parametrize("path_is_none", [False, True])
def test_checkpoints_invalid_options(standard, tmp_path, path_is_none):
    path = None if path_is_none else tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match="^save must be 'last' or 'best', got 'invalid'$"):
        standard.Checkpoints(path, save="invalid")
    for every in (0, -1):
        with pytest.raises(ValueError, match="^save_every must be >= 1$"):
            standard.Checkpoints(path, save_every=every)


@pytest.mark.parametrize("save", ["last", "ultima", "best", "mejor"])
def test_checkpoints_none_path_writes_nothing(standard, tmp_path, monkeypatch, save):
    def unexpected_save(*args, **kwargs):
        pytest.fail("checkpoint attempted to save with path=None")

    monkeypatch.setattr(standard.torch, "save", unexpected_save)
    checkpoints = standard.Checkpoints(None, save=save, save_every=1, verbose=False)
    model = nn.Linear(1, 1, bias=False)
    for epoch in range(3):
        checkpoints.after_epoch(model, epoch, 3, epoch)
    assert list(tmp_path.iterdir()) == []


def test_checkpoints_legacy_method_is_alias(standard, tmp_path):
    assert standard.Checkpoints.despues_de_epoca is standard.Checkpoints.after_epoch
    path = tmp_path / "checkpoint.pt"
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(7)
    standard.Checkpoints(path, verbose=False).despues_de_epoca(model, 0, 1, 0)
    _check_saved(path, model, 7)


@pytest.mark.parametrize("training", [True, False])
def test_evaluate_accuracy_and_restores_mode(standard, training):
    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x):
            self.calls.append((self.training, torch.is_grad_enabled()))
            return x

    model = RecordingModel().train(training)
    logits = torch.tensor([[2., 0.], [0., 2.], [2., 0.], [0., 2.], [2., 0.]])
    labels = torch.tensor([0, 1, 1, 0, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    assert standard.evaluate(model, loader, torch.device("cpu")) == pytest.approx(60)
    assert model.calls == [(False, False)] * 3
    assert model.training is training


def _quantizer_input():
    byte_values = torch.arange(256, dtype=torch.float32) / 255
    fractions = (torch.tensor([0., 31., 127., 254.]).unsqueeze(1) + torch.tensor([0.25, 0.75])) / 255
    return torch.cat((byte_values, fractions.flatten()))


def _quantized(x, bits):
    rounded = torch.round(x * 255).to(torch.int64)
    step = 2 ** (8 - bits)
    return ((rounded // step) * step).float() / 255


def test_input_bit_quantizer_eight_bits_identity(standard):
    x = _quantizer_input()
    original = x.clone()
    assert standard.InputBitQuantizer(8)(x) is x
    assert torch.equal(x, original)


@pytest.mark.parametrize("bits", range(1, 8))
def test_input_bit_quantizer_keeps_top_bits(standard, bits):
    x = _quantizer_input()
    original = x.clone()
    result = standard.InputBitQuantizer(bits)(x)
    assert result.shape == x.shape
    assert torch.equal(result, _quantized(original, bits))
    assert torch.equal(x, original)


@pytest.mark.parametrize("bits", range(1, 9))
def test_input_bit_quantizer_pickle_roundtrip(standard, bits):
    quantizer = pickle.loads(pickle.dumps(standard.InputBitQuantizer(bits)))
    assert type(quantizer) is standard.InputBitQuantizer
    assert quantizer.shift == 8 - bits
    x = _quantizer_input()
    result = quantizer(x)
    if bits == 8:
        assert result is x
    else:
        assert torch.equal(result, _quantized(x, bits))


@pytest.mark.parametrize("average_last,average", [(1, 60), (2, 50), (4, 50), (10, 50)])
def test_summary_best_trailing_average_and_aliases(standard, average_last, average):
    history = [(0, 1., 10, 20), (1, 0.8, 30, 80), (2, 0.6, 50, 40), (3, 0.4, 70, 60)]
    original = history.copy()
    result = standard._summary(history, average_last)
    assert result["best_test"] == pytest.approx(80)
    assert result["final_average"] == pytest.approx(average)
    assert result["mejor_test"] == result["best_test"]
    assert result["promedio_final"] == result["final_average"]
    assert result["historial"] is result["history"] is history
    assert history == original
    single = standard._summary(history[:1], average_last)
    assert single["best_test"] == pytest.approx(20)
    assert single["final_average"] == pytest.approx(20)
