"""Smoke tests for distillation using synthetic data, not model-quality measurements."""
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRAIN_DIR = Path(__file__).resolve().parents[1] / "tools" / "compiler_core" / "train"


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def training_modules(monkeypatch):
    package = ModuleType("_distillation_smoke_train")
    package.__path__ = [str(TRAIN_DIR)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    standard = _load(package.__name__ + ".standard", TRAIN_DIR / "standard.py", monkeypatch)
    distillation = _load(package.__name__ + ".distillation", TRAIN_DIR / "distillation.py", monkeypatch)
    return standard, distillation


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _synthetic_loaders():
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,), dtype=torch.long)
    dataset = TensorDataset(x, y)
    return (DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0),
            DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0))


def _state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_train_teacher_returns_same_model_in_eval_mode(training_modules):
    torch.manual_seed(0)
    _, distillation = training_modules
    trainloader, testloader = _synthetic_loaders()
    teacher = nn.Linear(4, 2).train()
    before = _state(teacher)

    model, accuracy = distillation.train_teacher(teacher, trainloader, testloader, epochs=1,
                                                 device="cpu", verbose=False)
    assert model is teacher
    assert not teacher.training
    assert math.isfinite(accuracy) and 0 <= accuracy <= 100
    assert any(not torch.equal(before[name], value) for name, value in teacher.state_dict().items())


def test_distill_student_contract_and_last_checkpoint(training_modules, tmp_path):
    torch.manual_seed(0)
    standard, distillation = training_modules
    trainloader, testloader = _synthetic_loaders()
    teacher, student = nn.Linear(4, 2), nn.Linear(4, 2)
    teacher_before, student_before = _state(teacher), _state(student)
    checkpoint = tmp_path / "student.pt"
    result = distillation.distill_student(student, teacher, trainloader, testloader, epochs=2,
                                          average_last=5, checkpoint_path=checkpoint, save="last",
                                          device="cpu", verbose=False)
    standard_result = standard.train_model(nn.Linear(4, 2), trainloader, testloader, epochs=2,
                                           device="cpu", verbose=False)
    keys = {"best_test", "final_average", "history", "mejor_test", "promedio_final", "historial"}
    assert set(result) == set(standard_result) == keys
    assert result["mejor_test"] == result["best_test"]
    assert result["promedio_final"] == result["final_average"]
    assert result["historial"] == result["history"]
    history = result["history"]
    assert len(history) == 2
    assert all(isinstance(row, tuple) and len(row) == 4 for row in history)
    assert [row[0] for row in history] == [0, 1]
    for _, loss, train_acc, test_acc in history:
        assert math.isfinite(loss)
        assert math.isfinite(train_acc) and 0 <= train_acc <= 100
        assert math.isfinite(test_acc) and 0 <= test_acc <= 100
    assert result["best_test"] == max(row[3] for row in history)
    assert result["final_average"] == pytest.approx(sum(row[3] for row in history) / 2)
    assert any(not torch.equal(student_before[name], value) for name, value in student.state_dict().items())
    assert all(torch.equal(teacher_before[name], value) for name, value in teacher.state_dict().items())
    assert not teacher.training
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(student.state_dict())
    assert all(torch.equal(saved[name], value) for name, value in student.state_dict().items())


def test_legacy_arguments_match_new_arguments(training_modules, monkeypatch):
    _, distillation = training_modules
    real_summary = distillation._summary
    averages = []

    def recording_summary(history, average_last):
        averages.append(average_last)
        return real_summary(history, average_last)

    monkeypatch.setattr(distillation, "_summary", recording_summary)

    def run(**kwargs):
        torch.manual_seed(0)
        trainloader, testloader = _synthetic_loaders()
        teacher, student = nn.Linear(4, 2), nn.Linear(4, 2)
        result = distillation.distill_student(student, teacher, trainloader, testloader, epochs=2,
                                              device="cpu", verbose=False, **kwargs)
        return result, _state(student)

    new, new_state = run(temperature=2.0, average_last=1)
    old, old_state = run(temperatura=2.0, promedio_ultimas=1)
    assert new == old
    assert set(new_state) == set(old_state)
    assert all(torch.equal(new_state[name], old_state[name]) for name in new_state)
    assert averages == [1, 1]
    assert new["final_average"] == new["history"][-1][3]
    assert old["final_average"] == old["history"][-1][3]


def test_teacher_input_only_transforms_teacher_batches(training_modules):
    torch.manual_seed(0)
    _, distillation = training_modules
    trainloader, testloader = _synthetic_loaders()
    teacher, student = nn.Linear(4, 2), nn.Linear(4, 2)
    original = tuple(t.clone() for t in trainloader.dataset.tensors)
    batches = [x.clone() for x, _ in trainloader]
    transformed_inputs, teacher_inputs, student_inputs = [], [], []

    def teacher_input(x):
        transformed_inputs.append(x.detach().clone())
        return x + 10

    def record_teacher(module, args):
        teacher_inputs.append(args[0].detach().clone())

    def record_student(module, args):
        student_inputs.append((module.training, args[0].detach().clone()))

    hooks = [teacher.register_forward_pre_hook(record_teacher), student.register_forward_pre_hook(record_student)]
    try:
        distillation.distill_student(student, teacher, trainloader, testloader, epochs=2,
                                     teacher_input=teacher_input, device="cpu", verbose=False)
    finally:
        for hook in hooks:
            hook.remove()

    expected = batches * 2
    assert len(transformed_inputs) == len(teacher_inputs) == 2 * len(trainloader)
    assert all(torch.equal(actual, batch) for actual, batch in zip(transformed_inputs, expected))
    assert all(torch.equal(actual, batch + 10) for actual, batch in zip(teacher_inputs, expected))
    train_inputs = [x for training, x in student_inputs if training]
    eval_inputs = [x for training, x in student_inputs if not training]
    assert len(train_inputs) == 2 * len(trainloader)
    assert len(eval_inputs) == 2 * len(testloader)
    assert all(torch.equal(actual, batch) for actual, batch in zip(train_inputs, expected))
    assert all(torch.equal(actual, batch) for actual, batch in zip(eval_inputs, expected))
    assert all(torch.equal(before, after) for before, after in zip(original, trainloader.dataset.tensors))
