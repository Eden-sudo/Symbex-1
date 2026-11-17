"""The checkpoint audit delegates its summary to the training health helper."""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "dev", "binary_health.py")
spec = importlib.util.spec_from_file_location("binary_health_dev_test_helper",
                                              os.path.join(REPO, "tools", "compiler_core", "train", "health.py"))
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)
EMPTY = "    no checkpoint found (or none above --limit)\n"


def run_audit(tmp_path, *args):
    result = subprocess.run([sys.executable, SCRIPT, *map(str, args)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"exit {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    return result.stdout


def summary(path, weights):
    h = health.binary_health([w.float() for w in weights])
    flag = " [!]" if h["frozen"] > 1 else ""
    return f'{h["frozen"]:6.1f} % frozen  max|w| {h["max_abs"]:7.3f}  {h["n"]:9d} w  {path.name}{flag}\n'


@pytest.mark.parametrize("wrapped", [False, True])
def test_summary_and_tensors(tmp_path, wrapped):
    weights = {"a.weight": torch.tensor([[1.0, -1.0, 1.5], [-2.0, 0.0, 0.5]], dtype=torch.float64),
               "b.weight": torch.tensor([[0.2, -0.3], [0.4, 3.0]])}
    state = dict(weights, **{"a.bias": torch.tensor([100.0]), "c.weight": torch.tensor([100.0]),
                             "d.weight": torch.tensor([[100]], dtype=torch.int64)})
    path = tmp_path / "model.pth"
    torch.save({"state_dict": state} if wrapped else state, path)
    expected = summary(path, weights.values())
    assert expected.rstrip().endswith(" [!]")
    assert run_audit(tmp_path, path) == expected
    lines = ""
    for name, tensor in weights.items():
        w = tensor.float()
        lines += (f"          {100.0 * float((w.abs() > 1).float().mean()):6.1f} %  max|w| {float(w.abs().max()):7.3f}"
                  f"  {tuple(w.shape)}  {name}\n")
    assert run_audit(tmp_path, path, "--tensors") == expected + lines
    frozen = health.binary_health([w.float() for w in weights.values()])["frozen"]
    for limit in (frozen, frozen + 1):
        assert run_audit(tmp_path, path, "--tensors", "--limit", limit) == EMPTY


def test_empty_and_excluded(tmp_path):
    assert run_audit(tmp_path, tmp_path) == EMPTY
    path = tmp_path / "excluded.pth"
    torch.save({"x.bias": torch.ones(2, 2), "x.weight": torch.ones(2)}, path)
    assert run_audit(tmp_path, path) == EMPTY


def test_flag_and_zero_limits(tmp_path):
    path = tmp_path / "boundary.pth"
    w = torch.zeros(1, 100)
    w[0, 0] = 2
    torch.save({"x.weight": w}, path)
    assert run_audit(tmp_path, path) == summary(path, [w])
    assert "[!]" not in summary(path, [w])
    w.zero_()
    torch.save({"x.weight": w}, path)
    for limit in (0, -1):
        assert run_audit(tmp_path, path, "--limit", limit) == summary(path, [w])


def test_rounded_values_control_output(tmp_path):
    path = tmp_path / "round.pth"
    w = torch.tensor([[1.23451]], dtype=torch.float32)
    torch.save({"x.weight": w}, path)
    assert run_audit(tmp_path, path) == summary(path, [w])
    w = torch.zeros(1, 10099)
    w[0, :101] = 2
    torch.save({"x.weight": w}, path)
    assert health.binary_health([w])["frozen"] == 1.0
    assert run_audit(tmp_path, path) == summary(path, [w])
    assert "[!]" not in summary(path, [w])
    assert run_audit(tmp_path, path, "--limit", 1) == EMPTY


def test_corrupt_checkpoint_does_not_stop_next(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    good = tmp_path / "good.pth"
    w = torch.tensor([[2.0]])
    torch.save({"x.weight": w}, good)
    output = run_audit(tmp_path, bad, good)
    first, rest = output.split("\n", 1)
    assert first.startswith(f"    [?] {bad.name}: not read (") and first.endswith(")")
    assert first.split("(")[-1][:-1].isidentifier()
    assert rest == summary(good, [w])
