"""Export failures must be visible to callers and must not leave a usable-looking header."""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import cdt  # noqa: E402


def config(tmp_path):
    cfg = copy.deepcopy(cdt.DEFAULTS)
    cfg["model"].update(file=str(tmp_path / "model.pt"), method="plain")
    cfg["data"].update(directory=str(tmp_path), format="npy", mean=[], std=[], input_bits=8)
    cfg["target"].update(options=["bin"], widths=[1.0], search_epochs=1)
    cfg["training"].update(epochs=1, distill=False, seed=42, batch_size=8)
    cfg["output"].update(directory=str(tmp_path / "out"), verbose=False, arithmetic="and", int8_reference=False,
                         validation_samples=8, lib_path=os.path.join(REPO, "Symbex1_PIO/lib/lib_symbex"))
    cfg["run"].update({"from": "train", "to": "export"})
    return cfg


def run_cli(tmp_path, rejected, old_header=False):
    cfg = config(tmp_path)
    torch.manual_seed(42)
    torch.save(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)), cfg["model"]["file"])
    x = np.arange(128, dtype=np.uint8).reshape(32, 4)
    if rejected:
        x = 0.1 + x.astype(np.float32) / 512.0
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", np.arange(32, dtype=np.int64) % 2)
    val = np.zeros(32, dtype=bool)
    val[:8] = True
    np.save(tmp_path / "val.npy", val)
    out = tmp_path / "out"
    out.mkdir()
    if old_header:
        (out / "model_symbex.h").write_text("old header")
    path = tmp_path / "config.toml"
    path.write_text(cdt.to_toml(cfg))
    result = subprocess.run([sys.executable, os.path.join(REPO, "tools/cdt.py"), str(path)],
                            capture_output=True, text=True, check=False, timeout=120)
    return result, out


@pytest.mark.parametrize("old_header", [False, True])
def test_export_refused(tmp_path, old_header):
    result, out = run_cli(tmp_path, rejected=True, old_header=old_header)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "inputs are not 8-bit data" in output, output
    assert "[X] Export failed; see the reason above." in output, output
    assert not (out / "model_symbex.h").exists(), output


def test_export_succeeds(tmp_path):
    assert shutil.which("gcc"), "needs gcc on PATH"
    result, out = run_cli(tmp_path, rejected=False)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert (out / "model_symbex.h").read_text(), output
    assert (out / "export.json").exists(), output
    assert json.loads((out / "export.json").read_text())["c_exactness"] == 100, output


def setup_main(monkeypatch, tmp_path, start="train", end="chip"):
    cfg = config(tmp_path)
    cfg["run"].update({"from": start, "to": end})
    cfg["chip"]["board"] = "esp32"
    monkeypatch.setattr(sys, "argv", ["cdt.py"])
    monkeypatch.setattr(cdt, "load_config", lambda args: (cfg, {}))
    monkeypatch.setattr(cdt, "load_model", lambda cfg: (object(), "test"))
    monkeypatch.setattr(cdt, "stage_inspect", lambda *args: "mlp")
    monkeypatch.setattr(cdt, "load_data", lambda cfg: ([1], [1], None))
    monkeypatch.setattr(cdt, "_quiet_estimate", lambda *args: {})
    monkeypatch.setattr(cdt, "stage_train", lambda *args, **kwargs: {})
    monkeypatch.setattr(cdt, "stage_export", lambda *args: None)
    calls = []
    monkeypatch.setattr(cdt, "load_export", lambda *args: calls.append("load_export") or {})
    monkeypatch.setattr(cdt, "stage_chip", lambda *args: calls.append("chip"))
    out = tmp_path / "out"
    out.mkdir()
    return out, calls


@pytest.mark.parametrize("old_header", [False, True])
def test_none_stops_before_chip(monkeypatch, tmp_path, old_header):
    out, calls = setup_main(monkeypatch, tmp_path)
    header = out / "model_symbex.h"
    if old_header:
        header.write_text("old header")
    with pytest.raises(SystemExit, match="Export failed") as exc:
        cdt.main()
    assert isinstance(exc.value.code, str)  # sys.exit(message) produces exit status 1
    assert not header.exists()
    assert calls == []


def test_cleanup_failure(monkeypatch, tmp_path):
    out, calls = setup_main(monkeypatch, tmp_path)
    header = out / "model_symbex.h"
    header.write_text("old header")
    remove = os.remove

    def fail_header(path):
        if path == str(header):
            raise PermissionError("denied")
        return remove(path)

    monkeypatch.setattr(cdt.os, "remove", fail_header)
    with pytest.raises(SystemExit) as exc:
        cdt.main()
    assert f"[X] Export failed; could not remove {header}:" in exc.value.code
    assert "denied" in exc.value.code
    assert header.exists()
    assert calls == []


@pytest.mark.parametrize("start,end,expected", [("train", "train", []), ("chip", "chip", ["load_export", "chip"])])
def test_other_stages_preserve_header(monkeypatch, tmp_path, start, end, expected):
    out, calls = setup_main(monkeypatch, tmp_path, start, end)
    header = out / "model_symbex.h"
    header.write_text("old header")
    cdt.main()
    assert header.read_text() == "old header"
    assert calls == expected
