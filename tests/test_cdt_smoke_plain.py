"""The CDT end to end (tools/cdt.py, main): inspect, estimate, train and export on a tiny synthetic dataset.

Method plain, CPU, no chip stage. The stages run through main() as the command line does; a stage that exits, a missing
stage report or a C exactness below 100 % fails the test.
"""
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import cdt  # noqa: E402

LIB_SYMBEX = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")

MODEL_SOURCE = """import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                 nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)
"""

REPORTS = ["config_used.toml", "estimate.json", "model_binarized.pth", "train.json", "predictions.npz",
           "model_symbex.h", "export.json", "validation_x.u8", "validation_y.u8", "validation_sim.u8"]


def write_data(folder):
    folder.mkdir()
    rng = np.random.default_rng(0)
    y = np.arange(300) % 3
    v = rng.integers(0, 64, size=(300, 16))
    for c in range(3):
        v[y == c, 5 * c:5 * c + 5] += 192                  # class c: features 5c..5c+4 high
    np.save(folder / "x.npy", (v / 255.0).astype(np.float32))   # 8-bit exact, as the exporter needs
    np.save(folder / "y.npy", y)


def write_model(folder):
    import torch
    py_path = folder / "tiny_mlp.py"
    py_path.write_text(MODEL_SOURCE)
    spec = importlib.util.spec_from_file_location("tiny_mlp_smoke", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    pth_path = folder / "tiny_mlp.pth"
    torch.save(module.TinyMLP().state_dict(), pth_path)
    return pth_path, f"{py_path}:TinyMLP"


def config(tmp_path, model_file, definition):
    cfg = copy.deepcopy(cdt.DEFAULTS)
    cfg["model"].update(file=str(model_file), definition=definition, method="plain")
    cfg["data"].update(format="npy", directory=str(tmp_path / "data"), mean=[], std=[], input_bits=8,
                       validation_split=0.25, split_seed=0)
    cfg["target"].update(options=["bin"], widths=[1.0], search_epochs=1)
    cfg["training"].update(epochs=2, batch_size=25)        # 225 training samples: no last batch of one for BatchNorm
    cfg["output"].update(directory=str(tmp_path / "out"), validation_samples=64, lib_path=LIB_SYMBEX, verbose=False)
    cfg["run"].update({"from": "inspect", "to": "export"})
    return cfg


def test_cdt_plain_mlp_runs_inspect_to_export_with_bit_exact_header(tmp_path, monkeypatch, capsys):
    if cdt.tomllib is None:
        pytest.skip("the CDT needs tomllib (Python 3.11+)")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(sys, "path", list(sys.path))       # load_model inserts the model folder
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    write_data(tmp_path / "data")
    model_file, definition = write_model(tmp_path)
    config_path = tmp_path / "smoke.toml"
    config_path.write_text(cdt.to_toml(config(tmp_path, model_file, definition)))
    monkeypatch.setattr(sys, "argv", ["cdt.py", str(config_path)])

    try:
        cdt.main()
    except SystemExit as e:
        pytest.fail(f"CDT exited: {e.code}\n{capsys.readouterr().out}")

    out = capsys.readouterr().out
    folder = tmp_path / "out"
    for name in REPORTS:
        assert (folder / name).exists(), f"{name} missing; CDT output:\n{out}"
    assert not (folder / "chip.json").exists(), "the chip stage ran although run.to is export"

    header = (folder / "model_symbex.h").read_text()
    assert "#define SYMBEX_MLP_N_INPUTS 16" in header
    assert "symbex_mlp_infer" in header

    train = json.loads((folder / "train.json").read_text())
    assert train["plan"] == {"net.0": "bin", "net.3": "bin", "net.6": "bin"}
    assert len(json.loads((folder / "estimate.json").read_text())["layers"]) == 3

    ex = json.loads((folder / "export.json").read_text())
    assert ex["n"] == 64
    assert ex["arithmetic"] == "and"
    if ex["c_exactness"] is None:
        if ex["c_note"].startswith("gcc not found"):
            pytest.skip(ex["c_note"])
        pytest.fail(f"C exactness not checked: {ex['c_note']}")
    assert ex["c_exactness"] == 100, f"exported header not bit-exact: {ex['c_exactness']}% ({ex['c_note']})"
