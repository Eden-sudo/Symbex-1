"""The CDT end to end (tools/cdt.py): inspect, estimate, train and export, method ternary, on tiny synthetic data.

The CDT is called the way its main() does (a TOML config on the command line), on CPU, with the real defaults except
for the sizes. Any stage that raises or stops fails the test; the export must write its header and its C exactness
must be 100 % (only a missing gcc leaves the C check out, with a warning).
"""
import copy
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import cdt  # noqa: E402
from compiler_core.export.dense import load_validation  # noqa: E402

LIB_SYMBEX = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")
N, N_IN, N_CLASSES, N_VAL_EXPORT = 300, 16, 3, 64

MODEL_SOURCE = '''
import torch.nn as nn

class TinyMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(16, 32, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                         nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                         nn.Linear(32, 3))
'''


def write_data(folder):
    rng = np.random.default_rng(0)
    y = np.arange(N) % N_CLASSES
    centers = rng.uniform(0.15, 0.85, (N_CLASSES, N_IN))
    x = np.clip(centers[y] + rng.normal(0, 0.04, (N, N_IN)), 0, 1)
    x = (np.round(x * 255) / 255).astype(np.float32)       # the export needs 8-bit inputs
    folder.mkdir()
    np.save(folder / "x.npy", x)
    np.save(folder / "y.npy", y)
    return x, y


def write_model(folder, x, y):
    py_path = folder / "tiny_mlp.py"
    py_path.write_text(MODEL_SOURCE)
    spec = importlib.util.spec_from_file_location("tiny_mlp_smoke", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    model = module.TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(xt), yt).backward()
        optimizer.step()
    model.eval()
    pth_path = folder / "tiny_mlp.pth"
    torch.save(model.state_dict(), pth_path)
    return str(pth_path), f"{py_path}:TinyMLP"


def smoke_config(tmp_path, model_file, definition):
    cfg = copy.deepcopy(cdt.DEFAULTS)
    cfg["model"].update(file=model_file, definition=definition, method="ternary")
    cfg["data"].update(format="npy", directory=str(tmp_path / "data"), validation_split=0.25, split_seed=0)
    cfg["target"].update(search_epochs=1, max_accuracy_drop=100.0, widths=[1.0])
    cfg["training"].update(epochs=2, batch_size=25, seed=0)
    cfg["output"].update(directory=str(tmp_path / "out"), validation_samples=N_VAL_EXPORT, lib_path=LIB_SYMBEX,
                         verbose=False, int8_reference=False)
    cfg["run"].update({"from": "inspect", "to": "export"})
    return cfg


def run_cdt(cfg, tmp_path, monkeypatch):
    toml = tmp_path / "smoke.toml"
    toml.write_text(cdt.to_toml(cfg))
    monkeypatch.setattr(sys, "argv", ["cdt.py", str(toml)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    start = time.monotonic()
    try:
        cdt.main()
    except SystemExit as e:
        pytest.fail(f"the CDT stopped: {e}")
    return time.monotonic() - start


def test_cdt_ternary_runs_inspect_to_export_and_the_header_is_bit_exact(tmp_path, monkeypatch):
    x, y = write_data(tmp_path / "data")
    model_file, definition = write_model(tmp_path, x, y)
    elapsed = run_cdt(smoke_config(tmp_path, model_file, definition), tmp_path, monkeypatch)
    assert elapsed < 120, f"the CDT took {elapsed:.0f} s"

    out = tmp_path / "out"
    for name in ("config_used.toml", "estimate.json", "model_binarized.pth", "train.json", "predictions.npz",
                 "model_symbex.h", "export.json", "validation_x.u8"):
        assert (out / name).exists(), f"{name} was not written"
    assert not (out / "chip.json").exists(), "the chip stage ran although run.to = export"

    train = json.loads((out / "train.json").read_text())
    assert len(train["plan"]) == 3 and all(option == "bin" for option in train["plan"].values())
    assert train["width"] == 1.0

    header = (out / "model_symbex.h").read_text()
    for define in ("#define SYMBEX_MLP_TERNARY 1", f"#define SYMBEX_MLP_N_INPUTS {N_IN}",
                   f"#define SYMBEX_MLP_N_CLASSES {N_CLASSES}"):
        assert define in header

    ex = json.loads((out / "export.json").read_text())
    assert ex["method"] == "ternary" and ex["n"] == N_VAL_EXPORT
    assert ex["bit_exact"]["equal"] is True

    vx, vy, sim = load_validation(str(out), N_IN)
    assert vx.shape == (N_VAL_EXPORT, N_IN) and len(vy) == N_VAL_EXPORT and len(sim) == N_VAL_EXPORT

    if ex["c_exactness"] is None:
        if ex["c_note"].startswith("gcc not found"):
            warnings.warn(f"C exactness not checked: {ex['c_note']}")
            return
        pytest.fail(f"C exactness not checked: {ex['c_note']}")
    assert ex["c_exactness"] == 100, ex["c_note"]
