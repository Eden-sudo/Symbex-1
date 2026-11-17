"""Exercise the CDT CLI through export on a small synthetic dataset."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cdt
import numpy as np
import torch


def test_cdt_smoke_plain(tmp_path, rng):
    assert shutil.which("gcc"), "needs gcc on PATH"
    repo = Path(__file__).resolve().parents[1]
    x = np.concatenate([rng.normal(center, 0.03, (100, 16)) for center in (0.2, 0.5, 0.8)])
    x = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", np.repeat(np.arange(3, dtype=np.int64), 100))

    net = tmp_path / "net.py"
    net.write_text("""import torch.nn as nn

class Net(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
""", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("smoke_net", net)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.save(module.Net().state_dict(), tmp_path / "model.pth")

    cfg = copy.deepcopy(cdt.DEFAULTS)
    cfg["model"].update(file=str(tmp_path / "model.pth"), definition=f"{net}:Net", method="plain")
    cfg["data"].update(directory=str(tmp_path), format="npy", split_seed=0)
    cfg["training"].update(epochs=2, batch_size=32)
    cfg["target"].update(search_epochs=1, options=["bin"], widths=[1.0])
    out = tmp_path / "run"
    cfg["output"].update(directory=str(out), validation_samples=64,
                         lib_path=str(repo / "Symbex1_PIO/lib/lib_symbex"))
    cfg["run"].update(to="export")
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cdt.to_toml(cfg), encoding="utf-8")

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        result = subprocess.run([sys.executable, str(repo / "tools/cdt.py"), str(cfg_path)],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"CDT timed out after 60 s\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}") from exc
    diag = f"CDT exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, diag
    required = ("config_used.toml", "estimate.json", "train.json", "model_symbex.h", "export.json",
                "validation_x.u8")
    missing = [name for name in required if not (out / name).is_file()]
    assert not missing, f"Missing CDT artifacts: {missing}\n{diag}"
    report = json.loads((out / "export.json").read_text(encoding="utf-8"))
    assert report.get("c_exactness") == 100, (
        f"c_exactness={report.get('c_exactness')}; c_note={report.get('c_note')}\n{diag}"
    )
