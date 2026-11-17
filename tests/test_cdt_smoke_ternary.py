"""Exercise the CDT CLI through C export with a small ternary model."""
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


REPO = Path(__file__).resolve().parents[1]


def test_cdt_smoke_ternary(tmp_path, rng):
    assert shutil.which("gcc"), "needs gcc on PATH"

    x = np.concatenate([rng.normal(center, 0.03, (100, 16)) for center in (0.2, 0.5, 0.8)])
    x = np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", np.repeat(np.arange(3), 100))

    net_path = tmp_path / "net.py"
    net_path.write_text(
        "from torch import nn\n\n"
        "class Net(nn.Sequential):\n"
        "    def __init__(self):\n"
        "        super().__init__(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64),\n"
        "                         nn.ReLU(), nn.Linear(64, 3))\n"
    )
    spec = importlib.util.spec_from_file_location("smoke_net", net_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_path = tmp_path / "model.pth"
    torch.save(module.Net().state_dict(), model_path)

    out = tmp_path / "run"
    cfg = copy.deepcopy(cdt.DEFAULTS)
    cfg["model"].update(file=str(model_path), definition=f"{net_path}:Net", method="ternary")
    cfg["data"].update(directory=str(tmp_path), format="npy", split_seed=0)
    cfg["training"].update(epochs=2, batch_size=32)
    cfg["target"].update(search_epochs=1, options=["bin"], widths=[1.0])
    cfg["output"].update(directory=str(out), validation_samples=64,
                         lib_path=str(REPO / "Symbex1_PIO/lib/lib_symbex"))
    cfg["run"]["to"] = "export"
    config_path = tmp_path / "cfg.toml"
    config_path.write_text(cdt.to_toml(cfg))

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(REPO / "tools/cdt.py"), str(config_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=60,
    )
    diagnostic = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, diagnostic
    required = ("config_used.toml", "estimate.json", "train.json", "model_symbex.h",
                "export.json", "validation_x.u8")
    missing = [name for name in required if not (out / name).is_file()]
    assert not missing, f"Missing CDT artifacts: {missing}\n{diagnostic}"
    report = json.loads((out / "export.json").read_text())
    assert report.get("c_exactness") == 100, (
        f"c_exactness={report.get('c_exactness')}; c_note={report.get('c_note')}\n{diagnostic}"
    )
