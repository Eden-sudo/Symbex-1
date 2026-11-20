import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_five_cpu_steps_on_sensor_toy(tmp_path):
    t, tokenizer, pre = load("train.py"), load("tokenizer.py"), load("pretokenize.py")
    sp = tokenizer.train_tokenizer(tmp_path / "tok", rows=41)
    row = {"event": {"source": "fomo_voc_person", "class": "person", "count": 1},
           "text": "Detected person objects: 1."}
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val"):
        pre.write_split([row] * 8, sp, data / f"{split}.bin")
    # No test.bin exists: training must not read it.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        history = t.train(data, tmp_path / "run", vocab=sp.vocab_size(), steps=5,
                          dim=16, layers=1, heads=2, hidden=32, seq=32, lr=0.02)
    finally:
        torch.set_num_threads(old_threads)
    assert len(history) == 6 and history[-1]["step"] == 5
    assert all(math.isfinite(row[key]) for row in history for key in ("train", "val"))
    assert history[-1]["train"] < history[0]["train"]
    assert history[-1]["val"] < history[0]["val"]
    checkpoint = torch.load(tmp_path / "run/model.pt", weights_only=True)
    assert checkpoint["val_loss"] == min(row["val"] for row in history)
    model, _ = t.build_model(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    assert isinstance(model, t.base.TinyLM)


def test_invalid_config_and_short_data(tmp_path):
    t = load("train.py")
    for config in ({"dim": 15}, {"heads": 0}, {"layers": 0}, {"kind": "unknown"}):
        with pytest.raises(ValueError):
            t.build_model(512, **config)
    for name in ("train", "val"):
        np.array([1, 2], dtype=np.uint16).tofile(tmp_path / f"{name}.bin")
    with pytest.raises(ValueError, match="seq"):
        t.train(tmp_path, tmp_path / "out", vocab=512, dim=8, heads=1, layers=1, hidden=8)
    assert not (tmp_path / "out").exists()
