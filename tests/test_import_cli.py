import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from tiny_import_helpers import TINY, checkpoint, load

cli = load("importers/import_model")


@pytest.mark.parametrize("weights", ["binary", "fp32"])
def test_cli_cpu_run_loaders(tmp_path, monkeypatch, weights):
    torch.set_num_threads(1)
    src, out = tmp_path / "source.bin", tmp_path / "run"
    # 64-wide heads make the integer attention contract valid; only 2 tokens.
    checkpoint(src, dim=64, heads=1, kv=1)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, str(TINY / "importers/import_model.py"),
                           str(src), str(out), "--weights", weights],
                          capture_output=True, text=True, timeout=20, env=env)
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out / "import.json").read_text())
    assert json.loads(proc.stdout) == manifest
    assert manifest["integer_attention_compatible"] and not manifest["trained"]
    ck = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    assert ck["step"] == 0
    train = load("train")
    a = ck["args"]
    model = train.TinyLM(7, a["dim"], a["layers"], a["heads"], a["hidden"], a["seq"], a["weights"])
    model.load_state_dict(ck["model"], strict=True)
    monkeypatch.setitem(sys.modules, "train", train)
    intsim = load("intsim")
    reloaded, args = intsim.load_model(out, 7, "cpu")
    idx = torch.tensor([[1, 2]])
    with torch.no_grad():
        torch.testing.assert_close(model(idx), reloaded(idx), rtol=0, atol=0)
        original = load("importers/mapping").to_model(*load("importers/llama2c").read_checkpoint(src))
        expected = load("importers/binarize").binarize_model(original) if weights == "binary" else original
        torch.testing.assert_close(model(idx), expected(idx), rtol=0, atol=0)
        if weights == "binary":
            params = intsim.integerize(reloaded, "binary")
            scores, scale = intsim.forward(params, idx)
            assert scores.shape == (1, 2, 7)
            assert torch.isfinite(intsim.float_logits(scores, scale)).all()


def test_gqa_run_preserves_expanded_weights(tmp_path):
    src, out = tmp_path / "toy.bin", tmp_path / "run"
    checkpoint(src, kv=1)
    manifest = cli.import_model(src, out, "fp32")
    assert manifest["kv_expanded"] and not manifest["integer_attention_compatible"]
    ck = torch.load(out / "model.pt", weights_only=True)
    w = ck["model"]["blocks.0.k.weight"].numpy()
    np.testing.assert_array_equal(w[:4], w[4:])


@pytest.mark.parametrize("case,match", [("untied", "untied classifier"),
    ("gguf", "GGUF architecture"), ("existing", "already exists"), ("bad", "header")])
def test_reject_without_output(tmp_path, case, match):
    src, out = tmp_path / "toy.bin", tmp_path / "run"
    checkpoint(src, shared=case != "untied")
    if case == "gguf":
        src.write_bytes(b"GGUF")
    if case == "bad":
        src.write_bytes(b"bad")
    if case == "existing":
        out.mkdir()
        (out / "keep").write_text("existing work")
    with pytest.raises(ValueError, match=match):
        cli.import_model(src, out)
    if case == "existing":
        assert [p.name for p in out.iterdir()] == ["keep"]
    else:
        assert not out.exists()
