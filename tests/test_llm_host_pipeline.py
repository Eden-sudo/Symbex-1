"""Synthetic checkpoint -> binary model -> factor blobs -> exact host C."""
import ctypes
import subprocess

import pytest
import torch

from tiny_import_helpers import TINY, checkpoint, load


@pytest.mark.parametrize("residual", [False, True])
def test_import_factor_export_matvec(tmp_path, residual):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source, run = tmp_path / "llama.bin", tmp_path / "run"
        checkpoint(source)
        manifest = load("importers/import_model").import_model(source, run, "binary")
        saved = torch.load(run / "model.pt", weights_only=True, map_location="cpu")
        args = saved["args"]
        train = load("train")
        model = train.TinyLM(manifest["vocab_size"], args["dim"], args["layers"],
                             args["heads"], args["hidden"], args["seq"], args["weights"])
        model.load_state_dict(saved["model"], strict=True)
        assert model.blocks[0].w1.kind == "binary"
        untouched = model.blocks[0].q
        load("littlebit_model").replace_linears(
            model, rank=3, names=("w1", "w2"), residual=residual, itq_iters=2, seed=19)
        assert model.blocks[0].q is untouched
        blob = load("littlebit_blob")
        library = tmp_path / "factor.so"
        subprocess.run(["/usr/bin/clang", "-std=c99", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", str(TINY / "c/littlebit_blob.c"), "-o", str(library)],
                       check=True, capture_output=True, timeout=15)
        kernel = ctypes.CDLL(str(library)).littlebit_blob_matvec
        kernel.argtypes = [ctypes.c_void_p, ctypes.c_size_t] * 3
        kernel.restype = ctypes.c_int
        rng = torch.Generator().manual_seed(23)
        nonzero = False
        for i, block in enumerate(model.blocks):
            for name in ("w1", "w2"):
                params = blob.integer.integerize(getattr(block, name))
                path = tmp_path / f"block{i}_{name}.lbt"
                path.write_bytes(blob.dumps(params))
                data = path.read_bytes()
                decoded = blob.loads(data)
                assert decoded == params
                raw = ctypes.create_string_buffer(data)
                n, o = params["n_in"], params["n_out"]
                vectors = [[0] * n, [-128] * n, [127] * n,
                           torch.randint(-128, 128, (n,), generator=rng).tolist()]
                for values in vectors:
                    expected = blob.integer.forward_int(params, values)
                    x, y = (ctypes.c_int8 * n)(*values), (ctypes.c_int32 * o)()
                    assert kernel(raw, len(data), x, n, y, o) == 0
                    assert list(y) == expected
                    nonzero |= any(expected)
        assert nonzero, "a zero-only comparison cannot establish the integration"
    finally:
        torch.set_num_threads(previous)
