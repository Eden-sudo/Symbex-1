"""Autoregressive Python/C sampler parity; model forward stays in Python."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location("det_loss", Path(__file__).with_name("eval_loss.py"))
loss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss)


def c_sampler(build_dir):
    """Compile only the small, existing C sampler, never firmware."""
    src = Path(__file__).with_name("c")
    library = Path(build_dir) / "chat_sample.so"
    subprocess.run(["/usr/bin/clang", "-shared", "-fPIC", "-std=c99", "-O1",
                    str(src / "chat_sample.c"), "-o", str(library)],
                   check=True, capture_output=True, timeout=20)
    lib = ctypes.CDLL(str(library))
    fn = lib.chat_sample
    fn.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int,
                   ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                   ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    fn.restype = ctypes.c_int

    def sample(scores, k, temperature, top_p, state):
        # Python validates before narrowing values through the C ABI.
        loss.local("sample_int").sample(scores, k, temperature, top_p, state)
        n = len(scores)
        rng = ctypes.c_uint32(state)
        token = fn((ctypes.c_int64 * n)(*scores), n, k, temperature, top_p,
                   ctypes.byref(rng), (ctypes.c_int * n)(), n)
        if token < 0:
            raise RuntimeError("C sampler rejected validated input")
        return token, rng.value
    return sample


def generate(next_scores, prompt, steps, seed, sampler, k, temperature, top_p):
    ids, state, states = list(prompt), seed, []
    generated = []
    for _ in range(steps):
        scores = next_scores(list(ids))
        token, state = sampler(scores, k, temperature, top_p, state)
        ids.append(token)
        generated.append(token)
        states.append(state)
    return {"ids": generated, "states": states, "final_state": state}


def check(next_scores, prompt, steps, seed, output, *, k=0, temperature=1, top_p=65536):
    """Each run gets independent history; next_scores must be a pure callback.

    Fixed budget, no EOS policy. Record mismatches as evidence, not success.
    This validates samplers, not C transformer forward or tokenizer decoding.
    """
    if type(steps) is not int or steps <= 0 or not prompt:
        raise ValueError("positive steps and a nonempty prompt are required")
    if any(type(i) is not int or i < 0 for i in prompt):
        raise ValueError("prompt must contain nonnegative integer IDs")
    py = loss.local("sample_int").sample
    py([0], k, temperature, top_p, seed)
    with tempfile.TemporaryDirectory(prefix="tiny-eval-") as directory:
        c = c_sampler(directory)
        runs = [generate(next_scores, prompt, steps, seed, sampler, k, temperature, top_p)
                for sampler in (py, py, c)]
    record = {"schema_version": 1, "scope": "sampler parity; Python model forward",
              "seed": seed, "prompt": list(prompt), "steps": steps,
              "sampler": {"k": k, "temperature": temperature, "top_p": top_p},
              "python_first": runs[0], "python_repeat": runs[1], "c": runs[2],
              "equal": runs[0] == runs[1] == runs[2]}
    Path(output).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("output")
    p.add_argument("--prompt", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--temperature", type=int, default=1)
    p.add_argument("--top-p", type=int, default=65536)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    model, args, sim, params = loss.load_run(a.run, True)
    if any(i < 0 or i >= model.emb.num_embeddings for i in a.prompt):
        p.error("prompt IDs outside model vocabulary")

    @torch.inference_mode()
    def scores(ids):
        z, _ = sim.forward(params, torch.tensor([ids[-args["seq"]:]]))
        return z[0, -1].tolist()

    result = check(scores, a.prompt, a.steps, a.seed, a.output,
                   k=a.k, temperature=a.temperature, top_p=a.top_p)
    if not result["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
