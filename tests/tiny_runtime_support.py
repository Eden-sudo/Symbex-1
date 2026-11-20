"""Small host-only runtime test helpers; no model downloads or training."""
import importlib.util
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LM = ROOT / "benchmarks/experiments/tiny_llm"


def module(name):
    spec = importlib.util.spec_from_file_location(name, LM / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def compile_c(tmp_path, source, sources=()):
    src = tmp_path / "test.c"
    exe = tmp_path / "test"
    src.write_text(source)
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-I", str(LM / "c"), str(src),
                    *(str(LM / "c" / s) for s in sources), "-o", str(exe)],
                   check=True, capture_output=True, text=True, timeout=15)
    return exe


def run(exe, data="", args=()):
    return subprocess.run([str(exe), *map(str, args)], input=data, text=True,
                          capture_output=True, check=True, timeout=10).stdout
