"""Import neighboring experiment files without modifying sys.path."""
import importlib.util
from pathlib import Path


def load(name):
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
