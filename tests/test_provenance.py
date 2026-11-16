"""Provenance of a measurement: compiler_core/chip/provenance.py writes the day, the repository commit and the last
lib_symbex commit into a result file; tools/dev/results_age.py reads them (or the file's commit date) back."""
import datetime
import os
import subprocess
import sys

from compiler_core.chip.provenance import provenance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def test_provenance_fields():
    p = provenance(tflm="esp-tflite-micro 1.4.1")
    assert p["date"] == datetime.date.today().isoformat()
    assert p["tflm"] == "esp-tflite-micro 1.4.1"
    assert p["commit"] == _git("rev-parse", "--short=12", "HEAD")
    assert p["lib_commit"] == _git("log", "-1", "--format=%h", "--abbrev=12", "--", "Symbex1_PIO/lib/lib_symbex")
    assert isinstance(p["lib_dirty"], bool)


def test_results_age_runs():
    out = subprocess.run([sys.executable, os.path.join(REPO, "tools", "dev", "results_age.py"), "--match", "resnet8"],
                         capture_output=True, text=True, check=True).stdout
    assert "cnn_cifar_resnet8" in out


def test_results_age_classifies_baselines():
    import importlib.util
    spec = importlib.util.spec_from_file_location("results_age", os.path.join(REPO, "tools", "dev", "results_age.py"))
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    for f in ("mlp/mlp_fashion/results/earlier/int4_pc.json", "x/results/tflite/tflite_s3.json",
              "x/results/lowbit/a.json", "x/results/ucbnn/ucbnn_gvsoc.json"):
        assert ra.NOT_SYMBEX.search(f), f
    for f in ("x/results/symbex/symbex_seeds_s3_swar_w1.json", "x/results/earlier/symbex_s42_chip_uno_flash_swar.json"):
        assert not ra.NOT_SYMBEX.search(f), f
