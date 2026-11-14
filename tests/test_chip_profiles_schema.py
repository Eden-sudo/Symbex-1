"""Every chip profile in tools/chip_profiles follows the schema the code that writes it produces.

Kernel profiles (<board>_<popcount>.json) are written by run_kernels (tools/compiler_core/chip/kernels.py); count
profiles (count_<platform>.json) by tools/dev/count_bench.py. The accepted `weights` values are read from the
--weights choices of tools/chip_test.py, never hardcoded here. A profile that does not follow its schema fails its own
parametrized test, named after the file.
"""
import ast
import copy
import glob
import json
import math
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(REPO, "tools", "chip_profiles")
CHIP_TEST = os.path.join(REPO, "tools", "chip_test.py")
PROFILES = sorted(glob.glob(os.path.join(PROFILE_DIR, "*.json")))
KERNEL_PROFILES = [p for p in PROFILES if not os.path.basename(p).startswith("count_")]
COUNT_PROFILES = [p for p in PROFILES if os.path.basename(p).startswith("count_")]

KERNEL_KEYS = ("version", "board", "popcount", "hal", "n_out", "widths", "weights", "cpu_mhz", "measurements")
MEASUREMENT_KEYS = ("kernel", "n_in", "ns", "checksum", "cycles_per_neuron", "cycles_per_weight")
COUNT_KEYS = ("platform", "name", "features", "unit", "positions", "outputs", "checksums_ok", "measurements",
              "fastest", "integer_checksums_ok", "integer")


def _is_int(x):
    return type(x) is int


def _is_pos_int(x):
    return _is_int(x) and x > 0


def _is_pos_num(x):
    return type(x) in (int, float) and x > 0 and math.isfinite(x)


def _is_text(x):
    return isinstance(x, str) and x.strip() != ""


def accepted_weights():
    with open(CHIP_TEST) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--weights"):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return set(ast.literal_eval(kw.value))
    pytest.fail("--weights choices not found in tools/chip_test.py")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"{os.path.basename(path)}: invalid JSON: {e}")


def _missing(entry, keys, prefix=""):
    return [f"{prefix}missing key: {k}" for k in keys if k not in entry]


def kernel_profile_problems(profile, weights):
    if not isinstance(profile, dict):
        return ["not a JSON object"]
    problems = _missing(profile, KERNEL_KEYS)
    if problems:
        return problems
    if not (_is_int(profile["version"]) and profile["version"] == 2):
        problems.append(f"version is {profile['version']!r}, expected 2")
    for key in ("board", "popcount", "hal"):
        if not _is_text(profile[key]):
            problems.append(f"{key} is not a non-empty string: {profile[key]!r}")
    n_out_ok = _is_pos_int(profile["n_out"])
    if not n_out_ok:
        problems.append(f"n_out is not a positive int: {profile['n_out']!r}")
    w = profile["widths"]
    if not (isinstance(w, list) and w and all(_is_pos_int(x) for x in w)
            and all(a < b for a, b in zip(w, w[1:]))):
        problems.append(f"widths not a strictly increasing list of positive ints: {w!r}")
    if profile["weights"] not in weights:
        problems.append(f"weights {profile['weights']!r} not in {sorted(weights)}")
    cpu_ok = _is_pos_num(profile["cpu_mhz"])
    if not cpu_ok:
        problems.append(f"cpu_mhz is not a positive number: {profile['cpu_mhz']!r}")
    rows = profile["measurements"]
    if not (isinstance(rows, list) and rows):
        problems.append("measurements is not a non-empty list")
        return problems
    for i, r in enumerate(rows):
        prefix = f"measurements[{i}]: "
        if not isinstance(r, dict):
            problems.append(prefix + "not a JSON object")
            continue
        missing = _missing(r, MEASUREMENT_KEYS, prefix)
        if missing:
            problems += missing
            continue
        checks = (("kernel", _is_text), ("n_in", _is_pos_int), ("ns", _is_pos_int), ("checksum", _is_int),
                  ("cycles_per_neuron", _is_pos_num), ("cycles_per_weight", _is_pos_num))
        bad = [k for k, ok in checks if not ok(r[k])]
        problems += [f"{prefix}invalid {k}: {r[k]!r}" for k in bad]
        if bad or not (n_out_ok and cpu_ok):
            continue
        cyc = r["ns"] * profile["cpu_mhz"] / 1000
        for key, expected in (("cycles_per_neuron", cyc / profile["n_out"]),
                              ("cycles_per_weight", cyc / (profile["n_out"] * r["n_in"]))):
            if not math.isclose(r[key], expected, rel_tol=0.01):
                problems.append(f"{prefix}kernel {r['kernel']} n_in {r['n_in']}: {key} expected {expected:.6g}, "
                                f"stored {r[key]!r}")
    return problems


def _row_problems(rows, name, checks):
    if not isinstance(rows, list):
        return [f"{name} is not a list"]
    problems = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            problems.append(f"{name}[{i}]: not a JSON object")
            continue
        problems += _missing(r, checks, f"{name}[{i}]: ")
        problems += [f"{name}[{i}]: invalid {k}: {r[k]!r}" for k, ok in checks.items() if k in r and not ok(r[k])]
    return problems


def count_profile_problems(profile):
    if not isinstance(profile, dict):
        return ["not a JSON object"]
    problems = _missing(profile, COUNT_KEYS)
    if problems:
        return problems
    checks = (("platform", _is_text), ("name", _is_text), ("features", lambda x: isinstance(x, str)),
              ("unit", lambda x: x in ("cycles", "ns")), ("positions", _is_pos_int), ("outputs", _is_pos_int),
              ("checksums_ok", lambda x: type(x) is bool), ("integer_checksums_ok", lambda x: type(x) is bool))
    problems += [f"invalid {k}: {profile[k]!r}" for k, ok in checks if not ok(profile[k])]
    if not (isinstance(profile["measurements"], list) and profile["measurements"]):
        problems.append("measurements is not a non-empty list")
    else:
        problems += _row_problems(profile["measurements"], "measurements", {
            "method": _is_text, "form": lambda x: x in ("xnor", "and"), "words": _is_pos_int,
            "per_output": _is_pos_num, "checksum": _is_int})
    problems += _row_problems(profile["integer"], "integer", {
        "method": _is_text, "elements": _is_pos_int, "per_mac": _is_pos_num, "checksum": _is_int})
    fastest = profile["fastest"]
    if not isinstance(fastest, dict):
        problems.append("fastest is not a JSON object")
        return problems
    for key, entry in fastest.items():
        if not (isinstance(key, str) and key.isdigit()):
            problems.append(f"fastest: key is not a string of digits: {key!r}")
        if not isinstance(entry, dict):
            problems.append(f"fastest[{key}]: not a JSON object")
            continue
        problems += _missing(entry, ("method", "form", "per_output"), f"fastest[{key}]: ")
        if "per_output" in entry and not _is_pos_num(entry["per_output"]):
            problems.append(f"fastest[{key}]: invalid per_output: {entry['per_output']!r}")
    return problems


def test_profiles_found():
    assert KERNEL_PROFILES, "no kernel profiles in tools/chip_profiles"
    assert COUNT_PROFILES, "no count profiles in tools/chip_profiles"


def test_weights_choices_read_from_chip_test():
    weights = accepted_weights()
    assert weights and all(isinstance(w, str) for w in weights)


@pytest.mark.parametrize("path", KERNEL_PROFILES, ids=os.path.basename)
def test_kernel_profile_follows_schema(path):
    problems = kernel_profile_problems(load(path), accepted_weights())
    assert not problems, f"{os.path.basename(path)}: " + "; ".join(problems)


@pytest.mark.parametrize("path", COUNT_PROFILES, ids=os.path.basename)
def test_count_profile_follows_schema(path):
    problems = count_profile_problems(load(path))
    assert not problems, f"{os.path.basename(path)}: " + "; ".join(problems)


def modified_copy(src, tmp_path, change=None):
    profile = copy.deepcopy(load(src))
    if change:
        change(profile)
    dst = tmp_path / os.path.basename(src)
    with open(dst, "w") as f:
        json.dump(profile, f)
    return load(str(dst))


def kernel_problems_after(tmp_path, change=None):
    return kernel_profile_problems(modified_copy(KERNEL_PROFILES[0], tmp_path, change), accepted_weights())


def test_unmodified_copy_passes(tmp_path):
    assert kernel_problems_after(tmp_path) == []
    assert count_profile_problems(modified_copy(COUNT_PROFILES[0], tmp_path)) == []


def test_kernel_profile_missing_key_fails(tmp_path):
    problems = kernel_problems_after(tmp_path, lambda p: p.pop("hal"))
    assert "missing key: hal" in problems


def test_kernel_profile_decreasing_widths_fails(tmp_path):
    problems = kernel_problems_after(tmp_path, lambda p: p.update(widths=sorted(p["widths"], reverse=True)))
    assert any("widths" in m for m in problems)


def test_kernel_profile_repeated_width_fails(tmp_path):
    problems = kernel_problems_after(tmp_path, lambda p: p.update(widths=[32, 32, 64]))
    assert any("widths" in m for m in problems)


def test_kernel_profile_unknown_weights_fails(tmp_path):
    problems = kernel_problems_after(tmp_path, lambda p: p.update(weights="psram"))
    assert any("weights" in m for m in problems)


def test_kernel_profile_cycles_off_by_more_than_1pct_fails(tmp_path):
    def scale(factor):
        return lambda p: p["measurements"][0].update(
            cycles_per_neuron=p["measurements"][0]["cycles_per_neuron"] * factor)
    problems = kernel_problems_after(tmp_path, scale(1.02))
    assert any("measurements[0]" in m and "cycles_per_neuron" in m for m in problems)
    assert kernel_problems_after(tmp_path, scale(1.005)) == []


def test_kernel_profile_measurement_missing_key_fails(tmp_path):
    problems = kernel_problems_after(tmp_path, lambda p: p["measurements"][0].pop("ns"))
    assert any("measurements[0]" in m and "ns" in m for m in problems)


def test_kernel_profile_bool_n_out_fails(tmp_path):
    problems = kernel_problems_after(tmp_path, lambda p: p.update(n_out=True))
    assert any("n_out" in m for m in problems)


def test_count_profile_missing_key_fails(tmp_path):
    profile = modified_copy(COUNT_PROFILES[0], tmp_path, lambda p: p.pop("measurements"))
    assert "missing key: measurements" in count_profile_problems(profile)
