"""Test energy integration and input validation with small synthetic logs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools/energy_per_token.py"


@pytest.fixture(scope="module")
def energy():
    spec = importlib.util.spec_from_file_location("energy_per_token", TOOL)
    assert spec is not None and spec.loader is not None, f"Cannot load {TOOL}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def meter(tmp_path, text):
    path = tmp_path / "meter.csv"
    path.write_text(text, encoding="utf-8")
    return path


def phases(tmp_path, data, suffix=".json"):
    path = tmp_path / ("phases" + suffix)
    path.write_text(json.dumps(data) if suffix.lower() == ".json" else data, encoding="utf-8")
    return path


def phase(name="active", start=0, end=2, tokens=4):
    return dict(name=name, start_s=start, end_s=end, tokens=tokens)


def cli(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True, timeout=5)


def test_constant_and_step_and_irregular(energy, tmp_path):
    p = phases(tmp_path, [phase()])
    m = meter(tmp_path, "time_s,voltage_v,current_a\n0,5,.2\n1,5,.2\n2,5,.2\n")
    result = energy.analyze(energy.read_meter(m, energy.DEFAULT_COLUMNS), energy.read_phases(p))[0]
    assert result == dict(name="active", duration_s=2, mean_power_mw=1000,
                          energy_mj=2000, net_energy_mj=2000, tokens=4, mj_per_token=500)
    for text, expected, mean in [
        ("0,1,1\n1,1,1\n2,3,1\n3,3,1\n", 6000, 2000),
        ("0,1,1\n1,3,1\n3,3,1\n", 8000, 8000 / 3),
    ]:
        m = meter(tmp_path, "time_s,voltage_v,current_a\n" + text)
        result = energy.integrate_phase(energy.read_meter(m, energy.DEFAULT_COLUMNS), phase(end=3))
        assert result["energy_mj"] == pytest.approx(expected)
        assert result["mean_power_mw"] == pytest.approx(mean)


def test_interpolation_of_power(energy, tmp_path):
    m = meter(tmp_path, "time_s,voltage_v,current_a\n0,0,0\n1,1,1\n2,2,1\n3,3,1\n")
    result = energy.integrate_phase(energy.read_meter(m, energy.DEFAULT_COLUMNS), phase(start=.5, end=2.5))
    assert result["energy_mj"] == pytest.approx(3000)


def test_idle_and_negative(energy, tmp_path):
    m = meter(tmp_path, "time_s,voltage_v,current_a\n0,1,1\n1,1,1\n2,3,1\n3,3,1\n4,3,1\n5,0,0\n6,0,0\n")
    data = [phase("active", 2, 4), phase("idle", 0, 1, 0), phase("low", 5, 6, 2)]
    results = energy.analyze(energy.read_meter(m, energy.DEFAULT_COLUMNS), data, "idle")
    assert results[0]["energy_mj"] == pytest.approx(6000)
    assert results[0]["net_energy_mj"] == pytest.approx(4000)
    assert results[0]["mj_per_token"] == pytest.approx(1000)
    assert results[1]["net_energy_mj"] == pytest.approx(0)
    assert results[1]["mj_per_token"] is None
    assert results[2]["net_energy_mj"] == pytest.approx(-1000)
    assert results[2]["mj_per_token"] == pytest.approx(-500)


def test_formats_mapping_overlap_and_cli(energy, tmp_path):
    m = meter(tmp_path, "Time,voltage_v,current_a,extra\n0,5,.2,x\n1,5,.2,x\n2,5,.2,x\n")
    data = [dict(phase(), extra=True), phase("second", 1, 2, 0)]
    j = phases(tmp_path, data)
    c = phases(tmp_path, "name,start_s,end_s,tokens,extra\nactive,0,2,4,x\nsecond,1,2,0,x\n", ".CSV")
    samples = energy.read_meter(m, energy.parse_columns(" time_s = Time "))
    assert energy.analyze(samples, energy.read_phases(j)) == energy.analyze(samples, energy.read_phases(c))
    result = cli(m, j, "--columns", "time_s=Time")
    assert result.returncode == 0 and result.stderr == ""
    output = json.loads(result.stdout)
    assert [row["name"] for row in output] == ["active", "second"]
    assert set(output[0]) == {"name", "duration_s", "mean_power_mw", "energy_mj", "net_energy_mj", "tokens", "mj_per_token"}


@pytest.mark.parametrize("text", ["", "foo=x", "time_s=", "time_s=x,time_s=y", "time_s=voltage_v", "time_s=x,"])
def test_bad_mappings(tmp_path, text):
    result = cli("meter.csv", "phases.json", "--columns", text)
    assert result.returncode == 2 and result.stdout == ""


@pytest.mark.parametrize("body", [
    "", "0,1,1\n", "0,1,1\n0,1,1\n", "1,1,1\n0,1,1\n",
    "0,-1,1\n1,1,1\n", "0,1,-1\n1,1,1\n", "0,nan,1\n1,1,1\n",
    "0,1,inf\n1,1,1\n", "0,1e308,1e308\n1,1,1\n", "0,,1\n1,1,1\n",
    "0,abc,1\n1,1,1\n", "0,1\n1,1,1\n", "0,1,1,x\n1,1,1\n",
])
def test_bad_meter(energy, tmp_path, body):
    m = meter(tmp_path, "time_s,voltage_v,current_a\n" + body)
    with pytest.raises(energy.EnergyInputError, match="meter.csv"):
        energy.read_meter(m, energy.DEFAULT_COLUMNS)


@pytest.mark.parametrize("header", ["time_s,voltage_v", "time_s,voltage_v,current_a,current_a"])
def test_bad_headers(energy, tmp_path, header):
    with pytest.raises(energy.EnergyInputError, match="header"):
        energy.read_meter(meter(tmp_path, header + "\n"), energy.DEFAULT_COLUMNS)


@pytest.mark.parametrize("data", [[], {}, [1], [phase(name=" ")], [phase(), phase()],
    [phase(start=2, end=2)], [phase(start=3, end=2)], [phase(start="0")],
    [phase(start=float("nan"))], [phase(tokens=-1)], [phase(tokens=True)],
    [phase(tokens=1.5)], [phase(tokens="1")], [{"name": "x"}]])
def test_bad_phases(energy, tmp_path, data):
    with pytest.raises(energy.EnergyInputError, match="phases"):
        energy.read_phases(phases(tmp_path, data))


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 4), (.1, .9), (.1, 1.1)])
def test_bad_intervals(energy, tmp_path, start, end):
    m = meter(tmp_path, "time_s,voltage_v,current_a\n0,1,1\n1,1,1\n2,1,1\n3,1,1\n")
    with pytest.raises(energy.EnergyInputError, match="active"):
        energy.integrate_phase(energy.read_meter(m, energy.DEFAULT_COLUMNS), phase(start=start, end=end))


def test_errors_and_help(energy, tmp_path):
    m = meter(tmp_path, "time_s,voltage_v,current_a\n0,1,1\n1,1,1\n2,1,1\n")
    p = phases(tmp_path, [phase(), phase("bad", 1, 3)])
    result = cli(m, p)
    assert result.returncode == 1 and not result.stdout and "bad" in result.stderr and "Traceback" not in result.stderr
    p = phases(tmp_path, [phase()])
    assert cli(m, p, "--idle", "missing").returncode == 1
    assert cli(m, p, "--columns", "time_s=absent").returncode == 1
    assert cli().returncode == 2
    help_result = cli("--help")
    assert help_result.returncode == 0 and "two original meter samples" in help_result.stdout
    p.write_text("[", encoding="utf-8")
    with pytest.raises(energy.EnergyInputError, match="JSON"):
        energy.read_phases(p)
    with pytest.raises(energy.EnergyInputError, match="extension"):
        energy.read_phases(tmp_path / "bad.txt")
    with pytest.raises(energy.EnergyInputError, match="missing"):
        energy.read_phases(tmp_path / "missing.json")
