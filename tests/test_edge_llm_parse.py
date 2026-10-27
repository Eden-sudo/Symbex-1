"""benchmarks/edge_llm_probe/parse_log.py: parsing of the probe's serial lines and the projection arithmetic.

The log below is SYNTHETIC: invented numbers chosen so the expected results can be computed by hand. It says nothing
about any real chip."""
import importlib.util
import json
import os

import pytest

PARSER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "benchmarks", "edge_llm_probe", "parse_log.py")
_spec = importlib.util.spec_from_file_location("edge_llm_parse_log", PARSER)
parse_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parse_log)

SYNTHETIC_LOG = """\
ets Jul 29 2019 12:21:46 boot noise that is not a probe line
# SYMBEX edge-LLM probe: comment lines are ignored
INFO,probe_version,1
INFO,chip,ESP32-S3
INFO,cpu_mhz,100
INFO,psram_size,8388608
INFO,flash_size,16777216
INFO,phase_seconds,20
INFO,sd,SDHC_7580MB
INFO,camera,skipped
BW,sram_128KB,read32_seq,4194304,200.00,190.00
BW,sram_16KB,read32_seq,4194304,260.00,250.00
BW,psram_4MB,read32_seq,4194304,40.00,38.00
BW,psram_16KB,read32_seq,4194304,150.00,149.00
BW,flash_mmap_1MB,read32_seq,4194304,20.00,19.50
BW,sd_1bit,file_read_buf32k,8388608,2.00,1.90
KERN,lut4,sram,1024,1024,4.000,17476.3,OK
KERN,lut4,psram,1024,1024,6.000,26214.4,OK
KERN,lut4,psram,2048,2048,5.000,87381.3,OK
KERN,lut4_2core_lookup,psram,2048,2048,3.000,52428.8,OK
KERN,bitserial4,psram,2048,2048,2.000,34952.5,FAIL
KERN,mask,psram,2048,2048,9.000,157286.4,OK
KERN,xnor,psram,2048,2048,1.000,17476.3,OK
KERN,lut4,flash,1024,1024,8.000,34952.5,OK
PHASE,1,idle,start
PHASE,1,idle,end,1990
PHASE,6,lut4-gemv,start
PHASE,6,lut4-gemv,end,400
PHASE,6,lut4-gemv,start
PHASE,6,lut4-gemv,end,410
this line is not a probe line: BW,broken,line
"""


@pytest.fixture
def data():
    return parse_log.parse_log(SYNTHETIC_LOG)


def test_parses_info_bandwidth_kernels_and_phases(data):
    assert data["info"]["chip"] == "ESP32-S3"
    assert data["info"]["cpu_mhz"] == "100"
    assert len(data["bw"]) == 6
    psram = [b for b in data["bw"] if b["tier"] == "psram_4MB"][0]
    assert (psram["bytes"], psram["mbps_best"], psram["mbps_median"]) == (4194304, 40.0, 38.0)
    assert len(data["kern"]) == 8
    bad = [k for k in data["kern"] if k["kernel"] == "bitserial4"][0]
    assert bad["ok"] is False and bad["m"] == 2048 and bad["cycles_per_weight"] == 2.0
    # the schedule loops: the last completed round of a phase wins
    assert {p["name"]: p["iterations"] for p in data["phases"]} == {"idle": 1990, "lut4-gemv": 410}


def test_ignores_comments_and_malformed_lines(data):
    assert not any(b["tier"] == "broken" for b in data["bw"])
    assert parse_log.parse_log("# INFO,chip,fake\nBW,x,y\nKERN,a,b\n")["info"] == {}


def test_uses_only_the_last_boot_of_a_log():
    two_boots = "INFO,probe_version,1\nINFO,chip,OLD\nINFO,probe_version,1\nINFO,chip,NEW\n"
    parsed = parse_log.parse_log(two_boots)
    assert parsed["info"]["chip"] == "NEW" and parsed["boots_in_log"] == 2


def test_tier_bandwidth_ignores_the_cached_16kb_tests(data):
    assert parse_log.tier_bandwidth(data) == {"sram": 200.0, "psram": 40.0, "flash": 20.0, "sd": 2.0}


def test_best_kernel_is_the_fastest_verified_candidate_at_the_largest_shape(data):
    # psram 2048x2048: bitserial4 (2.0) FAILED, xnor is not a candidate, lut4_2core_lookup (3.0) beats lut4 (5.0), mask (9.0)
    assert parse_log.best_kernel(data, "psram") == ("lut4_2core_lookup", 3.0, (2048, 2048))
    assert parse_log.best_kernel(data, "sram") == ("lut4", 4.0, (1024, 1024))
    assert parse_log.best_kernel(data, "sd") == parse_log.best_kernel(data, "psram")  # SD streams through PSRAM


def test_projection_math_for_a_model_in_psram(data):
    rows = parse_log.project(data, models=[("8M", 8e6, 256, 8)])
    by_tier = {r["tier"]: r for r in rows}
    # 8M params = 1 MB of weights: too big for 300 KB of SRAM, fits PSRAM and flash, so no SD row
    assert set(by_tier) == {"psram", "flash"}
    r = by_tier["psram"]
    assert r["weight_bytes"] == 1e6
    assert r["s_mem"] == pytest.approx(1e6 / 40e6)                    # 0.025 s
    assert r["s_cpu"] == pytest.approx(8e6 * 3.0 / 100e6)             # 0.24 s at 100 MHz, 3 cycles/weight
    assert r["s_per_token_optimistic"] == pytest.approx(0.24)         # max
    assert r["s_per_token_pessimistic"] == pytest.approx(0.265)       # sum
    assert r["tok_s_optimistic"] == pytest.approx(1 / 0.24)
    assert r["tok_s_pessimistic"] == pytest.approx(1 / 0.265)
    assert r["kernel"] == "lut4_2core_lookup"


def test_memory_bound_case_takes_the_memory_time(data):
    rows = parse_log.project(data, models=[("8M", 8e6, 256, 8)])
    flash = [r for r in rows if r["tier"] == "flash"][0]
    # flash kernels: only lut4 at 1024x1024 (8.0 cyc/w): t_cpu = 0.64 s, t_mem = 1e6 / 20e6 = 0.05 s
    assert flash["s_mem"] == pytest.approx(0.05) and flash["s_cpu"] == pytest.approx(0.64)
    assert flash["s_per_token_optimistic"] == pytest.approx(0.64)
    assert flash["s_per_token_pessimistic"] == pytest.approx(0.69)


def test_sd_row_only_when_the_model_fits_no_other_tier(data):
    rows = parse_log.project(data, models=[("200M", 200e6, 1024, 16)])
    # 25 MB of weights: PSRAM holds 8 MB - 1 MB, flash 16 MB - 3 MB = 13 MB -> still no fit -> SD
    assert [r["tier"] for r in rows] == ["sd"]
    assert rows[0]["s_mem"] == pytest.approx(25e6 / 2e6)              # 12.5 s at 2 MB/s


def test_missing_measurements_give_none_not_a_made_up_number():
    rows = parse_log.project(parse_log.parse_log("INFO,probe_version,1\nINFO,cpu_mhz,240\nINFO,psram_size,8388608\n"),
                             models=[("8M", 8e6, 256, 8)])
    assert rows and all(r["s_per_token_optimistic"] is None and r["kernel"] is None for r in rows)


def test_energy_per_token_is_power_times_projected_time(data):
    rows = parse_log.project(data, models=[("8M", 8e6, 256, 8)])
    watts = parse_log.parse_watts(["idle=1.0", "lut4-2core=1.5"])
    rows = parse_log.add_energy(rows, watts)
    r = [x for x in rows if x["tier"] == "psram"][0]
    assert r["mj_per_token_optimistic"] == pytest.approx(1.5 * 0.24 * 1000)
    assert r["mj_per_token_above_idle_pessimistic"] == pytest.approx(0.5 * 0.265 * 1000)
    # no reading for the lut4 phase -> the flash row (kernel lut4) has no energy figure
    flash = [x for x in rows if x["tier"] == "flash"][0]
    assert flash["mj_per_token_optimistic"] is None


def test_watts_argument_validation():
    assert parse_log.parse_watts(["idle=1.25"]) == {"idle": 1.25}
    with pytest.raises(ValueError):
        parse_log.parse_watts(["idle"])


def test_cli_writes_json_and_marks_the_projection(tmp_path, capsys):
    log = tmp_path / "synthetic.log"
    log.write_text(SYNTHETIC_LOG)
    parse_log.main([str(log), "--out-dir", str(tmp_path), "--watts", "idle=1.0", "lut4-gemv=1.4"])
    out = capsys.readouterr().out
    assert "## Measurements (from the log)" in out and "## PROJECTION (not a measurement)" in out
    files = [f for f in os.listdir(tmp_path) if f.startswith("esp32s3_") and f.endswith(".json")]
    assert len(files) == 1
    saved = json.loads((tmp_path / files[0]).read_text())
    assert saved["projection"]["label"] == "PROJECTION, not a measurement"
    assert saved["meter_watts"] == {"idle": 1.0, "lut4-gemv": 1.4}
