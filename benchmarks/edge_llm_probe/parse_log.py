#!/usr/bin/env python3
"""parse_log.py -- turn a serial log of the edge-LLM probe firmware into a JSON result and a markdown report.

    python parse_log.py probe_s3.log                        # parse a saved log
    python parse_log.py --port /dev/ttyUSB0 --duration 600  # capture from the board (needs pyserial), then parse
    python parse_log.py probe_s3.log --watts idle=1.10 lut4-gemv=1.62 lut4-2core=1.95

Writes results/<chip>_<timestamp>.json next to this script (or --out-dir) and prints markdown tables:
the MEASUREMENTS (from the log) and, clearly separated, a PROJECTION (arithmetic on those measurements plus stated
assumptions; it is not a measurement). Lines starting with '#' and any other text in the log are ignored.
"""
import argparse
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- projection assumptions (all stated in the report) -------------------------------------------------------------
SRAM_USABLE_KB = 300          # internal SRAM left for weights after the stack, buffers and the Arduino runtime
PSRAM_RESERVE_MB = 1.0        # PSRAM kept for the KV cache, activations and (optionally) camera frame buffers
FLASH_RESERVED_MB = 3.0       # bootloader, partition table, nvs and the application image
# (name, params, d_model, layers): decoder-only, 4x MLP (12 d^2 per layer), ~4k vocab, tied embeddings.
# Every parameter is treated as a 1-bit weight read once per token (the tied LM head included).
MODELS = [("15M", 15e6, 384, 8), ("30M", 30e6, 512, 9), ("110M", 110e6, 768, 15), ("200M", 200e6, 1024, 16)]
# kernels that can run a real model (xnor needs 1-bit activations: a lower bound, not a candidate; ref is a reference)
CANDIDATE_KERNELS = ("lut4", "lut4_2core_lookup", "bitserial4", "mask")
TIERS = ("sram", "psram", "flash", "sd")
# phase whose meter reading is used for the energy of the compute-bound kernel
PHASE_FOR_KERNEL = {"lut4": "lut4-gemv", "lut4_2core_lookup": "lut4-2core", "bitserial4": "lut4-gemv", "mask": "mask-gemv"}


# ---- parsing ---------------------------------------------------------------------------------------------------------
_LINE = re.compile(r"(INFO|BW|CAM|KERN|PHASE),([^\r\n]*)")


def _num(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_log(text):
    """Parse the probe log. If it holds several boots, the last one is used. Returns a dict (JSON-serialisable)."""
    boots, current = [], []
    for raw in text.splitlines():
        m = _LINE.search(raw)
        if not m or raw.lstrip().startswith("#"):
            continue
        kind, rest = m.group(1), m.group(2).strip().split(",")
        if kind == "INFO" and rest[0] == "probe_version" and current:
            boots.append(current)
            current = []
        current.append((kind, rest))
    if current:
        boots.append(current)
    lines = boots[-1] if boots else []

    data = {"info": {}, "bw": [], "kern": [], "cam": None, "phases": [], "boots_in_log": len(boots)}
    open_phases = {}
    for kind, f in lines:
        if kind == "INFO" and len(f) >= 2:
            data["info"][f[0]] = ",".join(f[1:])
        elif kind == "BW" and len(f) == 5 and _num(f[3]) is not None and _num(f[4]) is not None:
            data["bw"].append({"tier": f[0], "test": f[1], "bytes": int(float(f[2])),
                               "mbps_best": float(f[3]), "mbps_median": float(f[4])})
        elif kind == "KERN" and len(f) == 7 and _num(f[4]) is not None and _num(f[5]) is not None:
            data["kern"].append({"kernel": f[0], "mem": f[1], "m": int(f[2]), "n": int(f[3]),
                                 "cycles_per_weight": float(f[4]), "us_per_call": float(f[5]),
                                 "ok": f[6] == "OK"})
        elif kind == "CAM" and len(f) == 4:
            data["cam"] = {"width": int(f[0]), "height": int(f[1]), "jpeg_bytes": int(f[2]), "fps": float(f[3])}
        elif kind == "PHASE" and len(f) >= 3:
            number, name, what = int(f[0]), f[1], f[2]
            if what == "start":
                open_phases[number] = name
            elif what == "end" and len(f) == 4 and number in open_phases:
                data["phases"].append({"number": number, "name": name, "iterations": int(f[3])})
                del open_phases[number]
    # the schedule loops: keep the last completed round of every phase
    last = {}
    for p in data["phases"]:
        last[p["name"]] = p
    data["phases"] = sorted(last.values(), key=lambda p: p["number"])
    return data


# ---- projection ------------------------------------------------------------------------------------------------------
def tier_bandwidth(data):
    """Best sequential MB/s per tier from the large-buffer tests (the 16 KB cached tests are not a tier's bandwidth)."""
    pick = {"sram": [("sram_128KB", "read32_seq")],
            "psram": [("psram_4MB", "read32_seq"), ("psram_1MB", "read32_seq")],
            "flash": [("flash_mmap_1MB", "read32_seq")],
            "sd": [("sd_4bit", "file_read_buf32k"), ("sd_1bit", "file_read_buf32k")]}
    out = {}
    for tier, candidates in pick.items():
        vals = [b["mbps_best"] for tier_name, test in candidates for b in data["bw"]
                if b["tier"] == tier_name and b["test"] == test]
        if vals:
            out[tier] = max(vals)
    return out


def tier_capacity_bytes(info):
    """Bytes of weights each tier can hold under the stated assumptions."""
    cap = {"sram": SRAM_USABLE_KB * 1024}
    psram = _num(info.get("psram_size", "0")) or 0
    if psram > 0:
        cap["psram"] = max(0, psram - PSRAM_RESERVE_MB * 1024 * 1024)
    flash = _num(info.get("flash_size", "0")) or 0
    if flash > 0:
        cap["flash"] = max(0, flash - FLASH_RESERVED_MB * 1024 * 1024)
    return cap


def best_kernel(data, tier):
    """(kernel, cycles per weight) of the fastest verified candidate kernel at the largest shape measured where the
    weights sit in `tier` (the SD tier streams through PSRAM, so its figure is the PSRAM one)."""
    mem = {"sram": "sram", "psram": "psram", "flash": "flash", "sd": "psram"}[tier]
    rows = [r for r in data["kern"] if r["mem"] == mem and r["kernel"] in CANDIDATE_KERNELS and r["ok"]]
    if not rows:
        return None
    largest = max(r["m"] * r["n"] for r in rows)
    rows = [r for r in rows if r["m"] * r["n"] == largest]
    row = min(rows, key=lambda r: r["cycles_per_weight"])
    return row["kernel"], row["cycles_per_weight"], (row["m"], row["n"])


def project(data, models=MODELS):
    """PROJECTION rows: for every model and every tier it fits (SD only when nothing else fits).
    t_mem = weight bytes / measured bandwidth; t_cpu = params * cycles per weight / f_cpu.
    optimistic = max(t_mem, t_cpu) (memory and compute fully overlapped), pessimistic = t_mem + t_cpu."""
    bw = tier_bandwidth(data)
    cap = tier_capacity_bytes(data["info"])
    f_hz = (_num(data["info"].get("cpu_mhz", "0")) or 0) * 1e6
    rows = []
    for name, params, d_model, layers in models:
        weight_bytes = params / 8
        fitting = [t for t in ("sram", "psram", "flash") if t in cap and weight_bytes <= cap[t]]
        tiers = fitting or ["sd"]
        for tier in tiers:
            row = {"model": name, "params": params, "d_model": d_model, "layers": layers,
                   "weight_bytes": weight_bytes, "tier": tier, "bandwidth_mbps": bw.get(tier)}
            kern = best_kernel(data, tier)
            if tier not in bw or kern is None or f_hz <= 0:
                row.update(kernel=None, cycles_per_weight=None, s_mem=None, s_cpu=None,
                           s_per_token_optimistic=None, s_per_token_pessimistic=None, tok_s_optimistic=None,
                           tok_s_pessimistic=None)
            else:
                s_mem = weight_bytes / (bw[tier] * 1e6)
                s_cpu = params * kern[1] / f_hz
                opt, pess = max(s_mem, s_cpu), s_mem + s_cpu
                row.update(kernel=kern[0], cycles_per_weight=kern[1], s_mem=s_mem, s_cpu=s_cpu,
                           s_per_token_optimistic=opt, s_per_token_pessimistic=pess,
                           tok_s_optimistic=1 / opt, tok_s_pessimistic=1 / pess)
            rows.append(row)
    return rows


def parse_watts(items):
    """['idle=1.1', 'lut4-gemv=1.6'] -> {'idle': 1.1, 'lut4-gemv': 1.6}"""
    out = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not name or _num(value) is None:
            raise ValueError(f"--watts expects phase=value, got '{item}'")
        out[name] = float(value)
    return out


def add_energy(rows, watts):
    """Energy per token = P * t (P: the meter reading of the phase running that kernel, t: the projected s/token).
    Total is the board reading; above-idle subtracts the idle reading. Needs the phase's watts; else None."""
    for row in rows:
        p = watts.get(PHASE_FOR_KERNEL.get(row.get("kernel") or "", ""))
        idle = watts.get("idle")
        for key, label in (("s_per_token_optimistic", "optimistic"), ("s_per_token_pessimistic", "pessimistic")):
            t = row.get(key)
            row[f"mj_per_token_{label}"] = None if (p is None or t is None) else p * t * 1000
            row[f"mj_per_token_above_idle_{label}"] = None if (p is None or t is None or idle is None) \
                else (p - idle) * t * 1000
    return rows


# ---- markdown --------------------------------------------------------------------------------------------------------
def _fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_measurements(data):
    info = data["info"]
    parts = ["## Measurements (from the log)", ""]
    keys = ["chip", "chip_revision", "cores", "cpu_mhz", "heap_internal_free", "psram_size", "flash_size", "sd",
            "camera", "board", "sdk"]
    parts.append(_table(["key", "value"], [[k, info[k]] for k in keys if k in info]))
    if data["cam"]:
        c = data["cam"]
        parts += ["", f"Camera smoke test: {c['width']}x{c['height']} JPEG, {c['jpeg_bytes']} B/frame, {c['fps']:.1f} fps"]
    parts += ["", "### Memory read bandwidth (MB = 10^6 bytes)", ""]
    parts.append(_table(["tier", "test", "bytes", "best MB/s", "median MB/s"],
                        [[b["tier"], b["test"], str(b["bytes"]), f"{b['mbps_best']:.2f}", f"{b['mbps_median']:.2f}"]
                         for b in data["bw"]]))
    parts += ["", "### 1-bit kernels (cycles per weight; psram and flash are cache-cold)", ""]
    parts.append(_table(["kernel", "mem", "M", "N", "cycles/weight", "us/call", "check"],
                        [[k["kernel"], k["mem"], str(k["m"]), str(k["n"]), f"{k['cycles_per_weight']:.3f}",
                          f"{k['us_per_call']:.1f}", "OK" if k["ok"] else "FAIL"] for k in data["kern"]]))
    if data["phases"]:
        secs = _num(info.get("phase_seconds", "")) or None
        parts += ["", "### Energy schedule phases (iterations completed; watts come from the meter, see --watts)", ""]
        parts.append(_table(["#", "phase", "iterations", "iterations/s"],
                            [[str(p["number"]), p["name"], str(p["iterations"]),
                              _fmt(p["iterations"] / secs if secs else None, ".1f")] for p in data["phases"]]))
    return "\n".join(parts)


def render_projection(rows, watts):
    parts = ["## PROJECTION (not a measurement)", "",
             "Arithmetic on the measured numbers above plus these assumptions; nothing here was run on a chip:",
             f"- weights are 1 bit each: bytes = params / 8; every parameter is read once per token (tied LM head included);",
             f"- tiers: SRAM {SRAM_USABLE_KB} KB usable; PSRAM minus {PSRAM_RESERVE_MB:g} MB reserve; flash minus "
             f"{FLASH_RESERVED_MB:g} MB (bootloader, nvs, app); SD only when no other tier holds the model;",
             "- t_mem = weight bytes / measured sequential bandwidth of the tier; t_cpu = params x best verified kernel "
             "cycles/weight (largest shape, weights in that tier) / CPU clock;",
             "- optimistic s/token = max(t_mem, t_cpu) (full overlap); pessimistic = t_mem + t_cpu (none). The kernel "
             "figures of PSRAM and flash already include their cache-miss stalls, so t_cpu there partly double-counts t_mem;",
             "- not included: attention, KV cache traffic, activation quantisation, embeddings lookup, sampling;",
             "- model shapes (d_model x layers, ~4k vocab) are illustrative: "
             + ", ".join(f"{n}: {d}x{l}" for n, _, d, l in MODELS) + ".", ""]
    header = ["model", "weights", "tier", "BW MB/s", "kernel", "cyc/w", "t_mem s", "t_cpu s", "s/tok opt", "s/tok pess",
              "tok/s opt", "tok/s pess"]
    body = []
    for r in rows:
        body.append([r["model"], f"{r['weight_bytes'] / 1e6:.2f} MB", r["tier"], _fmt(r["bandwidth_mbps"], ".1f"),
                     r["kernel"] or "n/a", _fmt(r["cycles_per_weight"], ".2f"), _fmt(r["s_mem"], ".3f"),
                     _fmt(r["s_cpu"], ".3f"), _fmt(r["s_per_token_optimistic"], ".3f"),
                     _fmt(r["s_per_token_pessimistic"], ".3f"), _fmt(r["tok_s_optimistic"], ".2f"),
                     _fmt(r["tok_s_pessimistic"], ".2f")])
    parts.append(_table(header, body))
    if watts:
        parts += ["", "### PROJECTED energy per token = meter watts x projected s/token (board-level, not chip-level)", "",
                  "Watts entered: " + ", ".join(f"{k}={v:g} W" for k, v in watts.items()), ""]
        eb = []
        for r in rows:
            eb.append([r["model"], r["tier"], r["kernel"] or "n/a", _fmt(r["mj_per_token_optimistic"], ".0f"),
                       _fmt(r["mj_per_token_pessimistic"], ".0f"), _fmt(r["mj_per_token_above_idle_optimistic"], ".0f"),
                       _fmt(r["mj_per_token_above_idle_pessimistic"], ".0f")])
        parts.append(_table(["model", "tier", "kernel", "mJ/tok total opt", "mJ/tok total pess",
                             "mJ/tok above idle opt", "mJ/tok above idle pess"], eb))
    return "\n".join(parts)


# ---- capture ---------------------------------------------------------------------------------------------------------
def capture(port, baud, duration, log_path):
    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("--port needs pyserial: pip install pyserial (or save the log with pio device monitor and pass the file)")
    deadline = datetime.datetime.now() + datetime.timedelta(seconds=duration)
    print(f"# capturing {duration} s from {port} into {log_path} (reset the board to see the boot lines)", file=sys.stderr)
    with serial.Serial(port, baud, timeout=1) as ser, open(log_path, "w", encoding="utf-8") as out:
        while datetime.datetime.now() < deadline:
            line = ser.readline().decode("utf-8", errors="replace")
            if line:
                out.write(line)
                out.flush()
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="saved serial log")
    ap.add_argument("--port", help="serial port to capture from (needs pyserial)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=int, default=900, help="seconds to capture with --port (default 900)")
    ap.add_argument("--watts", nargs="*", metavar="PHASE=W", help="meter readings, e.g. idle=1.10 lut4-gemv=1.62")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args(argv)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.port:
        text = capture(args.port, args.baud, args.duration, os.path.join(args.out_dir, f"capture_{stamp}.log"))
    elif args.log:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        ap.error("give a log file or --port")

    data = parse_log(text)
    if not data["info"] and not data["bw"] and not data["kern"]:
        sys.exit("no INFO/BW/KERN lines found in the log")
    watts = parse_watts(args.watts)
    rows = add_energy(project(data), watts)
    chip = re.sub(r"[^A-Za-z0-9]+", "", data["info"].get("chip", "unknown")).lower() or "unknown"
    result = {"schema": 1, "generated": stamp, "measurements": data, "meter_watts": watts,
              "projection": {"label": "PROJECTION, not a measurement",
                             "assumptions": {"sram_usable_kb": SRAM_USABLE_KB, "psram_reserve_mb": PSRAM_RESERVE_MB,
                                             "flash_reserved_mb": FLASH_RESERVED_MB,
                                             "models": [dict(name=n, params=p, d_model=d, layers=l)
                                                        for n, p, d, l in MODELS]},
                             "rows": rows}}
    out_path = os.path.join(args.out_dir, f"{chip}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(render_measurements(data))
    print()
    print(render_projection(rows, watts))
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
