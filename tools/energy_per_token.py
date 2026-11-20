"""Calculate energy per token from completed USB power meter logs.

The meter CSV header defaults to time_s,voltage_v,current_a (seconds, volts,
amps). Use --columns time_s=Time,voltage_v=Voltage for partial overrides.
Phases are a JSON list or CSV table with name,start_s,end_s,tokens. Power is
linearly interpolated at boundaries and integrated with the trapezoid rule;
each closed phase interval must contain at least two original meter samples.
Output is a JSON array of per-phase durations (s), power (mW), energy (mJ),
net energy after optional idle subtraction, tokens, and mJ/token (null for
zero tokens). For example:

    python tools/energy_per_token.py meter.csv phases.json --idle idle
"""
import argparse
from bisect import bisect_left, bisect_right
import csv
import json
import math
from pathlib import Path
import re
import sys


class EnergyInputError(ValueError):
    """Invalid measurement input or calculation."""


DEFAULT_COLUMNS = {key: key for key in ("time_s", "voltage_v", "current_a")}
PHASE_FIELDS = ("name", "start_s", "end_s", "tokens")


def parse_columns(text):
    columns = DEFAULT_COLUMNS.copy()
    seen = set()
    for entry in text.split(","):
        parts = entry.split("=")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("columns must be canonical=source pairs")
        key, value = (part.strip() for part in parts)
        if key not in columns or not value or key in seen:
            raise argparse.ArgumentTypeError("invalid or repeated column mapping")
        seen.add(key)
        columns[key] = value
    if len(set(columns.values())) != len(columns):
        raise argparse.ArgumentTypeError("mapped source columns must be distinct")
    return columns


def _number(value, context, json_number=False):
    if json_number and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise EnergyInputError(f"{context}: expected a number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise EnergyInputError(f"{context}: expected a finite number") from error
    if not math.isfinite(result):
        raise EnergyInputError(f"{context}: expected a finite number")
    return result


def _csv_rows(path, required):
    try:
        with open(path, encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            headers = reader.fieldnames
            if not headers or len(headers) != len(set(headers)):
                raise EnergyInputError(f"{path}: missing or duplicate CSV headers")
            missing = set(required) - set(headers)
            if missing:
                raise EnergyInputError(f"{path}: missing headers: {', '.join(sorted(missing))}")
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise EnergyInputError(f"{path}: row {reader.line_num}: invalid row width")
                rows.append((reader.line_num, row))
            return rows
    except (OSError, UnicodeError, csv.Error) as error:
        raise EnergyInputError(f"{path}: CSV read error: {error}") from error


def read_meter(path, columns):
    samples = []
    for line, row in _csv_rows(path, columns.values()):
        values = {key: _number(row[source], f"{path}: row {line} field {source}")
                  for key, source in columns.items()}
        t, voltage, current = (values[key] for key in DEFAULT_COLUMNS)
        if voltage < 0 or current < 0:
            raise EnergyInputError(f"{path}: row {line}: voltage and current must be nonnegative")
        if samples and t <= samples[-1][0]:
            raise EnergyInputError(f"{path}: row {line}: timestamps must strictly increase")
        power = voltage * current * 1000
        if not math.isfinite(power):
            raise EnergyInputError(f"{path}: row {line}: nonfinite power")
        samples.append((t, power))
    if len(samples) < 2:
        raise EnergyInputError(f"{path}: meter requires at least 2 samples")
    return samples


def read_phases(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise EnergyInputError(f"{path}: JSON read error: {error}") from error
        if not isinstance(data, list):
            raise EnergyInputError(f"{path}: phases must be a list")
        rows = list(enumerate(data, 1))
    elif suffix == ".csv":
        rows = _csv_rows(path, PHASE_FIELDS)
    else:
        raise EnergyInputError(f"{path}: unsupported phase file extension")
    if not rows:
        raise EnergyInputError(f"{path}: phases must not be empty")
    phases, names = [], set()
    for line, row in rows:
        context = f"{path}: row {line}"
        if not isinstance(row, dict):
            raise EnergyInputError(f"{context}: phase must be an object")
        missing = set(PHASE_FIELDS) - row.keys()
        if missing:
            raise EnergyInputError(f"{context}: missing fields: {', '.join(sorted(missing))}")
        name = row["name"]
        if not isinstance(name, str) or not name.strip():
            raise EnergyInputError(f"{context}: name must be nonempty")
        name = name.strip()
        if name in names:
            raise EnergyInputError(f"{context}: duplicate phase name {name!r}")
        names.add(name)
        start = _number(row["start_s"], f"{context} start_s", suffix == ".json")
        end = _number(row["end_s"], f"{context} end_s", suffix == ".json")
        if start >= end or not math.isfinite(end - start):
            raise EnergyInputError(f"{context}: phase {name!r} has invalid duration")
        tokens = row["tokens"]
        if suffix == ".csv":
            if not re.fullmatch(r"[0-9]+", tokens):
                raise EnergyInputError(f"{context}: tokens must be a nonnegative integer")
            try:
                tokens = int(tokens)
            except ValueError as error:
                raise EnergyInputError(f"{context}: invalid tokens") from error
        elif isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise EnergyInputError(f"{context}: tokens must be a nonnegative integer")
        phases.append(dict(name=name, start_s=start, end_s=end, tokens=tokens))
    return phases


def integrate_phase(samples, phase):
    name, start, end = phase["name"], phase["start_s"], phase["end_s"]
    if start < samples[0][0] or end > samples[-1][0]:
        raise EnergyInputError(f"phase {name!r}: outside meter log")
    times = [sample[0] for sample in samples]
    lo, hi = bisect_left(times, start), bisect_right(times, end)
    if hi - lo < 2:
        raise EnergyInputError(f"phase {name!r}: fewer than 2 samples")

    def power_at(t):
        index = bisect_left(times, t)
        if index < len(times) and times[index] == t:
            return samples[index][1]
        t0, p0 = samples[index - 1]
        t1, p1 = samples[index]
        fraction = (t - t0) / (t1 - t0)
        return p0 * (1 - fraction) + p1 * fraction

    try:
        points = [(start, power_at(start))]
        points.extend((t, p) for t, p in samples[lo:hi] if start < t < end)
        points.append((end, power_at(end)))
        energy = math.fsum((t1 - t0) * (p0 / 2 + p1 / 2)
                           for (t0, p0), (t1, p1) in zip(points, points[1:]))
        duration = end - start
        mean = energy / duration
        if not all(math.isfinite(v) for v in (duration, energy, mean)):
            raise ValueError("nonfinite result")
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise EnergyInputError(f"phase {name!r}: invalid integration: {error}") from error
    return dict(name=name, duration_s=duration, mean_power_mw=mean,
                energy_mj=energy, tokens=phase["tokens"])


def analyze(samples, phases, idle=None):
    results = [integrate_phase(samples, phase) for phase in phases]
    if idle is not None and idle not in {result["name"] for result in results}:
        raise EnergyInputError(f"unknown idle phase {idle!r}")
    baseline = next((result["mean_power_mw"] for result in results
                     if result["name"] == idle), 0)
    for result in results:
        try:
            net = result["energy_mj"] - baseline * result["duration_s"]
            per_token = net / result["tokens"] if result["tokens"] else None
            if not math.isfinite(net) or (per_token is not None and not math.isfinite(per_token)):
                raise ValueError("nonfinite result")
        except (OverflowError, ValueError) as error:
            raise EnergyInputError(f"phase {result['name']!r}: invalid calculation: {error}") from error
        result["net_energy_mj"] = net
        result["mj_per_token"] = per_token
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meter", help="meter CSV file")
    parser.add_argument("phases", help="phase JSON or CSV file")
    parser.add_argument("--idle", help="phase name used as the idle power baseline")
    parser.add_argument("--columns", type=parse_columns, default=DEFAULT_COLUMNS,
                        help="comma-separated meter column overrides, canonical=source")
    args = parser.parse_args(argv)
    try:
        results = analyze(read_meter(args.meter, args.columns), read_phases(args.phases), args.idle)
        output = json.dumps(results, allow_nan=False)
    except EnergyInputError as error:
        print(f"energy_per_token: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
