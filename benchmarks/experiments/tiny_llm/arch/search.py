"""Small analytical grid search; no torch, training, downloads or board access."""

import argparse
from dataclasses import asdict
import importlib.util
import itertools
import json
from pathlib import Path
import sys


def load_planner():
    name = "tiny_llm_arch_search_planner"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("planner.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load_planner()


def grid_configs(grid):
    """Yield valid configurations; malformed grids fail rather than vanish."""
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid must be a nonempty object of Config fields and value lists")
    if set(grid) - set(P.Config.__dataclass_fields__):
        raise ValueError("unknown Config field in grid")
    if any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("every grid axis must be a nonempty list")
    for values in itertools.product(*grid.values()):
        kwargs = dict(zip(grid, values))
        if "weight_bits" in kwargs:
            kwargs["weight_bits"] = tuple(kwargs["weight_bits"])
        yield P.Config(**kwargs)


def pareto(rows):
    """Maximize actual parameter count and decode speed; strict dominance.

    Exact objective ties survive: different context/precision choices may have
    different quality. Parameter count is not a quality score, especially for
    factorizations. Sorting is speed-first, not a recommendation of quality.
    """
    ordered = sorted(rows, key=lambda r: (-r["tokens_per_second"], -r["parameters"]))
    front = []
    best_parameters = -1
    best_speed = None
    for row in ordered:
        count, speed = row["parameters"], row["tokens_per_second"]
        if count > best_parameters:
            front.append(row)
            best_parameters, best_speed = count, speed
        elif count == best_parameters and speed == best_speed:
            front.append(row)
    return front


def search(configs, *, workload="mean", firmware="autoload", tokens=327680000, batch=128, seq=256,
           current_only=False, memory_kwargs=None):
    rows = []
    seen = set()
    rejected = dict(memory=0, training_memory=0, firmware=0, training_context=0)
    for c in configs:
        if c in seen:
            continue
        seen.add(c)
        m = P.memory(c, **(memory_kwargs or {}))
        if not m["fits"]:
            rejected["memory"] += 1
            continue
        if current_only and m["firmware_gaps"]:
            rejected["firmware"] += 1
            continue
        if seq > c.ctx:
            rejected["training_context"] += 1
            continue
        t = P.training(c, tokens, batch=batch, seq=seq)
        if not t["fits_24gib"]:
            rejected["training_memory"] += 1
            continue
        s = P.size_model(c)
        rows.append(dict(config=asdict(c), parameters=s["parameters"], dense_parameters=s["dense_parameters"],
                         tokens_per_second=P.latency(c, workload, firmware)["tokens_per_second"],
                         flash_bytes=s["flash_bytes"], sram_bytes=m["totals"]["sram"],
                         psram_bytes=m["totals"]["psram"], training_hours=t["hours"],
                         training_memory_bytes=t["memory_bytes"], firmware_gaps=m["firmware_gaps"]))
    return dict(evaluated=len(seen), feasible=len(rows), rejected=rejected,
                workload=workload, firmware=firmware, frontier=pareto(rows))


def default_configs():
    # Fix context/vocab/precision so faster does not just mean less capability.
    # Separate custom-grid runs can compare GQA, packed KV and LittleBit.
    for dim, layers, hidden, head_bits in itertools.product((256, 384, 512), (4, 8, 12), (768, 1024), (4, 8)):
        yield P.Config(dim=dim, layers=layers, heads=dim // 64, kv_heads=dim // 64,
                       hidden=hidden, head_bits=head_bits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, help="JSON object: Config fields mapped to value lists")
    parser.add_argument("--workload", choices=("mean", "full", "story"), default="mean")
    parser.add_argument("--firmware", choices=("autoload", "plain"), default="autoload",
                        help="latency calibration: current firmware (chip_run.py --autoload 3) or without autoload")
    parser.add_argument("--tokens", type=int, default=327680000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--current-only", action="store_true", help="exclude known firmware gaps")
    parser.add_argument("--top", type=int, default=10, help="table rows; --json always emits the full frontier")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.top <= 0 or args.tokens < 0 or args.batch <= 0 or args.seq <= 0:
        parser.error("top, batch, seq must be positive; tokens must be nonnegative")
    try:
        configs = grid_configs(json.loads(args.grid.read_text())) if args.grid else default_configs()
        result = search(configs, workload=args.workload, firmware=args.firmware, tokens=args.tokens, batch=args.batch,
                        seq=args.seq, current_only=args.current_only)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"ESTIMATES, not measurements; {args.workload} decode, {args.firmware} firmware; speed-ranked, not quality-ranked")
    print(f"{result['evaluated']} configs, {result['feasible']} feasible, {len(result['frontier'])} Pareto; rejected {result['rejected']}")
    print("rank d/L/H/KV/ff/vocab/ctx head/kv/weights  params(M) tok/s flash(MiB) psram(MiB) SRAM(KiB) train(h) gaps")
    for i, row in enumerate(result["frontier"][:args.top], 1):
        c = row["config"]
        shape = "/".join(str(c[k]) for k in ("dim", "layers", "heads", "kv_heads", "hidden", "vocab", "ctx"))
        precision = f"{c['head_bits']}/{c['kv_bits']}/" + ",".join(str(b) for b in c["weight_bits"])
        print(f"{i:>4} {shape} {precision} {row['parameters']/1e6:.3f} {row['tokens_per_second']:.2f} "
              f"{row['flash_bytes']/1024**2:.3f} {row['psram_bytes']/1024**2:.3f} "
              f"{row['sram_bytes']/1024:.1f} {row['training_hours']:.2f} {len(row['firmware_gaps'])}")


if __name__ == "__main__":
    main()
