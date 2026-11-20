"""Allocate disjoint loaded-model regions from architecture byte estimates."""
import argparse
import importlib.util
import json
from pathlib import Path

REGIONS = {"weights": "psram", "kv": "psram", "luts": "sram",
           "activations": "sram", "scratch": "sram"}


def legacy_sizes(**architecture):
    """Adapt memory_budget without duplicating its formulas.

    weights includes the entire model container (norms, RoPE and constant LUTs).
    activations includes all legacy working buffers, including dynamic LUT and
    scratch. Zero separate regions intentionally prevent double counting.
    New architecture planners should pass their disjoint component sizes instead.
    """
    spec = importlib.util.spec_from_file_location("budget", Path(__file__).with_name("memory_budget.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    b = module.budget(**architecture)
    return dict(weights=b["model_bin_bytes"], kv=b["kv_cache_bytes"], luts=0,
                activations=b["per_token_internal_ram_bytes"], scratch=0)


def plan(sizes, *, sram_bytes, psram_bytes=8 * 1024 * 1024,
         sram_reserved=0, psram_reserved=0, alignment=16):
    """Sizes are non-overlapping bytes, including guards, not parameter counts.

    SRAM capacity must be supplied from the intended firmware's allocation
    budget, never inferred from the chip's nominal total SRAM. Both arena bases
    must satisfy alignment; offsets are relative to their respective bases.
    """
    if set(sizes) != set(REGIONS):
        raise ValueError("expected weights, kv, luts, activations, scratch")
    values = [*sizes.values(), sram_bytes, psram_bytes, sram_reserved, psram_reserved]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("sizes and capacities must be nonnegative integer bytes")
    if type(alignment) is not int or alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    if psram_bytes > 8 * 1024 * 1024 or sram_bytes > 512 * 1024:
        raise ValueError("capacity exceeds N16R8 physical memory")
    caps = dict(psram=psram_bytes, sram=sram_bytes)
    reserved = dict(psram=psram_reserved, sram=sram_reserved)
    ends = dict(psram=0, sram=0)
    regions = {}
    for name, arena in REGIONS.items():
        offset = (ends[arena] + alignment - 1) // alignment * alignment
        regions[name] = dict(arena=arena, offset=offset, size=sizes[name])
        ends[arena] = offset + sizes[name]
    arenas = {}
    for arena in ends:
        used = (ends[arena] + alignment - 1) // alignment * alignment
        if used + reserved[arena] > caps[arena]:
            raise ValueError(f"{arena} budget exceeded")
        payload = sum(sizes[n] for n in REGIONS if REGIONS[n] == arena)
        arenas[arena] = dict(capacity=caps[arena], reserved=reserved[arena],
                             used=used, payload=payload, padding=used-payload,
                             free=caps[arena]-reserved[arena]-used)
    return dict(alignment=alignment, regions=regions, arenas=arenas)


def c_header(result):
    lines = ["/* Generated offsets; allocate aligned, separate PSRAM/SRAM arenas. */",
             "#ifndef BOARD_RAM_LAYOUT_H", "#define BOARD_RAM_LAYOUT_H",
             f'#define BOARD_RAM_ALIGNMENT {result["alignment"]}u']
    for arena, info in result["arenas"].items():
        lines.append(f'#define BOARD_{arena.upper()}_BYTES {info["used"]}u')
    for name, info in result["regions"].items():
        prefix = f'BOARD_{info["arena"].upper()}_{name.upper()}'
        lines.extend([f'#define {prefix}_OFFSET {info["offset"]}u',
                      f'#define {prefix}_BYTES {info["size"]}u'])
    return "\n".join(lines + ["#endif", ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sizes", type=Path, help="JSON with five disjoint component byte counts")
    ap.add_argument("--sram-bytes", type=int, required=True)
    ap.add_argument("--sram-reserved", type=int, default=0)
    ap.add_argument("--psram-reserved", type=int, default=0)
    ap.add_argument("--header", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = plan(json.loads(args.sizes.read_text()), sram_bytes=args.sram_bytes,
                      sram_reserved=args.sram_reserved, psram_reserved=args.psram_reserved)
    except (ValueError, TypeError) as exc:
        ap.error(str(exc))
    args.header.write_text(c_header(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
