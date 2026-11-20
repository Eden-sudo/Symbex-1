"""Plan N16R8 flash without building firmware or touching hardware."""
import argparse
import json
import re
from pathlib import Path

FLASH_BYTES = 16 * 1024 * 1024
MODEL_OFFSET = 0x310000
SECTOR = 4096


def plan(containers):
    """Ordered name -> byte size; first container must be the legacy 'model'.

    Reserve the existing NVS/factory app layout. Containers are sector-aligned
    data partitions, not OTA slots. Sizes include all container metadata.
    """
    if not containers or next(iter(containers)) != "model":
        raise ValueError("first container must be model")
    rows = [dict(name="nvs", type="data", subtype="nvs", offset=0x9000, size=0x5000),
            dict(name="app", type="app", subtype="factory", offset=0x10000, size=0x300000)]
    offset = MODEL_OFFSET
    for name, size in containers.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,14}", name) or name in ("nvs", "app"):
            raise ValueError("invalid or reserved partition name")
        if type(size) is not int or size <= 0:
            raise ValueError("container sizes must be positive integer bytes")
        allocated = (size + SECTOR - 1) // SECTOR * SECTOR
        if offset + allocated > FLASH_BYTES:
            raise ValueError("containers exceed 16 MiB flash")
        rows.append(dict(name=name, type="data", subtype="0x40", offset=offset,
                         size=allocated, payload_bytes=size))
        offset += allocated
    # ESP-IDF partition table has at most 95 entries plus its checksum.
    if len(rows) > 95:
        raise ValueError("too many partitions")
    return {"flash_bytes": FLASH_BYTES, "partitions": rows,
            "offsets": {r["name"]: r["offset"] for r in rows},
            "end_offset": offset, "free_bytes": FLASH_BYTES - offset}


def partitions_csv(result):
    lines = ["# Name, Type, SubType, Offset, Size, Flags"]
    for r in result["partitions"]:
        lines.append(f'{r["name"]},{r["type"]},{r["subtype"]},'
                     f'0x{r["offset"]:x},0x{r["size"]:x},')
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("containers", type=Path, help='JSON object: {"model": bytes, ...}')
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--offsets", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = plan(json.loads(args.containers.read_text()))
    except (ValueError, TypeError, AttributeError) as exc:
        ap.error(str(exc))
    args.csv.write_text(partitions_csv(result))
    args.offsets.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
