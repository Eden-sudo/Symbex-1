"""Seed-isolated environmental JSONL splits with explicit leakage auditing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random

_spec = importlib.util.spec_from_file_location("sensor_modules", Path(__file__).with_name("_modules.py"))
_modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modules)
bilingual = _modules.load("bilingual.py")
dataset = _modules.load("make_dataset.py")
SPLITS = ("train", "val", "test")


def pair_key(row):
    # Metadata must not disguise a duplicate. Histories remain genuine input facts.
    record = bilingual.environment.canonical_record(row["event"])
    return json.dumps([record, row["text"]], sort_keys=True, ensure_ascii=False)


def check_leakage(splits):
    seen = {}
    for split in SPLITS:
        for row in splits[split]:
            key = pair_key(row)
            if key in seen and seen[key] != split:
                raise ValueError(f"identical record/sentence pair crosses {seen[key]} and {split}")
            seen[key] = split
    return {"cross_split_pairs": 0}


def generate_dataset(sizes=(256, 64, 64), seed=0, mode="template_holdout"):
    if len(sizes) != 3 or any(type(n) is not int or n < 0 for n in sizes):
        raise ValueError("sizes must contain three nonnegative integers")
    if type(seed) is not int or mode not in ("seed", "template_holdout"):
        raise ValueError("integer seed and supported split mode required")
    seeds = {name: int.from_bytes(hashlib.sha256(f"sensor-v2:{seed}:{name}".encode()).digest()[:8], "big")
             for name in SPLITS}
    phrasings = {name: ([2] if name == "test" else [0, 1])
                 if mode == "template_holdout" else [0, 1, 2] for name in SPLITS}
    splits, seen = {}, set()
    for name, size in zip(SPLITS, sizes):
        rng = random.Random(seeds[name])
        rows = []
        for i in range(size):
            # Bounded retries fail closed instead of quietly leaking or looping forever.
            for _ in range(100):
                event = bilingual.environment.make_record(rng, rng.randint(2, 16))
                event["language"] = ("en", "es")[i % 2]
                phrasing = rng.choice(phrasings[name])
                event["template_id"] = f"environment.{event['language']}.{phrasing}"
                row = {"event": event, "text": bilingual.render(event, phrasing)}
                key = pair_key(row)
                if key not in seen:
                    break
            else:
                raise ValueError("could not generate a unique record/sentence pair")
            seen.add(key)
            rows.append(row)
        splits[name] = rows
    audit = check_leakage(splits)
    meta = {"schema": "sensor_v2", "synthetic": True, "seed": seed, "split_seeds": seeds,
            "sizes": dict(zip(SPLITS, sizes)), "mode": mode, "phrasings": phrasings,
            "languages": ["en", "es"], "leakage": audit}
    return splits, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", nargs=3, type=int, default=(256, 64, 64))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("seed", "template_holdout"), default="template_holdout")
    args = parser.parse_args()
    splits, meta = generate_dataset(args.n, args.seed, args.mode)
    dataset.write_dataset(args.out, splits, meta)


if __name__ == "__main__":
    main()
