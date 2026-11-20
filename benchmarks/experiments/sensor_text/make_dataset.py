"""Generate synthetic sensor events and template-held-out English sentences."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    "har_ultracompact": ("benchmarks/conv1d/har_ultracompact/har_data.py", "CLASSES"),
    "ecg_mitbih": ("benchmarks/conv1d/ecg_mitbih/ecg_data.py", "CLASSES"),
    "fomo_voc_person": ("benchmarks/conv2d/fomo_voc_person/fomo_classes.py", "VOC_CLASSES"),
}
TEMPLATES = {
    "har_ultracompact": (
        "Activity {class} lasted {duration_seconds} seconds.",
        "The {class} activity was recorded for {duration_seconds} seconds.",
        "For {duration_seconds} seconds, the activity label was {class}.",
        "Recorded activity: {class}, with a duration of {duration_seconds} seconds.",
    ),
    "ecg_mitbih": (
        "Beat class {class} was recorded with heart rate changing from {heart_rate_before} to {heart_rate_after} beats per minute.",
        "The heart rate readings were {heart_rate_before} then {heart_rate_after} beats per minute, with beat class {class}.",
        "For beat class {class}, heart rate was initially {heart_rate_before} and subsequently {heart_rate_after} beats per minute.",
        "Recorded beat class: {class}; heart rate readings: {heart_rate_before} then {heart_rate_after} beats per minute.",
    ),
    "fomo_voc_person": (
        "Detected {class} objects: {count}.",
        "The detection count for {class} was {count}.",
        "For class {class}, the recorded detection count was {count}.",
        "Recorded detections of class {class} total {count}.",
    ),
}
TEMPLATE_IDS = {split: {source: [f"{source}.{letter}" for letter in letters]
                        for source in SOURCES}
                for split, letters in (("train", "ab"), ("val", "c"), ("test", "d"))}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        raise ImportError(f"cannot load source module {path}: {exc}") from exc
    return module


def load_source_classes():
    classes = {}
    for source, (path, symbol) in SOURCES.items():
        module = _load_module(f"sensor_text_{source}", ROOT / path)
        labels = getattr(module, symbol, None)
        if (not isinstance(labels, (list, tuple)) or not labels
                or any(not isinstance(label, str) or not label.strip() for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError(f"invalid {symbol} in {path}: expected unique nonempty labels")
        classes[source] = list(labels)
    return classes


def make_event(source, class_name, rng):
    event = {"source": source, "class": class_name}
    if source == "har_ultracompact":
        event["duration_seconds"] = rng.randint(1, 120)
    elif source == "ecg_mitbih":
        event["heart_rate_before"] = rng.randint(40, 180)
        event["heart_rate_after"] = rng.randint(40, 180)
    elif source == "fomo_voc_person":
        event["count"] = rng.randint(0, 5)
    else:
        raise ValueError(f"unknown source: {source}")
    return event


def render_event(event, template_id):
    source = event["source"]
    ids = [f"{source}.{letter}" for letter in "abcd"]
    if source not in TEMPLATES or template_id not in ids:
        raise ValueError(f"unknown or mismatched template: {template_id}")
    return TEMPLATES[source][ids.index(template_id)].format(**event)


def generate_dataset(sizes, seed):
    classes = load_source_classes()
    pairs = [(source, label) for source, labels in classes.items() for label in labels]
    if len(sizes) != 3 or any(type(n) is not int or n < 0 for n in sizes):
        raise ValueError("sizes must contain three nonnegative integers")
    if sizes[0] < len(pairs):
        raise ValueError(f"train size must be at least {len(pairs)}")
    rng = random.Random(seed)
    splits = {}
    counts = {}
    for split, size in zip(("train", "val", "test"), sizes):
        splits[split] = []
        counts[split] = {source: dict.fromkeys(labels, 0) for source, labels in classes.items()}
        for i in range(size):
            source, label = pairs[i % len(pairs)]
            event = make_event(source, label, rng)
            event["template_id"] = rng.choice(TEMPLATE_IDS[split][source])
            splits[split].append({"event": event, "text": render_event(event, event["template_id"])})
            counts[split][source][label] += 1
    meta = {"seed": seed, "sizes": dict(zip(splits, sizes)), "classes": classes,
            "template_ids": TEMPLATE_IDS, "counts_per_class": counts}
    return splits, meta


def write_dataset(out, splits, meta):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    for split in ("train", "val", "test"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in splits[split]:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with (out / "meta.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(meta, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", required=True, nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        splits, meta = generate_dataset(args.n, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    except (ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        write_dataset(args.out, splits, meta)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
