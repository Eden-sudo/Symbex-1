"""Result records following docs/en/methodology.md, including host-only evidence."""
import argparse
import json
import math
from pathlib import Path
import statistics


def result(seeds, metrics, *, evidence, source, environment, split, commit, engine):
    """metrics maps names to {unit, values}; values align exactly with seeds.

    Population standard deviation (ddof=0) is explicit. A single seed is a
    ranking, not a multi-seed result. Two seeds are labelled preliminary.
    'measured' is scoped by environment: host CPU is not board evidence.
    """
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if evidence not in ("estimate", "measured"):
        raise ValueError("evidence must be estimate or measured")
    if split not in ("train", "validation", "test", "synthetic"):
        raise ValueError("split must identify train/validation/test/synthetic")
    for name, value in (("source", source), ("environment", environment),
                        ("commit", commit), ("engine", engine)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must describe provenance")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a nonempty mapping")
    summaries = {}
    for name, metric in metrics.items():
        if not isinstance(name, str) or not name or not isinstance(metric, dict):
            raise ValueError("metrics require named records")
        if not isinstance(metric.get("unit"), str) or not metric["unit"].strip():
            raise ValueError("every metric requires a unit")
        values = metric.get("values")
        if not isinstance(values, list) or len(values) != len(seeds):
            raise ValueError("metric values must align with seeds")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("metric values must be finite numbers")
        summaries[name] = {"unit": metric["unit"], "values": list(values),
                           "mean": statistics.fmean(values),
                           "std": statistics.pstdev(values), "std_ddof": 0}
    return {"schema_version": 1, "seeds": list(seeds), "seed_count": len(seeds),
            "status": "ranking" if len(seeds) == 1 else "preliminary" if len(seeds) == 2 else "multi_seed",
            "evidence": evidence, "is_estimate": evidence == "estimate", "source": source,
            "environment": environment, "split": split, "commit": commit, "engine": engine,
            "metrics": summaries}


def write_result(path, seeds, metrics, **provenance):
    record = result(seeds, metrics, **provenance)
    # Validate and serialize before opening: malformed records cannot truncate a result.
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="JSON with seeds, metrics and provenance keyword arguments")
    p.add_argument("output", help="new result path; existing results are never overwritten")
    a = p.parse_args(argv)
    write_result(a.output, **json.loads(Path(a.input).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
