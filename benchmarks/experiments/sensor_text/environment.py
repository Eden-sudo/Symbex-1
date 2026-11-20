"""Synthetic environmental readings; integer units and endpoint trends, not telemetry."""
import random

START = "<sensor_v2>"
# Inclusive operating envelopes, not the entire physical range of each quantity.
FIELDS = (
    ("temperature", "C", -40, 85),
    ("humidity", "pct", 0, 100),
    ("co2", "ppm", 250, 5000),
    ("battery", "pct", 0, 100),
    ("motion", "bool", 0, 1),
    ("light", "lux", 0, 100000),
    ("sound", "dBA", 0, 140),
    ("door", "bool", 0, 1),
)
ANALOG = tuple(name for name, _, _, _ in FIELDS if name not in ("motion", "door"))


def trend(values):
    """Endpoint change across N readings; does not claim monotonic motion."""
    return "rising" if values[-1] > values[0] else "falling" if values[-1] < values[0] else "steady"


def canonical_record(event):
    names = {f[0] for f in FIELDS}
    expected = names | {"source", "window", "history", "trends"}
    if set(event) - {"template_id", "language"} != expected or event.get("source") != "environment":
        raise ValueError("missing or extra environmental fields")
    if "language" in event and event["language"] not in ("en", "es"):
        raise ValueError("language must be en or es")
    n = event["window"]
    if type(n) is not int or not 2 <= n <= 16:
        raise ValueError("window must be an integer in [2, 16]")
    if not isinstance(event["history"], dict) or set(event["history"]) != set(ANALOG):
        raise ValueError("history must contain every analog sensor")
    if not isinstance(event["trends"], dict) or set(event["trends"]) != set(ANALOG):
        raise ValueError("trends must contain every analog sensor")
    result = {"source": "environment", "window": n, "history": {}, "trends": {}}
    for name, _, low, high in FIELDS:
        value = event[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        result[name] = value
        if name in ANALOG:
            history = event["history"][name]
            if (not isinstance(history, list) or len(history) != n
                    or any(type(v) is not int or not low <= v <= high for v in history)
                    or history[-1] != value):
                raise ValueError(f"invalid {name} history")
            direction = trend(history)
            if event["trends"][name] != direction:
                raise ValueError(f"inconsistent {name} trend")
            result["history"][name] = list(history)
            result["trends"][name] = direction
    if "language" in event:
        result["language"] = event["language"]
    return result


def make_record(rng, window=4):
    if type(window) is not int or not 2 <= window <= 16:
        raise ValueError("window must be an integer in [2, 16]")
    event = {"source": "environment", "window": window, "history": {}, "trends": {}}
    for name, _, low, high in FIELDS:
        if name in ANALOG:
            values = [rng.randint(low, high) for _ in range(window)]
            event["history"][name] = values
            event["trends"][name] = trend(values)
            event[name] = values[-1]
        else:
            event[name] = rng.randint(low, high)
    return canonical_record(event)


def generate_records(size, seed):
    if type(size) is not int or size < 0 or type(seed) is not int:
        raise ValueError("size must be nonnegative and seed must be an integer")
    rng = random.Random(seed)
    return [make_record(rng, rng.randint(2, 16)) for _ in range(size)]
