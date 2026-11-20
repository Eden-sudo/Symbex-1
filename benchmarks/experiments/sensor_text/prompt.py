"""Canonical sensor prompt v1; see prompt_format.md."""
import re

START = "<sensor_v1>"
ANSWER = "<text>"
END = "</text>"
# Field name, unit, inclusive minimum and maximum; order is part of the wire format.
FIELDS = {
    "har_ultracompact": (("duration_seconds", "s", 1, 120),),
    "ecg_mitbih": (("heart_rate_before", "bpm", 40, 180),
                   ("heart_rate_after", "bpm", 40, 180)),
    "fomo_voc_person": (("count", "objects", 0, 5),),
}


def canonical_record(event):
    source = event.get("source")
    if source not in FIELDS:
        raise ValueError("unknown sensor source")
    label = event.get("class")
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", label):
        raise ValueError("class must be a 1..64 character ASCII identifier")
    names = {field[0] for field in FIELDS[source]}
    if set(event) - {"source", "class", "template_id"} != names:
        raise ValueError("missing or extra sensor fields")
    result = {"source": source, "class": label}
    for name, _, low, high in FIELDS[source]:
        value = event[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        result[name] = value
    return result


def serialize(event):
    record = canonical_record(event)
    parts = [START, f"source={record['source']}", f"class={record['class']}"]
    parts.extend(f"{name}={record[name]} {unit}" for name, unit, _, _ in FIELDS[record["source"]])
    return "|".join(parts) + "|" + ANSWER


def parse(text):
    if not isinstance(text, str):
        raise ValueError("prompt must be text")
    parts = text.split("|")
    if len(parts) < 5 or parts[0] != START or parts[-1] != ANSWER:
        raise ValueError("invalid prompt framing")
    try:
        record = dict(part.split("=", 1) for part in parts[1:-1])
        for name, unit, _, _ in FIELDS[record["source"]]:
            value, actual_unit = record[name].split(" ")
            if actual_unit != unit:
                raise ValueError("incorrect unit")
            record[name] = int(value)
        record = canonical_record(record)
        if serialize(record) != text:
            raise ValueError("noncanonical prompt")
        return record
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid or noncanonical prompt") from exc


def example(event, sentence):
    if not isinstance(sentence, str) or not sentence or any(c in sentence for c in "<>\r\n"):
        raise ValueError("sentence must be nonempty single-line text without framing tokens")
    return serialize(event) + sentence + END
