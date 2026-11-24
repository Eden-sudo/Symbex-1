"""Illustrative synthetic alert policy, not calibrated safety thresholds."""
import re

PHRASES = {
    "en": {"co2_high": "high CO2", "battery_low": "low battery", "sound_high": "high sound level"},
    "es": {"co2_high": "CO2 alto", "battery_low": "batería baja", "sound_high": "nivel sonoro alto"},
}
PREFIX = {"en": "Alerts: ", "es": "Alertas: "}
NONE = {"en": "none", "es": "ninguna"}


def labels(record):
    result = []
    if record["co2"] >= 1000:
        result.append("co2_high")
    if record["battery"] <= 20:
        result.append("battery_low")
    if record["sound"] >= 85:
        result.append("sound_high")
    return result


def render(record):
    language = record["language"]
    return PREFIX[language] + (", ".join(PHRASES[language][label] for label in labels(record)) or NONE[language])


def extract(sentence, language):
    """Only a terminal, explicit alert clause is accepted; negated prose is unsupported."""
    match = re.search(r"(?:^|; )" + re.escape(PREFIX[language]) + r"([^.;]+)\.$", sentence)
    if not match:
        return set(), False
    body = match.group(1)
    if body == NONE[language]:
        return set(), True
    inverse = {text: label for label, text in PHRASES[language].items()}
    parts = body.split(", ")
    if len(parts) != len(set(parts)) or any(part not in inverse for part in parts):
        return set(), False
    return {inverse[part] for part in parts}, True


def score(record, sentence):
    expected = set(labels(record))
    mentioned, supported = extract(sentence, record["language"])
    return {"missed_alerts": len(expected - mentioned), "false_alerts": len(mentioned - expected),
            "alerts_supported": supported}
