"""Three controlled phrasings per environmental fact in each language."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("sensor_modules", Path(__file__).with_name("_modules.py"))
_modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modules)
environment = _modules.load("environment.py")

LABELS = {
    "en": ("Temperature", "Humidity", "CO2", "Battery", "Motion", "Light", "Sound level", "Door"),
    "es": ("Temperatura", "Humedad", "CO2", "Batería", "Movimiento", "Luz", "Nivel sonoro", "Puerta"),
}
DIRECTIONS = {"en": {"rising": "rising", "falling": "falling", "steady": "steady"},
              "es": {"rising": "en aumento", "falling": "en descenso", "steady": "estable"}}
STATES = {"en": {"motion": ("absent", "detected"), "door": ("closed", "open")},
          "es": {"motion": ("ausente", "detectado"), "door": ("cerrada", "abierta")}}
ANALOG_TEMPLATES = {
    "en": ("{label}: {value} {unit}, {trend} over the last {window} readings",
           "{label} reads {value} {unit}; trend over {window} readings: {trend}",
           "For {label}, the value is {value} {unit} and the {window}-reading trend is {trend}"),
    "es": ("{label}: {value} {unit}, {trend} durante las últimas {window} lecturas",
           "{label} marca {value} {unit}; tendencia de {window} lecturas: {trend}",
           "Para {label}, el valor es {value} {unit} y la tendencia de {window} lecturas es {trend}"),
}
STATE_TEMPLATES = {
    "en": ("{label}: {state}", "{label} state is {state}", "For {label}, the recorded state is {state}"),
    "es": ("{label}: {state}", "Estado de {label}: {state}", "Para {label}, el estado registrado es {state}"),
}


def facts(event):
    r = environment.canonical_record(event)
    language = r.get("language")
    if language not in LABELS:
        raise ValueError("bilingual targets require an explicit language")
    result = {}
    for label, (name, unit, _, _) in zip(LABELS[language], environment.FIELDS):
        values = {"label": label}
        if name in environment.ANALOG:
            values.update(value=r[name], unit=unit, window=r["window"],
                          trend=DIRECTIONS[language][r["trends"][name]])
        else:
            values["state"] = STATES[language][name][r[name]]
        result[name] = values
    return result


def render(event, phrasing=0):
    if type(phrasing) is not int or not 0 <= phrasing < 3:
        raise ValueError("phrasing must be 0, 1 or 2")
    language = event.get("language")
    clauses = []
    for name, values in facts(event).items():
        templates = ANALOG_TEMPLATES if name in environment.ANALOG else STATE_TEMPLATES
        clauses.append(templates[language][phrasing].format(**values))
    return "; ".join(clauses) + "."


def serialize(event):
    r = environment.canonical_record(event)
    if r.get("language") not in LABELS:
        raise ValueError("prompt requires an explicit language")
    parts = [environment.START, "source=environment", "language=" + r["language"],
             "window=" + str(r["window"])]
    for name, unit, _, _ in environment.FIELDS:
        parts.append(f"{name}={r[name]} {unit}")
        if name in environment.ANALOG:
            parts.append(name + "_history=" + ",".join(map(str, r["history"][name])))
            parts.append(name + "_trend=" + r["trends"][name])
    return "|".join(parts) + "|<text>"


def parse(text):
    try:
        parts = text.split("|")
        if parts[0] != environment.START or parts[-1] != "<text>":
            raise ValueError("framing")
        raw = dict(part.split("=", 1) for part in parts[1:-1])
        r = {"source": raw.pop("source"), "language": raw.pop("language"),
             "window": int(raw.pop("window")), "history": {}, "trends": {}}
        for name, unit, _, _ in environment.FIELDS:
            value, actual_unit = raw.pop(name).split(" ")
            if actual_unit != unit:
                raise ValueError("unit")
            r[name] = int(value)
            if name in environment.ANALOG:
                r["history"][name] = list(map(int, raw.pop(name + "_history").split(",")))
                r["trends"][name] = raw.pop(name + "_trend")
        if raw or serialize(r) != text:
            raise ValueError("noncanonical")
        return environment.canonical_record(r)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("invalid environmental prompt") from exc
