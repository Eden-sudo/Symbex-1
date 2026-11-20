import importlib.util
from pathlib import Path
import string

import pytest

DIR = Path(__file__).resolve().parents[1] / "benchmarks/experiments/sensor_text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_fact_in_every_phrasing_and_language():
    b, p = load("bilingual.py"), load("prompt.py")
    for r in b.environment.generate_records(24, 9):
        for language in ("en", "es"):
            event = dict(r, language=language)
            texts = [b.render(event, i) for i in range(3)]
            assert len(set(texts)) == 3
            assert p.parse(p.serialize(event)) == event
            assert f"|language={language}|" in p.serialize(event)
            assert "template" not in p.serialize(dict(event, template_id="test-only"))
            for name, values in b.facts(event).items():
                templates = b.ANALOG_TEMPLATES if name in b.environment.ANALOG else b.STATE_TEMPLATES
                for i, template in enumerate(templates[language]):
                    placeholders = {f for _, f, _, _ in string.Formatter().parse(template) if f}
                    assert placeholders == set(values)
                    assert template.format(**values) in texts[i]
                    assert all(str(value) in template.format(**values) for value in values.values())
    with pytest.raises(ValueError):
        b.render(r)
    with pytest.raises(ValueError):
        b.render(dict(r, language="fr"))


def test_reject_noncanonical_v2_and_preserve_v1():
    b, p = load("bilingual.py"), load("prompt.py")
    event = dict(b.environment.generate_records(1, 0)[0], language="es")
    text = p.serialize(event)
    for bad in (text + "\n", text.replace("|language=es", "|language=es|language=es"),
                text.replace("|language=es", "|language=fr"), text.replace(" ppm", " ppb")):
        with pytest.raises(ValueError):
            p.parse(bad)
    old = {"source": "fomo_voc_person", "class": "person", "count": 1}
    assert p.serialize(old) == "<sensor_v1>|source=fomo_voc_person|class=person|count=1 objects|<text>"
    assert p.parse(p.serialize(old)) == old
