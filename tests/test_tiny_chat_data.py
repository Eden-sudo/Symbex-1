import pytest
from tiny_runtime_support import LM, module
from test_tiny_runtime_format import Vocab


def test_handmade_masks(monkeypatch):
    monkeypatch.syspath_prepend(str(LM))
    data, template, sp = module("chat_data"), module("chat_format"), Vocab()
    turns = [("system", "Be brief"), ("user", "Hola"), ("assistant", "¡Hola!"),
             ("user", "Bye"), ("assistant", "")]
    ids, mask = data.format_dialogue(sp, iter(turns))
    assert ids == template.build_chat(sp, turns)
    assert len(ids) == len(mask)
    assert [t for t, selected in zip(ids, mask) if selected] == sp.encode("¡Hola!") + [2, 2]
    # Check exact boundaries independently, including first content and EOS.
    start = 1
    for role, text in turns:
        n = len(sp.encode(template.MARKERS[role]))
        size = len(sp.encode(text)) + 1
        assert mask[start:start+n] == [False] * n
        assert mask[start+n:start+n+size] == [role == "assistant"] * size
        start += n + size
    assert not any(mask[start:]) and not mask[0]


def test_no_assistant_and_invalid_role(monkeypatch):
    monkeypatch.syspath_prepend(str(LM))
    data = module("chat_data")
    for turns in ([], [("user", "Assistant: not an answer")]):
        ids, mask = data.format_dialogue(Vocab(), turns)
        assert len(ids) == len(mask) and not any(mask)
    with pytest.raises(ValueError):
        data.format_dialogue(Vocab(), [("tool", "no")])
