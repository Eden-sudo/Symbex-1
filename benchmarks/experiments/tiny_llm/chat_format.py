"""Segment-tokenized chat template using existing vocabulary only.

prepare.py reserves only UNK=0/BOS=1/EOS=2; there are no chat role IDs.
Encode each marker and content independently, concatenate IDs (NOT equivalent
to encoding the concatenated text with SentencePiece). C takes those same
preencoded spans; it is a builder, not a second SentencePiece tokenizer.
"""
MARKERS = {"system": "System:", "user": "User:", "assistant": "Assistant:"}


def encode_turn(sp, role, text):
    if role not in MARKERS or not isinstance(text, str):
        raise ValueError("invalid role/text")
    return list(sp.encode(MARKERS[role])) + list(sp.encode(text)) + [sp.eos_id()]


def decode_turn(sp, role, ids):
    """Decode a framed turn's content; reject a wrong marker/terminator."""
    if role not in MARKERS:
        raise ValueError("invalid role")
    prefix = list(sp.encode(MARKERS[role]))
    if list(ids[:len(prefix)]) != prefix or len(ids) < len(prefix)+1 or ids[-1] != sp.eos_id():
        raise ValueError("invalid turn framing")
    return sp.decode(list(ids[len(prefix):-1]))


def build_chat(sp, turns):
    """BOS + complete framed turns + open assistant marker (no EOS)."""
    if (sp.unk_id(), sp.bos_id(), sp.eos_id()) != (0, 1, 2):
        raise ValueError("expected prepare.py special IDs 0/1/2")
    result = [1]
    for role, text in turns:
        result.extend(encode_turn(sp, role, text))
    result.extend(sp.encode(MARKERS["assistant"]))
    return result
