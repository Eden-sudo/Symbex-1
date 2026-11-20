"""Fine-tuning examples using the runtime's segment-tokenized chat template."""
from chat_format import MARKERS, build_chat, encode_turn


def format_dialogue(sp, turns):
    """Return (token_ids, loss_mask), aligned to tokens, not input positions.

    Only assistant content and its EOS are targets. Markers, BOS, other roles
    and the final open assistant marker are context only. For causal training
    use ids[:-1] as inputs and ids[1:], mask[1:] as targets and selection.
    No truncation/padding is implicit; an empty assistant still trains EOS.
    """
    turns = list(turns)
    ids = build_chat(sp, turns)
    mask = [False]
    for role, text in turns:
        span = encode_turn(sp, role, text)
        marker_size = len(sp.encode(MARKERS[role]))
        mask.extend([False] * marker_size)
        mask.extend([role == "assistant"] * (len(span) - marker_size))
    mask.extend([False] * len(sp.encode(MARKERS["assistant"])))
    return ids, mask
