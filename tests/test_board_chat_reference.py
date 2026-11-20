import sys
import pytest
import torch
from tiny_import_helpers import load


def test_toy_integer_model(monkeypatch):
    train = load("train")
    monkeypatch.setitem(sys.modules, "train", train)
    integer, ref = load("intsim"), load("board_chat_reference")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = train.TinyLM(8, 64, 1, 1, 32, 16, "binary").eval()
        params = integer.integerize(model, "binary")
        def forward(ids):
            z, hm = integer.forward(params, torch.tensor([ids]))
            return z[0, -1].tolist(), int(hm[0, -1])
        prompt = [1, 3, 4, 2, 5]
        result = ref.reference(forward, prompt, vocab=8, context=16, max_tokens=3)
        history, expected, hashes = prompt[:], [], []
        for _ in range(3):
            scores, hm = forward(history)
            hashes.append((sum(z*(i+1)*0x9E3779B97F4A7C15 for i, z in enumerate(scores))
                           + hm*0xD6E8FEB86659FD93) % 2**64)
            token = scores.index(max(scores))
            if token == 2:
                break
            expected.append(token)
            history.append(token)
        assert result["expected_ids"] == expected
        assert result["output_hashes"] == hashes
        assert prompt == [1, 3, 4, 2, 5]
        assert "CHAT_REFERENCE_ID" in ref.c_header(result)
    finally:
        torch.set_num_threads(old)


def test_eos_tie_context_and_invalid():
    m = load("board_chat_reference")
    def run(f, **kw):
        return m.reference(f, [1], vocab=3, context=kw.get("context", 4), max_tokens=kw.get("limit", 2))
    eos = run(lambda ids: ([-4, -3, 0], 7))
    assert eos["expected_ids"] == [] and eos["stop_reason"] == 1
    assert len(eos["output_hashes"]) == 1
    assert "chat_expected_ids[1] = {0}" in m.c_header(eos)
    tie = run(lambda ids: ([4, 4, 0], 1))
    assert tie["expected_ids"] == [0, 0] and tie["stop_reason"] == 3
    def forbidden(ids):
        raise AssertionError("forward not allowed")
    assert run(forbidden, limit=0)["stop_reason"] == 3
    assert run(forbidden, context=1)["stop_reason"] == 4
    with pytest.raises(ValueError):
        run(lambda ids: ([1.0, 2, 3], 1))
