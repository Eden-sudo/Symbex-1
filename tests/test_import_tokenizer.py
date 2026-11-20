import struct

import pytest

from tiny_import_helpers import load

reader = load("importers/tokenizer")


def test_token_bytes_scores_and_ids(tmp_path):
    path = tmp_path / "tokenizer.bin"
    tokens = [b"", b"<s>", b"</s>", b"\xff", "▁hi".encode(), b"a\x00b"]
    scores = [-3.5, 0., 1.25, -2., 8., 7.5]
    path.write_bytes(struct.pack("<i", 5) + b"".join(
        struct.pack("<fi", score, len(token)) + token for score, token in zip(scores, tokens)))
    assert reader.read_tokenizer(path, 6) == dict(max_token_length=5, tokens=tokens, scores=scores)


@pytest.mark.parametrize("data,vocab,match", [
    (b"", 1, "truncated"), (struct.pack("<i", -1), 1, "negative"),
    (struct.pack("<ifi", 1, 0., 2) + b"aa", 1, "invalid"),
    (struct.pack("<ifi", 1, float("nan"), 1) + b"a", 1, "invalid"),
    (struct.pack("<ifi", 1, 0., 1), 1, "truncated"),
    (struct.pack("<ifi", 1, 0., 0) + b"x", 1, "trailing"),
    (b"", 0, "positive"),
])
def test_malformed(tmp_path, data, vocab, match):
    path = tmp_path / "bad.bin"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=match):
        reader.read_tokenizer(path, vocab)
