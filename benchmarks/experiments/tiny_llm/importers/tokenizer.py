"""llama2.c tokenizer.bin: int32 max length, then (float score, int32 length, bytes).

The vocabulary count comes from the checkpoint, not this file. Entry index is
the token ID; bytes are preserved (no Unicode normalization or ID remapping).
run.c's BPE merge scores, BOS/EOS convention and byte fallback are tokenizer
behavior, not merely a list of strings. This is NOT a SentencePiece .model
converter: train.py's tok4096.model must not be substituted for imported IDs.
"""
import math
import struct


def read_tokenizer(path, vocab_size):
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer from the checkpoint")
    with open(path, "rb") as f:
        def read(n):
            data = f.read(n)
            if len(data) != n:
                raise ValueError("truncated tokenizer.bin")
            return data

        maximum, = struct.unpack("<i", read(4))
        if maximum < 0:
            raise ValueError("negative tokenizer maximum token length")
        tokens, scores = [], []
        for _ in range(vocab_size):
            score, size = struct.unpack("<fi", read(8))
            if not math.isfinite(score) or size < 0 or size > maximum:
                raise ValueError("invalid tokenizer score or token length")
            tokens.append(read(size))
            scores.append(score)
        if f.read(1):
            raise ValueError("trailing tokenizer data: check vocab_size")
    return {"max_token_length": maximum, "tokens": tokens, "scores": scores}
