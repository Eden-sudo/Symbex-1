"""Retokenize a bounded local training-text prefix; never read official test data."""
import argparse
import codecs
import json
from pathlib import Path
import tempfile

import numpy as np
import sentencepiece as spm

VOCABS = (1024, 2048, 4096)
MAX_BYTES = 4 * 1024 * 1024
DELIMITER = "<|endoftext|>"


def retokenize(source, *, vocab=4096, max_bytes=65536, tmp_parent=None):
    """Return a new temporary directory, owned by the caller after success.

    A truncated final story is discarded, including partial UTF-8. At EOF the
    final story need not have a delimiter. Every 100th story is validation.
    Vocabulary size is a ceiling for small corpora, recorded alongside actual size.
    """
    if vocab not in VOCABS or type(vocab) is not int:
        raise ValueError("vocab must be 1024, 2048 or 4096")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("max_bytes must be in [1, 4194304]")
    with Path(source).open("rb") as stream:
        raw = stream.read(max_bytes)
        truncated = bool(stream.read(1))
    text = codecs.getincrementaldecoder("utf-8")().decode(raw, final=not truncated)
    parts = text.split(DELIMITER)
    if truncated:
        parts.pop()
    stories = [part.strip() for part in parts if part.strip()]
    train = [story for i, story in enumerate(stories) if i % 100]
    val = stories[::100]
    if not train or not val:
        raise ValueError("slice needs at least two complete nonempty stories")
    # Clean partial artifacts on error; retain the fresh directory on success.
    with tempfile.TemporaryDirectory(prefix="tiny-vocab-stage-", dir=tmp_parent) as stage:
        prefix = str(Path(stage) / f"tok{vocab}")
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(train), model_prefix=prefix, model_type="bpe",
            vocab_size=vocab, hard_vocab_limit=False, byte_fallback=True,
            character_coverage=1.0, split_digits=True,
            allow_whitespace_only_pieces=True, normalization_rule_name="identity",
            num_threads=1, max_sentence_length=MAX_BYTES,
            unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2,
        )
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        counts = {}
        for name, items in (("train", train), ("val", val)):
            ids = [token for story in items for token in [sp.bos_id(), *sp.encode(story)]]
            np.asarray(ids, dtype="<u2").tofile(Path(stage) / f"{name}.bin")
            counts[name] = len(ids)
        info = {"requested_vocab": vocab, "actual_vocab": sp.get_piece_size(),
                "bytes_read": len(raw), "truncated": truncated, "val_every": 100,
                "stories": {"train": len(train), "val": len(val)}, "tokens": counts}
        (Path(stage) / "prepare.json").write_text(json.dumps(info, indent=2) + "\n")
        output = Path(tempfile.mkdtemp(prefix="tiny-vocab-", dir=tmp_parent))
        for artifact in Path(stage).iterdir():
            artifact.rename(output / artifact.name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local training text, not official valid/test")
    parser.add_argument("--vocab", type=int, choices=VOCABS, default=4096)
    parser.add_argument("--max-bytes", type=int, default=65536)
    parser.add_argument("--tmp-parent", type=Path)
    args = parser.parse_args()
    print(retokenize(args.source, vocab=args.vocab, max_bytes=args.max_bytes,
                     tmp_parent=args.tmp_parent))


if __name__ == "__main__":
    main()
