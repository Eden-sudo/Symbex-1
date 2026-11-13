"""
prepare.py -- TinyStories (GPT-4 version, Eldan & Li 2023) for the tiny language model study: a 4096-token
SentencePiece vocabulary and the stories as uint16 token files.

    python prepare.py                     # reads ../../data/tinystories/{train,valid}.txt
        (download once: https://huggingface.co/datasets/roneneldan/TinyStories, files TinyStoriesV2-GPT4-train.txt
         and TinyStoriesV2-GPT4-valid.txt, saved as train.txt and valid.txt)

Splits (the benchmark rule: every choice on a validation split carved out of the training data):
    train   every training story except the ones below
    val     every 100th training story (1 %)
    test    the official valid file, read only by train.py --test
The vocabulary is trained on 200,000 training stories (not val, not test). Every story starts with BOS (id 1).
Writes tok4096.model, train.bin, val.bin, test.bin and prepare.json next to the text files.
"""
import argparse
import json
import os
import time

import numpy as np
import sentencepiece as spm

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tinystories")
VOCAB = 4096
VAL_EVERY = 100
SP_STORIES = 200_000


def stories(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return [s.strip() for s in text.split("<|endoftext|>") if s.strip()]


def encode(sp, items, path):
    ids = sp.encode(items, num_threads=os.cpu_count())
    total = sum(len(t) + 1 for t in ids)
    out = np.empty(total, dtype=np.uint16)
    i = 0
    for t in ids:
        out[i] = sp.bos_id()
        out[i + 1:i + 1 + len(t)] = t
        i += len(t) + 1
    out.tofile(path)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DATA, help="folder with train.txt and valid.txt")
    args = ap.parse_args()
    t0 = time.time()
    train_all = stories(os.path.join(args.data, "train.txt"))
    test = stories(os.path.join(args.data, "valid.txt"))
    val = train_all[::VAL_EVERY]
    train = [s for i, s in enumerate(train_all) if i % VAL_EVERY]
    print(f"stories: train {len(train):,}, val {len(val):,}, test {len(test):,} ({time.time() - t0:.0f} s)")

    prefix = os.path.join(args.data, f"tok{VOCAB}")
    rng = np.random.default_rng(0)
    sample = [train[i] for i in rng.choice(len(train), SP_STORIES, replace=False)]
    spm.SentencePieceTrainer.train(sentence_iterator=iter(sample), model_prefix=prefix, model_type="bpe",
                                   vocab_size=VOCAB, character_coverage=1.0, byte_fallback=True,
                                   split_digits=True, allow_whitespace_only_pieces=True, normalization_rule_name="identity",
                                   num_threads=os.cpu_count(), max_sentence_length=16384,
                                   unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    print(f"vocabulary: {sp.get_piece_size()} tokens ({time.time() - t0:.0f} s)")

    counts = {name: encode(sp, items, os.path.join(args.data, f"{name}.bin"))
              for name, items in (("val", val), ("test", test), ("train", train))}
    for name, n in counts.items():
        print(f"{name}: {n:,} tokens")
    info = {"vocab": VOCAB, "val_every": VAL_EVERY, "sp_stories": SP_STORIES,
            "stories": {"train": len(train), "val": len(val), "test": len(test)}, "tokens": counts}
    with open(os.path.join(args.data, "prepare.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
