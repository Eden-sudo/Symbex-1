"""Train a small, lossless domain tokenizer on synthetic TRAIN examples only."""
import argparse
import importlib.util
from pathlib import Path

import sentencepiece as spm

_spec = importlib.util.spec_from_file_location("sensor_modules", Path(__file__).with_name("_modules.py"))
_modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modules)
prompt = _modules.load("prompt.py")
dataset = _modules.load("make_dataset.py")


def train_tokenizer(out, *, rows=256, vocab_size=512, seed=0, environment=False):
    """Create a fresh directory containing corpus.txt and tokenizer.{model,vocab}."""
    if type(vocab_size) is not int or not 512 <= vocab_size <= 1024:
        raise ValueError("vocab_size must be in [512, 1024]")
    generator = _modules.load("environment_dataset.py") if environment else dataset
    splits, _ = generator.generate_dataset([rows, 0, 0], seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    corpus = out / "corpus.txt"
    corpus.write_text("\n".join(prompt.example(row["event"], row["text"])
                                for row in splits["train"]) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(out / "tokenizer"), model_type="bpe",
        vocab_size=vocab_size, hard_vocab_limit=False, character_coverage=1.0,
        byte_fallback=True, normalization_rule_name="identity", add_dummy_prefix=False,
        remove_extra_whitespaces=False, split_by_number=True, split_digits=True,
        shuffle_input_sentence=False, num_threads=1,
        user_defined_symbols=[prompt.START, prompt.ANSWER, prompt.END, prompt.bilingual.environment.START],
        bos_id=1, eos_id=2, unk_id=0, pad_id=-1, minloglevel=2,
    )
    return spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment", action="store_true")
    args = parser.parse_args()
    train_tokenizer(args.out, rows=args.rows, vocab_size=args.vocab_size, seed=args.seed,
                    environment=args.environment)


if __name__ == "__main__":
    main()
