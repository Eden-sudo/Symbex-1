"""Write native-endian headerless uint16 streams for tiny_llm.train.load_split."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

_spec = importlib.util.spec_from_file_location("sensor_modules", Path(__file__).with_name("_modules.py"))
_modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modules)
prompt = _modules.load("prompt.py")


def encode_row(row, sp):
    if sp.vocab_size() > 32768 or min(sp.bos_id(), sp.eos_id()) < 0:
        raise ValueError("tokenizer needs BOS/EOS and IDs fitting signed int16 load_split")
    text = prompt.example(row["event"], row["text"])
    ids = sp.encode(text)
    if sp.unk_id() in ids or sp.decode(ids) != text:
        raise ValueError("tokenizer does not losslessly encode this example")
    result = [sp.bos_id(), *ids, sp.eos_id()]
    if any(not 0 <= value <= 32767 for value in result):
        raise ValueError("token ID would overflow load_split")
    return result


def write_split(rows, sp, destination):
    """Stream rows to a new file; caller retains responsibility for split selection."""
    count = 0
    with Path(destination).open("xb") as stream:
        for row in rows:
            ids = encode_row(row, sp)
            stream.write(np.asarray(ids, dtype=np.uint16).tobytes())
            count += len(ids)
    return count


def prepare(dataset_dir, model, out):
    sp = spm.SentencePieceProcessor(model_file=str(model))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split in ("train", "val", "test"):
        with (Path(dataset_dir) / f"{split}.jsonl").open(encoding="utf-8") as stream:
            counts[split] = write_split((json.loads(line) for line in stream), sp, out / f"{split}.bin")
    (out / "meta.json").write_text(json.dumps({"tokens": counts, "vocab": sp.vocab_size(),
        "dtype": "uint16", "byteorder": __import__("sys").byteorder,
        "format": "BOS + prompt + sentence + </text> + EOS; no header"}, indent=2) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.tokenizer, args.out)


if __name__ == "__main__":
    main()
