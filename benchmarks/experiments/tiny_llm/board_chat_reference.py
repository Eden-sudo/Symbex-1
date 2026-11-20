"""Generate CPU integer chat-turn references; no board or firmware dependency."""
import argparse
import importlib.util
import json
import zlib
from pathlib import Path


def score_hash(scores, hm):
    """chip_run's uint64 weighted score hash, implemented with Python integers."""
    return (sum(int(z) * (i + 1) * 0x9E3779B97F4A7C15 for i, z in enumerate(scores))
            + int(hm) * 0xD6E8FEB86659FD93) & ((1 << 64) - 1)


def reference(forward, prompt, *, vocab, context, max_tokens, eos=2):
    """forward(full_history) -> (int64 scores[vocab], integer output scale).

    Hashes cover every generation decision, including terminal EOS; expected
    IDs contain emitted tokens only. Prompt prefill is not independently hashed.
    Stable ties choose the smallest ID. No rolling/truncation or penalties.
    """
    prompt = list(prompt)
    if (type(vocab) is not int or vocab < 3 or type(context) is not int
            or not 1 <= context <= 256 or type(max_tokens) is not int or max_tokens < 0
            or type(eos) is not int or not 0 <= eos < vocab or not prompt
            or len(prompt) > context
            or any(type(t) is not int or not 0 <= t < vocab for t in prompt)):
        raise ValueError("invalid chat reference bounds or IDs")
    history, generated, hashes = prompt[:], [], []
    reason = 3 if max_tokens == 0 else 4
    while len(generated) < max_tokens and len(history) < context:
        scores, hm = forward(history[:])
        scores = list(scores)
        if (len(scores) != vocab or any(type(z) is not int or not -(1 << 63) <= z < 1 << 63
                                       for z in scores) or type(hm) is not int):
            raise ValueError("forward must return int64 scores and integer scale")
        hashes.append(score_hash(scores, hm))
        token = max(range(vocab), key=scores.__getitem__)
        if token == eos:
            reason = 1
            break
        generated.append(token)
        history.append(token)
        reason = 3 if len(generated) == max_tokens else 4
    result = dict(version=1, prompt_ids=prompt, expected_ids=generated,
                  output_hashes=hashes, stop_reason=reason, vocab=vocab,
                  context=context, max_tokens=max_tokens, eos=eos)
    result["reference_id"] = zlib.crc32(json.dumps(result, sort_keys=True).encode())
    return result


def c_header(ref):
    lines = ["/* CPU chat reference; hashes include EOS decisions. */", "#pragma once",
             "#include <stdint.h>", f'#define CHAT_REFERENCE_ID {ref["reference_id"]}u',
             f'#define CHAT_STOP_REASON {ref["stop_reason"]}']
    for key, ctype in (("prompt_ids", "int32_t"), ("expected_ids", "int32_t"),
                       ("output_hashes", "uint64_t")):
        values = ref[key]
        lines.append(f"#define CHAT_{key.upper()}_COUNT {len(values)}")
        data = ",".join(str(v) + ("ULL" if ctype == "uint64_t" else "") for v in values) or "0"
        lines.append(f"static const {ctype} chat_{key}[{max(1, len(values))}] = {{{data}}};")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="trusted local training checkpoint directory")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--turns", type=Path, required=True, help='JSON [["user", "hello"], ...]')
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--header", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)
    import torch
    import sentencepiece as spm
    # CLI execution places this script's directory on sys.path, as intsim expects.
    import intsim
    spec = importlib.util.spec_from_file_location("chat_format", Path(__file__).with_name("chat_format.py"))
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)
    torch.set_num_threads(1)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model, arch = intsim.load_model(args.run, sp.get_piece_size(), "cpu")
    params = intsim.integerize(model, arch["weights"])
    def forward(ids):
        with torch.no_grad():
            z, hm = intsim.forward(params, torch.tensor([ids]))
        return z[0, -1].tolist(), int(hm[0, -1])
    ref = reference(forward, fmt.build_chat(sp, json.loads(args.turns.read_text())),
                    vocab=sp.get_piece_size(), context=arch["seq"], max_tokens=args.max_tokens)
    args.header.write_text(c_header(ref))
    args.json.write_text(json.dumps(ref, indent=2) + "\n")


if __name__ == "__main__":
    main()
