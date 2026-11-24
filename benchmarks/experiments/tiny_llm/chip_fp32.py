"""
chip_fp32.py -- the comparison point of the binary 260K on chips without PSRAM (README "The comparison point on the
chip"): our fp32 model run by llama2.c's run.c (unchanged, chip_l2c/llama2c/), weights in flash, KV cache in RAM.

    python chip_fp32.py runs/fp32_v512d64l5kv4_lr3e-3_20k_s42 --board esp32 --port /dev/ttyUSB0
    python chip_fp32.py runs/fp32_v512d64l5kv4_lr3e-3_20k_s42 --board pico2_arm --port /dev/ttyACM1
    python chip_fp32.py runs/fp32_v512d64l5kv4_lr3e-3_20k_s42 --pc-only        # export + the PC checks only

Steps: exports runs/<run>/model.pt to a llama2.c version-0 checkpoint (runs/<run>/llama2c.bin: 7 int32 header, the
fp32 weights in its order, RoPE tables, tied head = positive vocab); greedy story in PyTorch (prompt "Once upon a time",
120 new tokens, no stop: the positions of chip_run.py's story); the same with run.c built on the PC (chip_l2c/pc_check.c)
-> must be equal; writes chip_l2c/src/gen_data.h + model_data.h, builds and uploads chip_l2c/ with PlatformIO, reads the
board: every story token against run.c on the PC, time per token. Writes results/chip_<board>_l2c_<run>.json.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import intsim as I  # noqa: E402
import train as T  # noqa: E402
from chip_run import BOARD_NAMES, PROMPT, RESET, model_sum, pio_platform, write_model_data  # noqa: E402
from compiler_core.chip.core import BOARD_MHZ, open_port, read_line  # noqa: E402

PROJ = os.path.join(HERE, "chip_l2c")
N_NEW = 120


def export_llama2c(model, a, path):
    """llama2.c's legacy (version 0) checkpoint, as its export.py legacy_export writes it"""
    assert a["weights"] == "fp32" and a.get("head", "fp32") == "fp32", "fp32 runs only (run.c is the fp32 reference)"
    blocks = model.blocks
    hidden = blocks[0].w1.weight.shape[0]
    vocab, dim = model.emb.weight.shape
    hd = dim // a["heads"]
    out = [struct.pack("7i", dim, hidden, len(blocks), a["heads"], a.get("kv_heads") or a["heads"], vocab, a["seq"])]
    def put(t):
        out.append(t.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes())
    put(model.emb.weight)
    for name in ("n1", "q", "k", "v", "o", "n2", "w1", "w2", "w3"):
        for b in blocks:
            put(getattr(b, name).weight)
    put(model.norm.weight)
    freq = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(a["seq"]).float(), freq)
    put(ang.cos())
    put(ang.sin())
    data = b"".join(out)
    with open(path, "wb") as f:
        f.write(data)
    return data


def torch_story(model, ids, n_new, seq):
    ids = list(ids)
    with torch.no_grad():
        for _ in range(n_new):
            z = model(torch.tensor([ids[-seq:]]))
            ids.append(int(z[0, -1].argmax()))
    return ids


def pc_story(model_bin, ids, n_new):
    exe = os.path.join(PROJ, "pc_check")
    subprocess.run(["cc", "-O2", "-o", exe, os.path.join(PROJ, "pc_check.c"), os.path.join(PROJ, "src", "l2c_mcu.c"),
                    "-lm"], check=True)
    r = subprocess.run([exe, model_bin, str(n_new)] + [str(i) for i in ids], check=True, capture_output=True, text=True)
    return list(ids) + [int(x) for x in r.stdout.split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--board", choices=("esp32", "pico2_arm", "pico2_riscv"), default="esp32")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--pc-only", action="store_true", help="export and the PC checks only, no board")
    ap.add_argument("--no-upload", dest="upload", action="store_false")
    args = ap.parse_args()
    import sentencepiece as spm
    run = os.path.normpath(args.run)
    vocab = torch.load(os.path.join(run, "model.pt"), map_location="cpu")["args"].get("vocab", 4096)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(T.DATA, f"tok{vocab}.model"))
    model, a = I.load_model(run, vocab, "cpu", float_only=True)
    mb_path = os.path.join(run, "llama2c.bin")
    data = export_llama2c(model, a, mb_path)
    prompt = [sp.bos_id()] + sp.encode(PROMPT)
    ref = torch_story(model, prompt, N_NEW, a["seq"])
    ids = pc_story(mb_path, prompt, N_NEW)
    torch_equal = sum(int(x == y) for x, y in zip(ref[len(prompt):], ids[len(prompt):]))
    print(f"run.c on the PC vs PyTorch: {torch_equal}/{N_NEW} story tokens equal")
    print(sp.decode(ids[1:]))
    if torch_equal != N_NEW:
        raise RuntimeError("run.c on the PC differs from PyTorch: the export is wrong")
    if args.pc_only:
        return

    lines = ["/* written by chip_fp32.py: do not edit */", "#pragma once", "#include <stdint.h>",
             f"#define MODEL_BYTES {len(data)}L", f"#define MODEL_SUM {model_sum(data)}u",
             f"#define GEN_IDS {len(ids)}", f"#define GEN_PROMPT {len(prompt)}",
             "static const int16_t gen_ids[GEN_IDS] = {" + ",".join(map(str, ids)) + "};"]
    gen_id = zlib.crc32("\n".join(lines).encode())
    lines += [f"#define GEN_ID {gen_id}u", ""]
    with open(os.path.join(PROJ, "src", "gen_data.h"), "w") as f:
        f.write("\n".join(lines))
    write_model_data(run, os.path.join(PROJ, "src", "model_data.h"), "llama2c.bin")
    if args.upload:
        print("building and uploading the firmware...")
        subprocess.run(["pio", "run", "-d", PROJ, "-e", args.board, "-t", "upload", "--upload-port", args.port],
                       check=True, capture_output=True)

    gen, info = [], {}
    with open_port(args.port, 115200, RESET[args.board]) as s:
        t0 = last = time.time()
        while True:
            line = read_line(s)
            if line:
                last = time.time()
            elif time.time() - last > 300:
                raise RuntimeError("chip: no line from the board for 5 minutes (serial link or firmware stuck)")
            if not line:
                s.write(b"\x55")
            elif line.startswith("ERROR"):
                raise RuntimeError(line)
            elif line.startswith("ID:"):
                if int(line[3:]) != gen_id:
                    raise RuntimeError("the board was built for another run (run without --no-upload)")
                info["id"] = True
            elif line.startswith("BOARD:"):
                if line[6:] != args.board:
                    raise RuntimeError(f"the board runs the {line[6:]} firmware, not {args.board}")
                info["board"] = True
            elif line.startswith("L2C ready") or line.startswith("LOADED"):
                print("  " + line)
                info[line.split()[0].lower()] = line
            elif line.startswith("G:"):
                tok, ok, us = map(int, line[2:].split(","))
                gen.append((tok, ok, us))
                print((" " if sp.id_to_piece(tok).startswith("▁") else "") + sp.decode([tok]), end="", flush=True)
            elif line == "DONE":
                print()
                break
            if time.time() - t0 > 900:
                raise RuntimeError("chip: no DONE after 15 minutes")
    problems = [p for p, bad in (("no ID line", not info.get("id")), ("no BOARD line", not info.get("board")),
                                 (f"{len(gen)} story tokens, expected {N_NEW}", len(gen) != N_NEW)) if bad]
    if problems:
        raise RuntimeError("chip check failed, no result written: " + "; ".join(problems))
    oks = [g[1] for g in gen]
    first_diff = oks.index(0) if 0 in oks else None
    us = np.array([g[2] for g in gen])
    res = {"run": os.path.basename(run), "board": BOARD_NAMES[args.board],
           "kernel": "llama2.c run.c fp32 (commit 350e04f, unchanged, plain C, -O2)", "model_in": "flash",
           "model_bytes": len(data), "kv_cache": "fp32", "kv_cache_in": "internal ram",
           "context": len(ids), "cpu_mhz": BOARD_MHZ[args.board], "platform": pio_platform(PROJ, args.board), "ready": info.get("l2c"), "loaded": info.get("loaded"),
           "pc_vs_torch_equal": torch_equal, "story_tokens": len(gen), "story_equal": int(sum(oks)),
           "first_diverging_story_token": first_diff,
           "ms_per_token_story_mean": round(us.mean() / 1000, 3), "tokens_per_second_story": round(1e6 / us.mean(), 3),
           "story": sp.decode(prompt[1:] + [g[0] for g in gen])}
    out = os.path.join(HERE, "results", f"chip_{args.board}_l2c_{os.path.basename(run)}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "story"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
