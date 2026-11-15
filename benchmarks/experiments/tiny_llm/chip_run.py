"""
chip_run.py -- tiny_llm on the ESP32-S3 (N16R8, native USB port): exactness against intsim.py and tokens per second.

    python chip_run.py runs/binary_d384l8_s42 --port /dev/ttyACM0
    python chip_run.py runs/binary_d384l8_s42 --port /dev/ttyACM0 --no-upload     # firmware and model already there

Needs runs/<run>/model.bin (python intsim.py <run> --export). Steps: writes chip/src/check_data.h (the validation
windows, a hash of every output of intsim.py, the greedy story), builds and uploads chip/ with PlatformIO, writes
model.bin to the "model" partition (0x310000) with esptool, then reads the board:
    every token of the windows: the hash of its 4096 outputs + final max equal to intsim.py's (must be 100 %)
    the greedy story: every token equal to intsim.py's, time per token (attention grows with the position)
Writes results/chip_s3_<run>.json and prints the story the chip wrote.
"""
import argparse
import json
import os
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
from compiler_core.chip.core import open_port, read_line  # noqa: E402

MODEL_OFFSET = 0x310000
PROMPT = "Once upon a time"
PARTS = ("embedding_norms", "qkv", "attention_values", "o", "w1_w3", "swiglu", "w2", "output_head",
         "attention_rope_cache", "attention_scores", "attention_softmax", "attention_division")  # TINYLM_PROF in tinylm.c


def out_hash(z, hm):
    """The firmware's hash: sum_r z[r] * (r + 1) * K + hm * K2, all mod 2^64."""
    zu = z.astype(np.int64).view(np.uint64)
    k = (np.arange(1, zu.shape[-1] + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    with np.errstate(over="ignore"):
        return (zu * k).sum(axis=-1, dtype=np.uint64) + hm.astype(np.int64).view(np.uint64) * np.uint64(
            0xD6E8FEB86659FD93)


def model_sum(data):
    s = 0
    for b in data:        # sum = sum * 31 + byte (uint32), as in main.cpp
        s = (s * 31 + b) & 0xFFFFFFFF
    return s


def write_check_data(run, windows, path):
    import sentencepiece as spm
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    sp = spm.SentencePieceProcessor(model_file=os.path.join(T.DATA, "tok4096.model"))
    model, a = I.load_model(run, sp.get_piece_size(), dev)
    P = I.to_device(I.integerize(model, a["weights"]), dev)
    val = T.load_split("val", dev)
    w = val[torch.arange(windows, device=dev)[:, None] * a["seq"] + torch.arange(a["seq"], device=dev)].long()
    z, hm = I.forward(P, w)
    h = out_hash(z.cpu().numpy(), hm.cpu().numpy())
    ids = I.generate(P, sp, PROMPT, 120, a["seq"], dev)
    n_prompt = 1 + len(sp.encode(PROMPT))
    assert len(ids) - 1 <= a["seq"], "story longer than the context: the C has no sliding window"
    with open(os.path.join(run, "model.bin"), "rb") as f:
        mb = f.read()
    tok = w.cpu().numpy()
    lines = ["/* written by chip_run.py: do not edit */", "#pragma once", "#include <stdint.h>",
             f"#define MODEL_BYTES {len(mb)}L", f"#define MODEL_SUM {model_sum(mb)}u",
             f"#define CHECK_WINDOWS {windows}", f"#define CHECK_TOKENS {tok.shape[1]}",
             f"#define GEN_IDS {len(ids)}", f"#define GEN_PROMPT {n_prompt}",
             "static const int16_t check_tokens[CHECK_WINDOWS][CHECK_TOKENS] = {"]
    lines += ["  {" + ",".join(map(str, r)) + "}," for r in tok]
    lines += ["};", "static const uint64_t check_hash[CHECK_WINDOWS][CHECK_TOKENS] = {"]
    lines += ["  {" + ",".join(f"{int(v)}ull" for v in r) + "}," for r in h]
    lines += ["};", "static const int16_t gen_ids[GEN_IDS] = {" + ",".join(map(str, ids)) + "};"]
    check_id = zlib.crc32("\n".join(lines).encode())   # the firmware prints it: proves which run it was built for
    lines += [f"#define CHECK_ID {check_id}u", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return ids, n_prompt, sp, check_id, a["seq"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--windows", type=int, default=2, help="validation windows of 256 tokens checked on the chip")
    ap.add_argument("--no-upload", dest="upload", action="store_false")
    ap.add_argument("--linear", choices=("lut8", "symbex"), default="lut8",
                    help="1-bit linear layers: lut8 (tinylm.c's table) or symbex (lib_symbex's 8-bit x 1-bit kernel)")
    args = ap.parse_args()
    run = os.path.normpath(args.run)
    proj = os.path.join(HERE, "chip")
    ids, n_prompt, sp, check_id, seq = write_check_data(run, args.windows, os.path.join(proj, "src", "check_data.h"))
    if args.upload:
        print("building and uploading the firmware...")
        env = dict(os.environ, PLATFORMIO_BUILD_FLAGS="-DTINYLM_SYMBEX_LINEAR" if args.linear == "symbex" else "")
        subprocess.run(["pio", "run", "-d", proj, "-t", "upload", "--upload-port", args.port], check=True,
                       capture_output=True, env=env)
        print("writing model.bin to the model partition...")
        subprocess.run(["pio", "pkg", "exec", "-p", "tool-esptoolpy", "--", "esptool.py", "--port", args.port,
                        "write_flash", hex(MODEL_OFFSET), os.path.abspath(os.path.join(run, "model.bin"))],
                       check=True, capture_output=True, cwd="/tmp")
        time.sleep(2)

    hashes, gen, info, prof = [], [], {}, {}
    with open_port(args.port, 115200, "usb") as s:
        t0 = time.time()
        while True:
            line = read_line(s)
            if not line:
                s.write(b"\x55")
            elif line.startswith("ERROR"):
                raise RuntimeError(line)
            elif line.startswith("ID:"):
                if int(line[3:]) != check_id:
                    raise RuntimeError("the board was built for another run or reference set (run without "
                                       "--no-upload)")
                info["id"] = True
            elif line.startswith("LINEAR:"):
                if line[7:] != args.linear:
                    raise RuntimeError(f"the board runs the {line[7:]} linear kernel, not {args.linear}")
            elif line.startswith("TINYLM ready") or line.startswith("LOADED"):
                print("  " + line)
                info[line.split()[0].lower()] = line
            elif line.startswith("H:"):
                w, t, ok, us = map(int, line[2:].split(","))
                hashes.append((w, t, ok, us))
                if t == seq - 1:
                    print(f"  window {w}: {sum(h[2] for h in hashes if h[0] == w)}/{seq} tokens equal, "
                          f"{np.mean([h[3] for h in hashes if h[0] == w]) / 1000:.1f} ms per token")
            elif line.startswith("P:"):
                k, us = map(int, line[2:].split(","))
                prof[PARTS[k]] = us
            elif line.startswith("G:"):
                tok, ok, us = map(int, line[2:].split(","))
                gen.append((tok, ok, us))
                piece = sp.id_to_piece(tok)
                print((" " if piece.startswith("▁") else "") + sp.decode([tok]), end="", flush=True)
            elif line == "DONE":
                print()
                break
            if time.time() - t0 > 1800:
                raise RuntimeError("chip: no DONE after 30 minutes")

    expected = {(w, t) for w in range(args.windows) for t in range(seq)}
    problems = []
    if not info.get("id"):
        problems.append("no ID line from the board")
    if len(hashes) != len(expected) or {(h[0], h[1]) for h in hashes} != expected:
        problems.append(f"{len(hashes)} validation records, expected each of {len(expected)} once")
    if len(gen) != len(ids) - n_prompt:
        problems.append(f"{len(gen)} story tokens, expected {len(ids) - n_prompt}")
    bad = sum(1 - h[2] for h in hashes) + sum(1 - g[1] for g in gen)
    if bad:
        problems.append(f"{bad} outputs differ from intsim.py")
    if problems:
        raise RuntimeError("chip check failed, no result written: " + "; ".join(problems))

    us_tok = np.array([h[3] for h in hashes])
    us_gen = np.array([g[2] for g in gen])
    story = sp.decode(ids[1:n_prompt] + [g[0] for g in gen])
    res = {"run": os.path.basename(run), "board": "esp32s3 N16R8", "kernel": ("lut8" if args.linear == "lut8" else "lib_symbex u8 x 1-bit (PIE)")
           + ({4: " linear, PIE int4 dot (output), PIE s8 dot (scores)",
               2: " linear, PIE int2 dot (output), PIE s8 dot (scores)"}.get(I.head_bits_of(run),
                                                                              " linear, PIE s8 dot (output, scores)")),
           "model_in": "psram",
           "kv_cache_in": "psram", "cpu_mhz": 240, "ready": info.get("tinylm"), "loaded": info.get("loaded"),
           "check_tokens": len(hashes), "check_equal": int(sum(h[2] for h in hashes)),
           "equal_percent": round(100 * sum(h[2] for h in hashes) / max(1, len(hashes)), 3),
           "story_tokens": len(gen), "story_equal": int(sum(g[1] for g in gen)),
           "ms_per_token_check_mean": round(us_tok.mean() / 1000, 3),
           "ms_per_token_at_position": {str(p): round(float(np.mean(us_tok[[i for i, h in enumerate(hashes)
                                                                            if h[1] == p]])) / 1000, 3)
                                        for p in (0, 63, 127, 255)},
           "ms_per_token_story_mean": round(us_gen.mean() / 1000, 3),
           "tokens_per_second_story": round(1e6 / us_gen.mean(), 3),
           "ms_per_token_by_part": {k: round(v / max(1, len(hashes)) / 1000, 3) for k, v in prof.items()},
           "story": story}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    suffix = "" if args.linear == "lut8" else "_" + args.linear
    out = os.path.join(HERE, "results", f"chip_s3_{os.path.basename(run)}{suffix}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "story"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
