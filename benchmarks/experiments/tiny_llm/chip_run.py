"""
chip_run.py -- tiny_llm on a chip: exactness against intsim.py and tokens per second. Default the ESP32-S3 (N16R8,
native USB port, model and KV cache in PSRAM, chip/); --board esp32|pico2_arm|pico2_riscv: chips without PSRAM
(chip_mcu/: model.bin a const array in flash, the KV cache in internal RAM; small models such as the 260K one).

    python chip_run.py runs/binary_d384l8_s42 --port /dev/ttyACM0
    python chip_run.py runs/binary_d384l8_s42 --port /dev/ttyACM0 --no-upload     # firmware and model already there
    python chip_run.py runs/binary_d384l8_head4_s42 --linear symbex --repeat-probe copy   # timing of a --repeat 2 model
    python chip_run.py runs/binary_v512d64l5kv4_s42 --board esp32 --port /dev/ttyUSB0       # classic ESP32
    python chip_run.py runs/binary_v512d64l5kv4_s42 --board pico2_arm --port /dev/ttyACM1   # Pico 2, Cortex-M33

Needs runs/<run>/model.bin (python intsim.py <run> --export). Steps: writes chip/src/check_data.h (the validation
windows, a hash of every output of intsim.py, the greedy story), builds and uploads chip/ with PlatformIO, writes
model.bin to the "model" partition (0x310000) with esptool, then reads the board:
    every token of the windows: the hash of its 4096 outputs + final max equal to intsim.py's (must be 100 %)
    the greedy story: every token equal to intsim.py's, time per token (attention grows with the position)
Writes results/chip_s3_<run>.json (chip_<board>_<run>.json) and prints the story the chip wrote. On the boards
without PSRAM model.bin goes into the firmware (chip_mcu/src/model_data.h) instead of a partition.
--repeat-probe psram|copy: layers 2i and 2i + 1 run on block 2i's weights (the schedule of a train.py --repeat 2
model; psram = read again from PSRAM, copy = copied to internal RAM before layer 2i, see chip/src/tinylm_esp.c).
Timing only: the outputs then differ from intsim.py, so the exactness check is skipped and the JSON says so
(results/chip_s3_<run>_<linear>_repeat_<mode>.json).
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
from compiler_core.chip.core import BOARD_MHZ, open_port, read_line  # noqa: E402

MODEL_OFFSET = 0x310000
PROMPT = "Once upon a time"
BOARDS = ("s3", "esp32", "pico2_arm", "pico2_riscv")
BOARD_NAMES = {"s3": "esp32s3 N16R8", "esp32": "esp32 (classic, no PSRAM)", "pico2_arm": "rp2350 Cortex-M33",
               "pico2_riscv": "rp2350 Hazard3"}
RESET = {"s3": "usb", "esp32": "rts", "pico2_arm": "pico", "pico2_riscv": "pico"}   # compiler_core.chip.open_port
PARTS = ("embedding_norms", "qkv", "attention_values", "o", "w1_w3", "swiglu", "w2", "output_head",
         "attention_rope_cache", "attention_scores", "attention_softmax", "attention_division",  # TINYLM_PROF in tinylm.c
         "values_pack", "values_decode_2bit", "values_copy_window", "values_weight_bytes",       # --fine-prof only
         "scores_window_int8", "scores_subset_table", "values_wide_plain_c")


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


def write_model_data(run, path):
    """model.bin as a 16-byte aligned const array (the boards without PSRAM read it in place from flash)"""
    with open(os.path.join(run, "model.bin"), "rb") as f:
        mb = f.read()
    lines = ["/* written by chip_run.py: do not edit */", "#pragma once", "#include <stdint.h>",
             f"static const uint8_t model_bin[{len(mb)}] __attribute__((aligned(16))) = {{"]
    lines += ["  " + ",".join(str(b) for b in mb[i:i + 32]) + "," for i in range(0, len(mb), 32)]
    lines += ["};", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_check_data(run, windows, path, p_cut=None):
    import sentencepiece as spm
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    vocab = torch.load(os.path.join(run, "model.pt"), map_location="cpu")["args"].get("vocab", 4096)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(T.DATA, f"tok{vocab}.model"))
    model, a = I.load_model(run, sp.get_piece_size(), dev)
    P = I.to_device(I.integerize(model, a["weights"]), dev)
    kv = I.read_kv_trailer(os.path.join(run, "model.bin"), a["layers"], a["heads"])
    if kv is not None:      # the low-bit KV cache the C runs (intsim.py --export wrote it): the same in the reference
        assert all((B["tq"].cpu().numpy() == t).all() for B, t in zip(P["blocks"], kv["tq"])), "model.bin is stale"
        P["kv"] = {k: kv[k] for k in ("k_bits", "v_bits", "k_scale", "recent")}
    if p_cut is not None:   # the attention cut: the firmware is built with -DTINYLM_P_CUT (the ID proves the pair)
        P["kv"] = dict(P.get("kv") or {}, p_cut=p_cut)
    val = T.load_split("val", dev, vocab)
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
    return ids, n_prompt, sp, check_id, a["seq"], P.get("kv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--board", choices=sorted(BOARDS), default="s3",
                    help="s3 (PSRAM, chip/) or a chip without PSRAM (chip_mcu/: model in flash, KV cache in RAM)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--windows", type=int, default=2, help="validation windows of seq tokens checked on the chip")
    ap.add_argument("--no-upload", dest="upload", action="store_false")
    ap.add_argument("--linear", choices=("lut8", "symbex"), default="lut8",
                    help="1-bit linear layers: lut8 (tinylm.c's table) or symbex (lib_symbex's 8-bit x 1-bit kernel)")
    ap.add_argument("--p-cut", type=float, help="attention weights p < p_max e^-n are 0 (README 'Reading the cache'): "
                    "reference and firmware (-DTINYLM_P_CUT=round(128 n)) both")
    ap.add_argument("--fine-prof", action="store_true", help="time the parts inside the attention too "
                    "(-DTINYLM_FINE_PROF; results/..._prof.json: the extra marks cost a little)")
    ap.add_argument("--autoload", type=int, choices=(0, 1, 2, 3),
                    help="data cache autoload on a miss, size 0..3 (-DTINYLM_AUTOLOAD; "
                         "../chip_kernels/psram_prefetch_bench.py)")
    ap.add_argument("--autoload-kv", action="store_true", help="--autoload over the KV cache too (region 1)")
    ap.add_argument("--repeat-probe", choices=("psram", "copy"),
                    help="timing only: run layers in pairs on one block's weights (see the docstring)")
    args = ap.parse_args()
    if args.autoload_kv and args.autoload is None:
        ap.error("--autoload-kv needs --autoload")
    s3 = args.board == "s3"
    if not s3 and (args.linear != "lut8" or args.autoload is not None or args.repeat_probe or args.fine_prof):
        ap.error("--board " + args.board + ": lut8 linear only, no --autoload / --repeat-probe / --fine-prof (S3 options)")
    run = os.path.normpath(args.run)
    proj = os.path.join(HERE, "chip" if s3 else "chip_mcu")
    ids, n_prompt, sp, check_id, seq, kv = write_check_data(run, args.windows, os.path.join(proj, "src", "check_data.h"),
                                                          args.p_cut)
    if not s3:
        write_model_data(run, os.path.join(proj, "src", "model_data.h"))
    if args.upload:
        print("building and uploading the firmware...")
        flags = ["-DTINYLM_SYMBEX_LINEAR"] if args.linear == "symbex" else []
        if args.repeat_probe:
            flags.append(f"-DTINYLM_REPEAT_PROBE={1 if args.repeat_probe == 'psram' else 2}")
        if args.fine_prof:
            flags.append("-DTINYLM_FINE_PROF")
        if args.autoload is not None:
            flags.append(f"-DTINYLM_AUTOLOAD={args.autoload}")
            if args.autoload_kv:
                flags.append("-DTINYLM_AUTOLOAD_KV=1")
        if args.p_cut is not None:
            flags.append(f"-DTINYLM_P_CUT={min(2048, round(args.p_cut * 128))}")
        env = dict(os.environ, PLATFORMIO_BUILD_FLAGS=" ".join(flags))
        subprocess.run(["pio", "run", "-d", proj, "-e", args.board, "-t", "upload", "--upload-port", args.port],
                       check=True, capture_output=True, env=env)
    if args.upload and s3:
        print("writing model.bin to the model partition...")
        subprocess.run(["pio", "pkg", "exec", "-p", "tool-esptoolpy", "--", "esptool.py", "--port", args.port,
                        "write_flash", hex(MODEL_OFFSET), os.path.abspath(os.path.join(run, "model.bin"))],
                       check=True, capture_output=True, cwd="/tmp")
        time.sleep(2)

    hashes, gen, info, prof = [], [], {}, {}
    with open_port(args.port, 115200, RESET[args.board]) as s:
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
            elif line.startswith("BOARD:"):
                if line[6:] != args.board:
                    raise RuntimeError(f"the board runs the {line[6:]} firmware, not {args.board}")
                info["board"] = True
            elif line.startswith("LINEAR:"):
                if line[7:] != args.linear:
                    raise RuntimeError(f"the board runs the {line[7:]} linear kernel, not {args.linear}")
            elif line.startswith("AUTOLOAD:"):
                info["autoload"] = int(line[9:])
            elif line.startswith("AUTOLOAD_KV:"):
                info["autoload_kv_bytes"] = int(line[12:])
            elif line.startswith("REPEAT:"):
                info["repeat"] = int(line[7:])
            elif line.startswith("RC:"):
                copy_us, failed = map(int, line[3:].split(","))
                info["repeat_copy_us"], info["repeat_failed"] = copy_us, failed
            elif line.startswith("RAM:"):
                print("  " + line)
                info["ram"] = line
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
    if not s3 and not info.get("board"):
        problems.append("no BOARD line from the board")
    if len(hashes) != len(expected) or {(h[0], h[1]) for h in hashes} != expected:
        problems.append(f"{len(hashes)} validation records, expected each of {len(expected)} once")
    if len(gen) != len(ids) - n_prompt:
        problems.append(f"{len(gen)} story tokens, expected {len(ids) - n_prompt}")
    bad = sum(1 - h[2] for h in hashes) + sum(1 - g[1] for g in gen)
    want_repeat = {None: None, "psram": 1, "copy": 2}[args.repeat_probe]
    if info.get("repeat") != want_repeat:
        problems.append(f"the board runs repeat probe {info.get('repeat')}, not {want_repeat} (run without --no-upload)")
    if info.get("autoload") != args.autoload:
        problems.append(f"the board runs autoload {info.get('autoload')}, not {args.autoload} (run without --no-upload)")
    if args.autoload_kv != ("autoload_kv_bytes" in info):
        problems.append("the board's autoload KV region differs from --autoload-kv (run without --no-upload)")
    if bad and not args.repeat_probe:       # the probe changes the outputs on purpose: timing only
        problems.append(f"{bad} outputs differ from intsim.py")
    if info.get("repeat_failed"):
        problems.append("repeat probe: no internal RAM for a block copy")
    if problems:
        raise RuntimeError("chip check failed, no result written: " + "; ".join(problems))

    us_tok = np.array([h[3] for h in hashes])
    us_gen = np.array([g[2] for g in gen])
    story = sp.decode(ids[1:n_prompt] + [g[0] for g in gen])
    res = {"run": os.path.basename(run), "board": BOARD_NAMES[args.board], "kernel": ("lut8" if args.linear == "lut8" else "lib_symbex u8 x 1-bit (PIE)")
           + (({4: " linear, PIE int4 dot (output), PIE s8 dot (scores)",
               2: " linear, PIE int2 dot (output), PIE s8 dot (scores)"}.get(I.head_bits_of(run),
                                                                              " linear, PIE s8 dot (output, scores)")
              if s3 else " linear, plain C (output, scores, values)")),
           "model_in": "psram" if s3 else "flash",
           "kv_cache": kv or "int8", "p_cut_nats": args.p_cut, "fine_prof": args.fine_prof, "dcache_autoload": args.autoload, "dcache_autoload_kv_bytes": info.get("autoload_kv_bytes"), "kv_cache_in": "psram" if s3 else "internal ram", "cpu_mhz": BOARD_MHZ[args.board], "ready": info.get("tinylm"), "loaded": info.get("loaded"),
           "check_tokens": len(hashes), "check_equal": int(sum(h[2] for h in hashes)),
           "equal_percent": round(100 * sum(h[2] for h in hashes) / max(1, len(hashes)), 3),
           "story_tokens": len(gen), "story_equal": int(sum(g[1] for g in gen)),
           "ms_per_token_check_mean": round(us_tok.mean() / 1000, 3),
           "ms_per_token_at_position": {str(p): round(float(np.mean(us_tok[[i for i, h in enumerate(hashes)
                                                                            if h[1] == p]])) / 1000, 3)
                                        for p in (0, 63, 127, 255, 511) if p < seq},
           "ms_per_token_story_mean": round(us_gen.mean() / 1000, 3),
           "tokens_per_second_story": round(1e6 / us_gen.mean(), 3),
           "ms_per_token_by_part": {k: round(v / max(1, len(hashes)) / 1000, 3) for k, v in prof.items()},
           "story": story}
    if args.repeat_probe:
        res["repeat_probe"] = {"mode": args.repeat_probe, "timing_only": True,
                               "note": "layers 2i and 2i + 1 on block 2i's weights; outputs differ from intsim.py",
                               "copy_ms_per_token": round(info.get("repeat_copy_us", 0) / max(1, len(hashes)) / 1000, 3)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    suffix = ("" if args.linear == "lut8" else "_" + args.linear) + (f"_repeat_{args.repeat_probe}"
                                                                      if args.repeat_probe else "") \
        + (f"_c{args.p_cut:g}" if args.p_cut is not None else "") + ("_prof" if args.fine_prof else "") \
        + (f"_al{args.autoload}" if args.autoload is not None else "") + ("kv" if args.autoload_kv else "")
    out = os.path.join(HERE, "results", f"chip_{args.board}_{os.path.basename(run)}{suffix}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "story"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
