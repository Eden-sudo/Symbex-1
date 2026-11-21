"""
cache_methods_table.py -- one table of the 260K's KV cache methods: accuracy, cache RAM, speed and flash cost.

Gathers what steps 6-9b measured (README "Step 6" / "Step 9" / "Step 9b", "Speed of each cache method") and adds
two things computed here:
    KV RAM      bytes of the cache at ctx 512, 20 key / value heads of width 8 (5 layers x 4), c/tinylm.c's layout:
                int8 8 bytes + int32 scale per position; 1 bit 1 + 4; unit 1 bit 1; 2 bits 2 + 4; window slot 8 + 4
    flash       model data beyond model.bin (temperatures int32 per head, center vectors: no format yet, counted as
                int32, the 24-byte low-bit trailer of tinylm.h), and the CODE of each method: the attention functions
                of ../chip_kernels/kv260k_bench.py compiled -O2 for the ESP32 (Xtensa), the M33 and Hazard3
                (PlatformIO toolchains), the sizes of the functions each method calls (nm -S), so a method's code
                cost is measured, not guessed. Small numbers are reported as they are, never rounded to zero.
Speed: ms per token of attention at a full context (512 positions), from results/kv260k_bench_<board>.json.
Loss: validation, 3 seeds (None = not trained yet). Methods without a timed kernel have speed None.

    python cache_methods_table.py          # -> results/v512d64l5kv4_cache_methods.json and a Markdown table
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERN = os.path.join(HERE, "..", "chip_kernels")
sys.path.insert(0, KERN)
import kv260k_bench as kb  # noqa: E402

PIO = os.path.expanduser("~/.platformio/packages")
TOOLS = {"esp32": ("toolchain-xtensa-esp32/bin/xtensa-esp32-elf-", ["-mlongcalls"]),
         "pico2_arm": ("toolchain-gccarmnoneeabi/bin/arm-none-eabi-", ["-mcpu=cortex-m33", "-mthumb"]),
         "pico2_riscv": ("toolchain-rp2040-earlephilhower/bin/riscv32-unknown-elf-",
                         ["-march=rv32imac_zba_zbb_zbs", "-mabi=ilp32"])}
# the functions each stage calls (kv260k_bench.py FIRMWARE); values_rows is inlined or a constprop clone
CALLS = {"s_i8": ["s_i8"], "s_k1": ["s_k1", "k1_rng", "make_lut"], "s_k1_win": ["s_k1_win", "k1_rng", "make_lut",
         "win_tail"], "s_k1u_win": ["s_k1u_win", "make_lut", "win_tail"], "s_k1c": ["s_k1c", "make_lut", "center_ab"],
         "s_k1uc_win": ["s_k1uc_win", "make_lut", "center_ab", "win_tail"], "sm_base": ["sm_base", "umax_of"],
         "sm_temp": ["sm_temp", "umax_of"], "v_i8": ["v_i8", "values_rows"], "v2_win": ["v2_win", "values_rows"]}

L, KVH, HD, SEQ, WIN, HEADS = 5, 4, 8, 512, 32, 8
NH = L * KVH
TEMP_B, CENTER_B, TRAILER_B = 4 * L * HEADS, 4 * NH * HD, 24


def kv_ram(k, v, window):
    """bytes: k / v in 'i8' (8 + 4), '1' (1 + 4), 'u' (1), '2' (2 + 4) per head and position; window slots 8 + 4
    for the keys, and for the values when they are below 8 bits"""
    per = {"i8": HD + 4, "1": HD // 8 + 4, "u": HD // 8, "2": HD // 4 + 4}
    b = SEQ * NH * (per[k] + per[v])
    if window:
        b += WIN * NH * (HD + 4) * (1 + (v != "i8"))
    return b


# name, trained run (loss source), key, value, window, temp, center, ops (scores, softmax, values)
METHODS = [
    ("int8 (base)", ("base", None), "i8", "i8", 0, 0, 0, ("s_i8", "sm_base", "v_i8")),
    ("+ temperature", ("temp", None), "i8", "i8", 0, 1, 0, ("s_i8", "sm_temp", "v_i8")),
    ("k1", ("k1", "k1"), "1", "i8", 0, 0, 0, ("s_k1", "sm_base", "v_i8")),
    ("k1c", ("k1", "k1c"), "1", "i8", 0, 0, 1, ("s_k1c", "sm_base", "v_i8")),
    ("k1tr", ("k1", "k1tr"), "1", "i8", 1, 1, 0, ("s_k1_win", "sm_temp", "v_i8")),
    ("k1ctr", ("k1", "k1ctr"), "1", "i8", 1, 1, 1, None),
    ("k1uctr", ("k1", "k1uctr"), "u", "i8", 1, 1, 1, ("s_k1uc_win", "sm_temp", "v_i8")),
    ("k1u + window + temp (no center)", (None, None), "u", "i8", 1, 1, 0, ("s_k1u_win", "sm_temp", "v_i8")),
    ("k2c (step 7)", ("k2c", None), "2", "i8", 0, 0, 1, None),
    ("k1trv2 (step 9b)", ("v2", "k1trv2"), "1", "2", 1, 1, 0, ("s_k1_win", "sm_temp", "v2_win")),
    ("k1uctrv2 (step 9b)", ("v2", "k1uctrv2"), "u", "2", 1, 1, 1, ("s_k1uc_win", "sm_temp", "v2_win")),
]


def losses():
    r = os.path.join(HERE, "results")
    k1 = json.load(open(os.path.join(r, "v512d64l5kv4_k1_train.json")))
    t = json.load(open(os.path.join(r, "v512d64l5kv4_temp.json")))["runs"]["binary"]
    v2p = os.path.join(r, "v512d64l5kv4_v2_train.json")
    v2 = json.load(open(v2p))["runs"] if os.path.exists(v2p) else {}
    out = {"base": k1["baseline_int8_cache"], "temp": {"mean": t["mean"], "sd": t["sd"]}, "k2c": k1["k2c_step7"]}
    return out, k1["runs"], v2


def code_sizes():
    """bytes of each function of the bench firmware per toolchain (nm -S on the -O2 object)"""
    sizes = {}
    with tempfile.TemporaryDirectory() as d:
        inc = os.path.join(d, "inc", "freertos")
        os.makedirs(inc)
        open(os.path.join(inc, "FreeRTOS.h"), "w").write("#include <stdlib.h>\n#define pdMS_TO_TICKS(x) (x)\n")
        open(os.path.join(inc, "task.h"), "w").write("static inline void vTaskDelay(int x){(void)x; exit(0);}\n")
        open(os.path.join(d, "inc", "esp_cpu.h"), "w").write(
            "static inline unsigned esp_cpu_get_cycle_count(void){extern volatile unsigned cyc; return cyc;}\n")
        kb.write_data(os.path.join(d, "bench_data.h"), "size")
        open(os.path.join(d, "main.c"), "w").write(kb.FIRMWARE)
        for board, (pre, flags) in TOOLS.items():
            gcc = os.path.join(PIO, pre + "gcc")
            if not os.path.exists(gcc):
                print(f"[!] {board}: no toolchain at {gcc}, code size skipped")
                continue
            obj = os.path.join(d, board + ".o")
            subprocess.run([gcc, "-O2", "-ffunction-sections", "-DESP_PLATFORM", "-I" + os.path.join(d, "inc"),
                            "-I" + d, *flags, "-c", os.path.join(d, "main.c"), "-o", obj], check=True)
            nm = subprocess.run([os.path.join(PIO, pre + "nm"), "-S", obj], check=True, capture_output=True,
                                text=True).stdout
            fs = {}
            for line in nm.splitlines():
                p = line.split()
                if len(p) == 4 and p[2] in "tT":
                    fs[p[3].split(".")[0].split("$")[0]] = int(p[1], 16)
            sizes[board] = fs
    return sizes


def method_code(fs, ops):
    names = sorted({f for op in ops for f in CALLS[op]})
    return sum(fs.get(f, 0) for f in names)


def main():
    loss_fixed, k1runs, v2runs = losses()
    sizes = code_sizes()
    speed = {}
    for board in TOOLS:
        p = os.path.join(KERN, "results", f"kv260k_bench_{board}.json")
        if os.path.exists(p):
            speed[board] = {r["op"]: r["us"] for r in json.load(open(p))["rows"]}
    base_ops = METHODS[0][7]
    rows = []
    for name, (src, run), k, v, win, temp, center, ops in METHODS:
        if src in ("k1",):
            lo = k1runs.get(run)
        elif src == "v2":
            lo = v2runs.get(run)
        elif src:
            lo = loss_fixed[src]
        else:
            lo = None
        row = dict(method=name, key=k, value=v, window=WIN if win else 0, temperature=bool(temp), center=bool(center),
                   val_loss=None if lo is None else round(lo["mean"], 4), val_sd=None if lo is None else lo["sd"],
                   vs_int8_cache=None if lo is None else round(lo["mean"] - loss_fixed["base"]["mean"], 4),
                   kv_ram_bytes=kv_ram(k, v, win),
                   flash_data_bytes=TEMP_B * temp + CENTER_B * center + TRAILER_B * (k != "i8" or v != "i8"),
                   ops=list(ops) if ops else None, ms_per_token={}, code_bytes={}, code_vs_base_bytes={})
        for board in TOOLS:
            if ops and board in speed:
                row["ms_per_token"][board] = round(sum(speed[board][o] for o in ops) * HEADS * L / 1000, 2)
            if ops and board in sizes:
                c = method_code(sizes[board], ops)
                row["code_bytes"][board] = c
                row["code_vs_base_bytes"][board] = c - method_code(sizes[board], base_ops)
        rows.append(row)
    out = dict(what="260K KV cache methods: validation loss (3 seeds), KV RAM (computed, ctx 512), attention speed "
                    "(kv260k_bench, full context, ms per token), flash: model data (computed) + attention code "
                    "(measured, nm -S of kv260k_bench's functions at -O2)",
               model_bin_bytes=os.path.getsize(os.path.join(HERE, "runs", "binary_v512d64l5kv4_s42", "model.bin"))
               if os.path.exists(os.path.join(HERE, "runs", "binary_v512d64l5kv4_s42", "model.bin")) else None,
               center_format="not defined yet (no integer model): counted as int32", rows=rows)
    path = os.path.join(HERE, "results", "v512d64l5kv4_cache_methods.json")
    json.dump(out, open(path, "w"), indent=1)
    print("| method | loss (vs int8 cache) | KV RAM KB | ms/token ESP32 / M33 / Hazard3 | flash data B | code B "
          "vs base ESP32 / M33 / Hazard3 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        lo = "pending" if r["val_loss"] is None else f"{r['val_loss']:.4f} ({r['vs_int8_cache']:+.3f})"
        ms = " / ".join(str(r["ms_per_token"].get(b, "—")) for b in TOOLS) if r["ms_per_token"] else "not timed"
        cd = " / ".join(f"{r['code_vs_base_bytes'][b]:+d}" for b in TOOLS if b in r["code_vs_base_bytes"]) \
            if r["code_vs_base_bytes"] else "not timed"
        print(f"| {r['method']} | {lo} | {r['kv_ram_bytes'] / 1000:.1f} | {ms} | {r['flash_data_bytes']} | {cd} |")
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
