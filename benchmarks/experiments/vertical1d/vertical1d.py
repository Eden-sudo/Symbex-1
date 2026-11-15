"""
vertical1d.py -- EXPERIMENT: the vertical (bit-sliced) engine for the binary 1D convolutions (vertical1d.h), tried on
the 1-bit networks the CDT already exported (HAR, ECG), before anything goes into lib_symbex.

    python vertical1d.py test                         # v1d_conv vs symbex_bool1d_conv on random layers (PC, sanitizers)
    python vertical1d.py make ../../conv1d/har_ultracompact/runs/max_int8_e100_s42 [--layers 1,2]
            # -> runs/<name>/: the same model_symbex.h with the chosen layers on v1d_conv, checked on the PC
    (then chip_test.py on runs/<name>, the flags of the source run's chip runs, e.g. --board s3 --popcount swar)
    python vertical1d.py report                       # per step: standard (source run) vs vertical, from the chip JSONs

Run from benchmarks/experiments/vertical1d/ (main environment). `make` takes a run folder with the CDT's model_symbex.h
(compiler_core/export/bool1d.py) and its validation files; it embeds vertical1d.h into a copy of the header, swaps
symbex_bool1d_conv for v1d_conv on the binary layers (default: every one with stride 1 and kernel x in_ch <= 1016;
--layers picks them), adds the work block (counted in SYMBEX_MLP_BYTES_BUFFERS), and checks the C against the source
run's integer simulation (validation_sim.u8, every window). The first layer (uint8 input) stays as it is.
Results: results/vertical1d.json (report).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
LIB = os.path.join(ROOT, "Symbex1_PIO", "lib", "lib_symbex")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.export import c_build                                                      # noqa: E402

RUNS = os.path.join(HERE, "runs")
SOURCES = ["layers/conv/symbex_bool1d.c", "layers/conv/symbex_bool.c", "hal/generic/symbex_hal_impl.c"]
FIELDS = ("in_ch", "out_ch", "kernel", "stride", "pool", "in_len", "out_len", "row_words", "weights", "thresholds",
          "weight_counts", "weights_i8")


def layers_of(header):
    """{index: dict of the SymbexBool1D fields} from the header (positional or designated initializers)"""
    out = {}
    for m in re.finditer(r"static const struct SymbexBool1D L(\d+) = \{([^}]*)\};", header):
        parts = [p.strip() for p in m.group(2).split(",")]
        if parts[0].startswith("."):
            d = {p.split("=")[0].strip(" ."): p.split("=")[1].strip() for p in parts}
        else:
            d = dict(zip(FIELDS, parts))
        out[int(m.group(1))] = {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in d.items()}
    return out


def v1d_words(L):
    """V1D_WORDS of vertical1d.h, in Python (checked against the C by a _Static_assert in the header)"""
    tiles = (L["out_len"] * L["pool"] + 127) // 128
    n = L["kernel"] * L["in_ch"]
    k8 = (n + 7) // 8 * 8 + 8
    pw = (((L["in_len"] + tiles * 128 + L["kernel"] + 31) >> 5) + 2 + 3) & ~3
    return (1280 + 4 + L["in_ch"] * pw + k8 * 4 + 2 * (k8 // 8 + 8) * 4 + (16 + 16 + 12 + 12 + 1) * 4
            + L["out_ch"] * tiles * 4 + ((k8 + 16) // 2 + 3) // 4 * 4)


def eligible(L):
    return L["weights"] != "NULL" and L["weights_i8"] == "NULL" and L["stride"] == 1 and L["kernel"] * L["in_ch"] <= 1016


def make(args):
    src = os.path.normpath(args.run)
    header = open(os.path.join(src, "model_symbex.h")).read()
    layers = layers_of(header)
    calls = {int(m.group(1)): m.group(0) for m in
             re.finditer(r"symbex_bool1d_conv\(&L(\d+), (\w+), (\w+), B_WINDOW, B_SCRATCH\);", header)}
    chosen = [int(i) for i in args.layers.split(",")] if args.layers else [i for i in sorted(calls) if eligible(layers[i])]
    for i in chosen:
        if i not in calls or not eligible(layers[i]):
            sys.exit(f"[X] layer {i} of {src} is not a binary 1D conv the vertical engine runs")
    if not chosen:
        sys.exit(f"[X] {src}: no layer for the vertical engine")
    name = os.path.basename(src) + "_v" + "".join(str(i) for i in chosen)
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    kernel = open(os.path.join(HERE, "vertical1d.h")).read()
    inc = '#include "layers/conv/symbex_bool1d.h"\n'
    assert header.count(inc) == 1
    header = header.replace(inc, inc + "/* ---- vertical1d.h (embedded: benchmarks/experiments/vertical1d) ---- */\n"
                            + kernel + "/* ---- end of vertical1d.h ---- */\n", 1)
    sizes = [f"V1D_WORDS({L['in_ch']}, {L['out_ch']}, {L['kernel']}, {L['pool']}, {L['in_len']}, {L['out_len']})"
             for L in (layers[i] for i in chosen)]
    expr = sizes[0]
    for s in sizes[1:]:
        expr = f"V1D_MAX({expr}, {s})"
    words = max(v1d_words(layers[i]) for i in chosen)
    anchor = "#define SYMBEX_MLP_N_INPUTS"
    assert header.count(anchor) == 1
    header = header.replace(anchor, f"static uint32_t V_WORK[{expr}] __attribute__((aligned(16)));   /* the vertical "
                            f"layers' work block */\n#ifdef __cplusplus\nstatic_assert(sizeof(V_WORK) == {4 * words}u, \"V1D_WORDS\");\n"
                            f"#else\n_Static_assert(sizeof(V_WORK) == {4 * words}u, \"V1D_WORDS\");\n#endif\n"
                            + anchor, 1)
    m = re.search(r"#define SYMBEX_MLP_BYTES_BUFFERS (\d+)u", header)
    header = header.replace(m.group(0), f"#define SYMBEX_MLP_BYTES_BUFFERS {int(m.group(1)) + 4 * words}u")
    for i in chosen:
        c = calls[i]
        header = header.replace(c, re.sub(r"symbex_bool1d_conv\((&L\d+, \w+, \w+), B_WINDOW, B_SCRATCH\)",
                                          r"v1d_conv(\1, V_WORK)", c), 1)
    if "B_WINDOW," not in header:                                         # every bool1d layer is vertical now
        w = re.search(r"static uint32_t B_WINDOW\[", header)
        header = header[:w.start()] + "__attribute__((unused)) " + header[w.start():]
    path = os.path.join(out, "model_symbex.h")
    open(path, "w").write(header)
    for f in ("validation_x.u8", "validation_y.u8", "validation_sim.u8"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    n_in = int(re.search(r"#define SYMBEX_MLP_N_INPUTS (\d+)", header).group(1))
    x = np.fromfile(os.path.join(out, "validation_x.u8"), dtype=np.uint8).reshape(-1, n_in)
    sim = np.fromfile(os.path.join(out, "validation_sim.u8"), dtype=np.uint8).astype(np.int64)
    equal, total, msg = c_build.check(path, LIB, x, sim, sources=SOURCES, flags=["-Wall", "-Werror"])
    ex = json.load(open(os.path.join(src, "export.json"))) if os.path.exists(os.path.join(src, "export.json")) else {}
    ex.update(source_run=os.path.relpath(src, ROOT), vertical_layers=chosen, vertical_work_bytes=4 * words,
              exactness=None if equal is None else 100 * equal / total, header=os.path.relpath(path, ROOT))
    json.dump(ex, open(os.path.join(out, "export.json"), "w"), indent=1)
    print(f"    {name}: layers {chosen} vertical, work block {4 * words / 1024:.1f} KB, exact on the PC "
          + (f"{100 * equal / total:.2f}% of {total}" if equal is not None else f"not checked ({msg})") + f" -> {out}")
    if equal is None and not args.allow_unchecked:     # no compiler, the build failed or the program crashed
        sys.exit("[X] the C check did not run (--allow-unchecked to accept that)")
    if equal is not None and equal != total:
        sys.exit("[X] not exact")


def test(args):
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "t")
        cmd = ["gcc", "-O2", "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
               "-I", os.path.join(LIB, "include"), "-I", HERE, "-o", exe, os.path.join(HERE, "test_vertical1d.c")]
        cmd += [os.path.join(LIB, "src", s) for s in SOURCES]
        subprocess.run(cmd, check=True)
        r = subprocess.run([exe, str(args.cases)], capture_output=True, text=True)
        print(r.stdout.strip())
        sys.exit(r.returncode)


def report(args):
    rows = []
    for folder in sorted(glob.glob(os.path.join(RUNS, "*"))):
        ex = json.load(open(os.path.join(folder, "export.json")))
        for path in sorted(glob.glob(os.path.join(folder, "chip_*.json"))):
            base = os.path.join(ROOT, ex["source_run"], os.path.basename(path))
            if not os.path.exists(base):
                continue
            v, s = json.load(open(path)), json.load(open(base))
            row = dict(run=os.path.basename(folder), source=ex["source_run"], chip=os.path.basename(path)[5:-5],
                       vertical_layers=ex["vertical_layers"], standard_ms=s["median_ms"], vertical_ms=v["median_ms"],
                       speedup=s["median_ms"] / v["median_ms"],
                       standard_steps=[t["median_ms"] for t in s["layers"]],
                       vertical_steps=[t["median_ms"] for t in v["layers"]],
                       equal_simulation=v["equal_simulation"], ram_vertical=v.get("ram"), ram_standard=s.get("ram"),
                       work_bytes=ex["vertical_work_bytes"])
            rows.append(row)
            print(f"    {row['run']:<40} {row['chip']:<22} standard {row['standard_ms']:7.3f} ms  vertical "
                  f"{row['vertical_ms']:7.3f} ms  {row['speedup']:.2f}x  steps "
                  + " / ".join(f"{a:.3f}->{b:.3f}" for a, b in zip(row["standard_steps"], row["vertical_steps"]))
                  + f"  = simulation {row['equal_simulation']:.1f}%")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "vertical1d.json")
    json.dump(dict(note="standard = the source run's chip JSON, vertical = the same network with the listed layers on "
                        "v1d_conv (vertical1d.h); times in ms, median over the windows", rows=rows), open(out, "w"), indent=1)
    print(f"[*] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["test", "make", "report"])
    ap.add_argument("run", nargs="?", help="make: a CDT bool1d run folder")
    ap.add_argument("--layers", help="make: the layer indices to run vertically, e.g. 1,2 (default: every eligible one)")
    ap.add_argument("--allow-unchecked", action="store_true",
                    help="make: exit 0 even when the C check could not run (no compiler, build failed, crash)")
    ap.add_argument("--cases", type=int, default=5000, help="test: random layers")
    args = ap.parse_args()
    {"test": test, "make": make, "report": report}[args.what](args)


if __name__ == "__main__":
    main()
