"""
kernels.py -- kernel benchmark: which way of computing a layer is faster on THIS chip, for THIS layer size.

Every chip favors a different operation (the ESP32-S3 has a SIMD multiplier and no popcount; the ESP32 has neither),
and the answer also depends on the row width: a fixed cost per neuron (calls, SIMD setup) dominates short rows, and
the AND layout pads every row to a multiple of 4 words (16 bytes) -- twice the reads for a 2-word row. So instead of
one big layer, the benchmark measures several widths (32 ... 1024 inputs, 64 neurons) and every variant:
    binary layers   xnor, and_pad, and_nopad
    first layer     first_mask, first_planes (XNOR layout), first_and_pad, first_and_nopad
    reference       int8 (optional, with_int8: Espressif's ESP-NN)
with the weights in flash (as deployed; weights="ram" to copy them to RAM). For every variant it fits
    cycles per neuron = fixed + per_word * words read per row
and model_cost interpolates between the measured widths of the same regime (rows of a multiple of 4 words, where
the S3 uses PIE, apart from the rest), so the cost of ANY model is predicted from its real layer shapes (padding
included). All binary variants compute
the same dot products, and so do all first-layer ones: their checksums must match, otherwise nothing is trusted.

    run_kernels(board, port, lib_path, project_dir, ...)   flashes firmware/kernels.c, returns the profile
    model_cost(profile, shapes, arithmetic)                expected cycles of a model with "and" / "xnor" (the CDT uses it)
    choose_arithmetic(profile, shapes=None)                "and" or "xnor" for those shapes (default: a 512-input MLP)
    load_profile / save_profile                            chip profiles: tools/chip_profiles/<board>_<popcount>.json
Profiles written before this version (one 128 x 512 layer, cycles per weight) are still read.
"""
import json
import os
import tempfile

import numpy as np

from .core import BOARDS, flash, lib_sources, open_port, read_line, wait_ready, write_project

PROFILE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chip_profiles"))
CPU_MHZ = 240
WIDTHS = [32, 64, 96, 128, 256, 512, 1024]
N_OUT = 64
BINARY = ["xnor", "and_pad", "and_nopad"]
FIRST = ["first_mask", "first_planes", "first_and_pad", "first_and_nopad"]
# the variant each arithmetic uses in the exported headers today
EXPORTED = {"xnor": ("xnor", "first_mask"), "and": ("and_pad", "first_and_pad")}


def words_of(n_in):
    return (n_in + 31) // 32


def stride_of(n_in):
    return -(-words_of(n_in) // 4) * 4


def row_words(kernel, n_in):
    """words read per row by a kernel: padded kernels read the 16-byte-multiple stride."""
    return stride_of(n_in) if kernel.endswith("_pad") else words_of(n_in)


def profile_path(board, popcount, profile_dir=PROFILE_DIR):
    return os.path.join(profile_dir, f"{board}_{popcount}.json")


def load_profile(board, popcount, profile_dir=PROFILE_DIR):
    """The chip profile, or None if there is none -- or if the file is empty or damaged (then it is measured again,
    instead of failing with a JSON error)."""
    path = profile_path(board, popcount, profile_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"    [!] chip profile {path} is empty or damaged ({e}): it will be measured again")
        return None


def save_profile(profile, profile_dir=PROFILE_DIR):
    """Writes the profile atomically (a temporary file, then a rename): an interrupted run can no longer leave an
    empty profile behind."""
    os.makedirs(profile_dir, exist_ok=True)
    path = profile_path(profile["board"], profile["popcount"], profile_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=1, default=float)        # default: numpy numbers, if any slipped in
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------------------------------
def kernel_cycles(profile, kernel, n_in, n_out):
    """Predicted cycles of one layer computed with `kernel`. Interpolated on the number of inputs between the measured
    widths of the SAME regime: rows of a multiple of 4 words (where the S3 uses PIE) and the rest (C fallback) can
    differ several times, so one curve over both would be wrong. Outside the measured range, the line of that regime
    (cycles per neuron on words per row)."""
    words = row_words(kernel, n_in)
    aligned = words % 4 == 0
    same = [r for r in profile.get("measurements", [])
            if r["kernel"] == kernel and (row_words(kernel, r["n_in"]) % 4 == 0) == aligned]
    # interpolate on the number of INPUTS: padded kernels read the same row for 32 ... 128 inputs, but their cost
    # still grows with the real inputs (preparing and adding them), so the row width alone does not identify a point
    by_inputs = sorted({(r["n_in"], r["cycles_per_neuron"]) for r in same})
    if len(by_inputs) >= 2 and by_inputs[0][0] <= n_in <= by_inputs[-1][0]:
        return n_out * float(np.interp(n_in, [p[0] for p in by_inputs], [p[1] for p in by_inputs]))
    pts = sorted({(row_words(kernel, r["n_in"]), r["cycles_per_neuron"]) for r in same})
    if len(pts) >= 2:
        f = fit(pts)
    elif len(pts) == 1:
        f = dict(fixed=0.0, per_word=pts[0][1] / pts[0][0])
    else:
        f = profile["fit"][kernel]
    return n_out * (f["fixed"] + f["per_word"] * words)


def model_cost(profile, shapes, arithmetic, binary_kernel=None, first_kernel=None, input_bits=8):
    """Expected cycles of a whole model with `arithmetic` ("and" / "xnor"). shapes: [(n_in, n_out, first)].
    By default the kernels the exported headers use (EXPORTED); binary_kernel / first_kernel override them.
    input_bits < 8 ([data] input_bits): the XNOR export computes the first layer over k bit planes
    (symbex_dense_input_u8_planes_bits), estimated from the measured 8-plane kernel: its fixed cost per neuron plus k / 8
    of the rest; the AND export uses
    the whole uint8 inputs whatever k is."""
    if "fit" not in profile:                           # profile of the previous benchmark: cycles per weight
        k = {r["name"]: r["cycles_per_weight"] for r in profile["kernels"]}
        name = arithmetic.upper()
        return sum(n_in * n_out * (k[f"first layer {name}"] if first else k[f"binary {name}"])
                   for n_in, n_out, first in shapes)
    b, f = EXPORTED[arithmetic]
    k_planes = 8
    if arithmetic == "xnor" and input_bits < 8 and "first_planes" in profile["fit"]:
        f, k_planes = "first_planes", input_bits
    b, f = binary_kernel or b, first_kernel or f

    def first_cycles(n_in, n_out):
        c8 = kernel_cycles(profile, f, n_in, n_out)
        if k_planes == 8:
            return c8
        fixed = n_out * max(profile["fit"][f]["fixed"], 0.0)      # the per-neuron part does not shrink with k
        return fixed + (c8 - fixed) * k_planes / 8
    return sum(first_cycles(n_in, n_out) if first else kernel_cycles(profile, b, n_in, n_out)
               for n_in, n_out, first in shapes)


def choose_arithmetic(profile, shapes=None):
    """'and' or 'xnor': the cheaper one for `shapes` ([(n_in, n_out, first)]; default a 512 -> 256 -> 10 MLP)."""
    shapes = shapes or [(512, 256, True), (256, 10, False)]
    cost = {a: model_cost(profile, shapes, a) for a in ("and", "xnor")}
    return min(cost, key=cost.get)


# ---------------------------------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------------------------------
def write_data_header(path, n_out, widths, seed=12345):
    """kernels_data.h: random weights for every width, in both layouts (XNOR rows and 16-byte AND rows)."""
    rng = np.random.default_rng(seed)
    lines = ["/* generated by chip/kernels.py: synthetic weights for the kernel benchmark */", "#include <stdint.h>",
             f"#define N_OUT {n_out}", f"#define N_SHAPES {len(widths)}",
             f"#define MAX_WORDS {max(words_of(w) for w in widths)}", f"#define MAX_STRIDE {max(stride_of(w) for w in widths)}",
             "struct KShape { int n_in, words, stride; const uint32_t* wx; const uint32_t* wa; const int32_t* count; };"]
    table = []
    for s, n_in in enumerate(widths):
        words, stride = words_of(n_in), stride_of(n_in)
        bits = rng.integers(0, 2, (n_out, n_in))
        def pack(n_words):
            padded = np.zeros((n_out, n_words * 32), dtype=np.uint8)
            padded[:, :n_in] = bits
            return np.packbits(padded, axis=1, bitorder="little").view("<u4").reshape(n_out, n_words).ravel()
        for name, arr in ((f"WX_{s}", pack(words)), (f"WA_{s}", pack(stride))):
            body = ",".join(f"0x{int(v):08X}u" for v in arr)
            lines.append(f"static const uint32_t {name}[{len(arr)}] __attribute__((aligned(16))) = {{{body}}};")
        lines.append(f"static const int32_t CNT_{s}[{n_out}] = {{{','.join(str(int(c)) for c in bits.sum(1))}}};")
        table.append(f"{{{n_in}, {words}, {stride}, WX_{s}, WA_{s}, CNT_{s}}}")
    lines.append("static const struct KShape SHAPES[N_SHAPES] = {" + ", ".join(table) + "};")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fit(points):
    """least squares of cycles-per-neuron on words-per-row: (fixed, per_word)."""
    w = np.array([p[0] for p in points], dtype=float)
    c = np.array([p[1] for p in points], dtype=float)
    per_word, fixed = np.polyfit(w, c, 1) if len(set(w)) > 1 else (c.mean() / w.mean(), 0.0)
    return dict(fixed=float(fixed), per_word=float(per_word))


def run_kernels(board, port, lib_path, project_dir, popcount="lut16", n_out=N_OUT, n_in=None, baud=115200,
                upload=True, verbose=True, with_int8=False, widths=None, weights="flash"):
    """Flashes the kernel benchmark and returns the chip profile (also printed).
    widths: input sizes to measure (default WIDTHS; n_in, if given, is added). weights: "flash" or "ram".
    with_int8: also measure INT8 with ESP-NN as a reference."""
    if board not in BOARDS:
        raise ValueError(f"chip: board must be one of {sorted(BOARDS)}")
    widths = sorted(set((widths or WIDTHS) + ([n_in] if n_in else [])))
    data_h = os.path.join(tempfile.mkdtemp(prefix="symbex_kernels_"), "kernels_data.h")
    write_data_header(data_h, n_out, widths)
    defines = {}
    if with_int8:
        defines["KERNELS_WITH_INT8"] = 1
    if weights == "ram":
        defines["KERNELS_WEIGHTS_IN_RAM"] = 1
    write_project(project_dir, board, port, lib_sources(lib_path), "kernels.c", defines,
                  extra_files={"kernels_data.h": data_h},
                  build_flags=["-DSYMBEX_POPCOUNT_NO_LUT16"] if popcount == "swar" else [],
                  components=["espressif/esp-nn"] if with_int8 else (), lib_path=lib_path)
    if upload:
        flash(project_dir, verbose, f"the kernel benchmark on {BOARDS[board]} ({port}, popcount {popcount})")
    rows = []
    with open_port(port, baud) as s:
        ready = wait_ready(s, "KERNELS ready")
        hal = ready.split("|")[1].strip() if "|" in ready else "?"
        while True:
            line = read_line(s)
            if line == "DONE":
                break
            if line.startswith("K:"):
                name, width, ns, chk = line[2:].rsplit(",", 3)
                rows.append(dict(kernel=name, n_in=int(width), ns=int(ns), checksum=int(chk)))
                if verbose:
                    print(f"      measured: {name:<16} {width:>5} inputs")
    for r in rows:
        cyc = r["ns"] * CPU_MHZ / 1000
        r["cycles_per_neuron"] = cyc / n_out
        r["cycles_per_weight"] = cyc / (n_out * r["n_in"])
    ok = True
    for w in widths:
        for group in (BINARY, FIRST):
            chks = {r["checksum"] for r in rows if r["n_in"] == w and r["kernel"] in group}
            ok &= len(chks) == 1
    fits = {}
    for k in BINARY + FIRST + ["int8"]:
        pts = [(row_words(k, r["n_in"]), r["cycles_per_neuron"]) for r in rows if r["kernel"] == k]
        if pts:
            fits[k] = fit(pts)
    profile = dict(version=2, board=board, popcount=popcount, hal=hal, n_out=n_out, widths=widths, weights=weights,
                   cpu_mhz=CPU_MHZ, measurements=rows, fit=fits, checksums_ok=ok)
    if ok:
        profile["best_arithmetic"] = choose_arithmetic(profile)
    if verbose:
        print_profile(profile)
    return profile


def print_profile(profile):
    rows, widths = profile["measurements"], profile["widths"]
    print(f"\n    KERNELS ON THE CHIP  ({BOARDS[profile['board']]}, {profile['cpu_mhz']} MHz, one core, {profile['hal']})")
    print(f"    {profile['n_out']} neurons per layer, weights in {profile['weights']}, median of 15 runs\n")
    kernels = [k for k in BINARY + FIRST + ["int8"] if k in profile["fit"]]
    for k in kernels:
        f = profile["fit"][k]
        print(f"    {k}")
        print(f"      {'inputs':>7}{'words/row':>11}{'cycles/neuron':>15}{'cycles/weight':>15}")
        for w in widths:
            r = next((r for r in rows if r["kernel"] == k and r["n_in"] == w), None)
            if r:
                print(f"      {w:>7}{row_words(k, w):>11}{r['cycles_per_neuron']:>15.1f}{r['cycles_per_weight']:>15.3f}")
        print(f"      line over every width: {f['fixed']:.1f} cycles per neuron + {f['per_word']:.2f} per word of the row\n")
    if not profile["checksums_ok"]:
        print("    [X] the variants gave different results: the timings are not trusted")
        return
    print("    checksums: every variant agrees (binary and first layer, every width)")
    print(f"\n    FASTEST VARIANT PER WIDTH (cycles per neuron)")
    print(f"    {'inputs':>7}  {'binary':<26}{'first layer':<30}")
    for w in widths:
        b = min(BINARY, key=lambda k: kernel_cycles(profile, k, w, 1))
        fl = min(FIRST, key=lambda k: kernel_cycles(profile, k, w, 1))
        print(f"    {w:>7}  {b:<12}{kernel_cycles(profile, b, w, 1):>8.1f}      {fl:<16}{kernel_cycles(profile, fl, w, 1):>8.1f}")
    print(f"\n    -> for a 512 -> 256 -> 10 MLP the exported arithmetic would be {profile['best_arithmetic'].upper()}; "
          f"with [output] arithmetic = \"auto\" the CDT decides from the real shapes of each model")
