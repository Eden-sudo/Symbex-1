"""
c_build.py -- the exactness check on this PC, shared by every exporter: build an exported model_symbex.h with
lib_symbex (the C compiler: $CC, else gcc, else cc), run it on the validation inputs and count how many results equal the integer simulation.

    check(header_path, lib_path, x_u8, expected, sources, ...)   -> (equal, total, message)
    TEST_MAIN                                                     the test program (reads uint8 inputs, prints the class)

Every header offers the interface of the MLP headers (symbex_mlp_infer, SYMBEX_MLP_N_INPUTS), so one test program
serves them all. Each exporter's validate_c names its library sources and flags and calls check().
"""
import os
import shlex
import shutil
import subprocess
import tempfile

import numpy as np

TEST_MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "model_symbex.h"
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    FILE* f = fopen(argv[1], "rb");
    int n = atoi(argv[2]);
    static uint8_t x[SYMBEX_MLP_N_INPUTS];
    for (int i = 0; i < n; i++) {
        if (fread(x, 1, SYMBEX_MLP_N_INPUTS, f) != SYMBEX_MLP_N_INPUTS) return 1;
        printf("%d\n", symbex_mlp_infer(x));
    }
    fclose(f);
    return 0;
}
"""


def _compiler():
    """(argv prefix, "") for the C compiler: $CC split like a shell (so 'ccache gcc' works), else gcc, else cc;
    (None, message) when it is not there."""
    cc = os.environ.get("CC", "")
    try:
        cmd = shlex.split(cc)
    except ValueError as e:
        return None, f"CC={cc!r} could not be split ({e}): the C exactness check was skipped"
    if cmd:
        if shutil.which(cmd[0]) is None:
            return None, f"{cmd[0]} (from CC) not found: the C exactness check was skipped"
        return cmd, ""
    for name in ("gcc", "cc"):
        if shutil.which(name) is not None:
            return [name], ""
    return None, "neither gcc nor cc found: the C exactness check was skipped"


def check(header_path, lib_path, x_u8, expected, sources, flags=(), std="gnu11", after=None):
    """the C compiler on this PC ($CC, else gcc, else cc): the header + lib_symbex, run on x_u8; returns
    (equal, total, message), equal None when it could not run (no compiler, library missing, no build, a crash;
    message says why).
    sources: library files, relative to <lib_path>/src (an absolute path is taken as it is: an application's own C).
    flags: extra compiler flags (-D..., -mpopcnt). std: the C standard.
    after(build, folder): an extra check run in the build folder after the results; build(c_file, exe) builds a
    program of that folder with the same sources and flags (None on success, else the compiler's error). It returns
    "" when the check passes, else the message (the result is then None)."""
    cc, why = _compiler()
    if cc is None:
        return None, 0, why
    inc = os.path.join(lib_path, "include")
    sources = [os.path.join(lib_path, "src", s) for s in sources]
    for r in [inc] + sources:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing): set [output] lib_path"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header_path, os.path.join(d, "model_symbex.h"))

        def build(c_file, exe):
            r = subprocess.run([*cc, "-O2", f"-std={std}", *flags, "-I", inc, "-I", d, "-o", exe, c_file] + sources,
                               capture_output=True, text=True)
            return None if r.returncode == 0 else r.stderr

        open(os.path.join(d, "main.c"), "w").write(TEST_MAIN)
        exe = os.path.join(d, "check")
        err = build(os.path.join(d, "main.c"), exe)
        if err is not None:
            return None, 0, "the C did not compile:\n" + err[-2000:]
        data = os.path.join(d, "x.u8")
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(data)
        r = subprocess.run([exe, data, str(len(x_u8))], capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C program failed while running"
        pred_c = np.array([int(v) for v in r.stdout.split()])
        if after is not None:
            msg = after(lambda c_file, out: build(os.path.join(d, c_file), os.path.join(d, out)), d)
            if msg:
                return None, 0, msg
    return int((pred_c == expected).sum()), len(expected), ""
