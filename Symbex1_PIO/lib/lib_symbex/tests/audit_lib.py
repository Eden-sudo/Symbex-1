"""
audit_lib.py -- health check of lib_symbex: what compiles, what is broken, what is untested.

    python audit_lib.py                    # from lib_symbex/tests/ (or pass --lib PATH)
    python audit_lib.py --lib ../ --extra ../../../src/main.cpp

Read-only: it never changes a file. For every .c file under src/ it reports, one block per file:
    compile     gcc -std=c11 -Wall -Wextra with each HAL configuration run_tests.py uses
                (generic | esp32s3 in C mode | the 8-bit paths, as on AVR): errors and warnings
    includes    every #include "..." that does not resolve to a file of the library
    spanish     identifiers and comments that still look Spanish (a word list; review by hand)
And for the whole library:
    declared but not defined    functions in include/ with no definition in src/
    defined but not declared    non-static functions in src/ missing from every header
    tests                       public functions that tests/run_tests.py never mentions
--extra adds other C/C++ files (an example main.cpp, for instance) to the include and Spanish checks.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HAL_CONFIGS = [
    ("generic", []),
    ("esp32s3 (C mode)", ["-DSYMBEX_HAL_TEST_NO_PIE", "-DCONFIG_IDF_TARGET_ESP32S3"]),
    ("8-bit paths (as on AVR)", ["-DSYMBEX_U8_BYTEWISE", "-DSYMBEX_POPCOUNT_FORCE_LUT8"]),
]
SPANISH = ["capa", "capas", "pesos", "umbral", "umbrales", "salida", "entrada", "bloque", "bloques", "preparar",
           "ventana", "escala", "escalas", "mapa", "mapas", "neurona", "neuronas", "canal", "canales", "fila", "filas",
           "columna", "tamano", "tamaño", "palabra", "palabras", "resultado", "imagen", "imagenes", "datos", "memoria",
           "clase", "clases", "binario", "binaria", "suma", "conteo", "cuenta", "espacial", "vector", "generico",
           "genérico", "sin", "para", "desde", "hasta", "donde", "cuando", "porque", "tambien", "también", "aca", "acá"]
SPANISH_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in SPANISH) + r")\b", re.IGNORECASE)
FUNC_DECL = re.compile(r"^\s*(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w\s\*]*?\b(symbex_\w+)\s*\([^;{]*\)\s*;", re.M)
FUNC_DEF = re.compile(r"^(?!\s*static)\s*[A-Za-z_][\w\s\*]*?\b(symbex_\w+)\s*\([^;{]*\)\s*\{", re.M)
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def files(root, exts):
    out = []
    for d, _, fs in os.walk(root):
        out += [os.path.join(d, f) for f in fs if f.endswith(exts)]
    return sorted(out)


def resolve(inc, src_file, include_dirs):
    cands = [os.path.join(os.path.dirname(src_file), inc)] + [os.path.join(d, inc) for d in include_dirs]
    return any(os.path.exists(c) for c in cands)


def strip_strings(code):
    return re.sub(r'"(\\.|[^"\\])*"', '""', code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
                    help="the lib_symbex folder (default: the parent of this script's folder)")
    ap.add_argument("--extra", nargs="*", default=[], help="other files to check (includes and Spanish only)")
    ap.add_argument("--cc", default="gcc")
    args = ap.parse_args()
    lib = os.path.abspath(args.lib)
    inc_dir, src_dir = os.path.join(lib, "include"), os.path.join(lib, "src")
    if not os.path.isdir(inc_dir) or not os.path.isdir(src_dir):
        sys.exit(f"[X] {lib} has no include/ and src/: pass --lib")
    if not shutil.which(args.cc):
        sys.exit(f"[X] {args.cc} not found")
    include_dirs = [inc_dir] + sorted({os.path.dirname(h) for h in files(inc_dir, (".h",))})
    c_files = files(src_dir, (".c",))
    headers = files(inc_dir, (".h",))
    print(f"[*] lib_symbex at {lib}: {len(c_files)} .c files, {len(headers)} headers\n")

    totals = dict(errors=0, warnings=0, includes=0, spanish=0)
    tmp = tempfile.mkdtemp()
    for path in c_files + [os.path.abspath(e) for e in args.extra]:
        rel = os.path.relpath(path, lib)
        code = open(path, encoding="utf-8", errors="replace").read()
        print(f"--- {rel} ({code.count(chr(10))} lines) ---")
        if path.endswith(".c") and path.startswith(src_dir):
            for name, flags in HAL_CONFIGS:
                cmd = [args.cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-c", path, "-o", os.path.join(tmp, "o.o"),
                       "-I", inc_dir, *flags]
                r = subprocess.run(cmd, capture_output=True, text=True)
                errs = [l for l in r.stderr.splitlines() if ": error:" in l or ": fatal error:" in l]
                warns = [l for l in r.stderr.splitlines() if ": warning:" in l]
                totals["errors"] += len(errs)
                totals["warnings"] += len(warns)
                state = "OK" if r.returncode == 0 and not warns else ("FAILS" if r.returncode else "warnings")
                print(f"    compile [{name:<24}] {state}" + (f"  ({len(errs)} errors, {len(warns)} warnings)" if errs or warns else ""))
                for l in (errs + warns)[:6]:
                    print("        " + l.split(lib + os.sep)[-1][:150])
        bad = [i for i in INCLUDE.findall(code) if not resolve(i, path, include_dirs)]
        totals["includes"] += len(bad)
        print(f"    includes : " + ("all resolve" if not bad else "BROKEN -> " + ", ".join(bad)))
        hits = {}
        for m in SPANISH_RE.finditer(strip_strings(code)):
            hits[m.group(1).lower()] = hits.get(m.group(1).lower(), 0) + 1
        n = sum(hits.values())
        totals["spanish"] += n
        print(f"    spanish  : " + ("none" if not n else f"{n} hits -> " + ", ".join(f"{w} x{c}" for w, c in sorted(hits.items(), key=lambda x: -x[1])[:8])))
        print()
    for h in headers:
        code = open(h, encoding="utf-8", errors="replace").read()
        bad = [i for i in INCLUDE.findall(code) if not resolve(i, h, include_dirs)]
        if bad:
            totals["includes"] += len(bad)
            print(f"--- {os.path.relpath(h, lib)}: BROKEN include -> {', '.join(bad)}\n")

    declared, defined = set(), set()
    for h in headers:
        declared |= set(FUNC_DECL.findall(open(h, encoding="utf-8", errors="replace").read()))
    for c in c_files:
        defined |= set(FUNC_DEF.findall(open(c, encoding="utf-8", errors="replace").read()))
    inline_in_headers = set()
    for h in headers:
        inline_in_headers |= set(re.findall(r"\b(symbex_\w+)\s*\([^;{]*\)\s*\{", open(h, encoding="utf-8", errors="replace").read()))
    no_def = sorted(declared - defined - inline_in_headers)
    no_decl = sorted(defined - declared)
    tests = os.path.join(lib, "tests", "run_tests.py")
    tested_src = open(tests, encoding="utf-8", errors="replace").read() if os.path.exists(tests) else ""
    untested = sorted(f for f in (declared | defined) if f not in tested_src)
    print("=== WHOLE LIBRARY ===")
    print(f"declared but not defined ({len(no_def)}): " + (", ".join(no_def) if no_def else "none"))
    print(f"defined but not declared ({len(no_decl)}): " + (", ".join(no_decl) if no_decl else "none"))
    print(f"never mentioned by tests/run_tests.py ({len(untested)} of {len(declared | defined)}):")
    by_module = {}
    for f in untested:
        by_module.setdefault(f.split("_")[1] if f.count("_") > 1 else "?", []).append(f)
    for mod, fs in sorted(by_module.items()):
        print(f"    {mod:<10} " + ", ".join(fs))
    print(f"\nTOTAL: {totals['errors']} compile errors, {totals['warnings']} warnings, {totals['includes']} broken includes, "
          f"{totals['spanish']} Spanish-looking words")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
