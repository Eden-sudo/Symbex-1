"""
How current is every result file? For each JSON in benchmarks/*/results/ (or an experiments folder), the day it was
last committed and how many lib_symbex commits came after it: a SYMBEX number measured before a library change may no
longer be what the library gives today.

    python results_age.py                        # one line per benchmark folder
    python results_age.py --files                # one line per file
    python results_age.py --files --stale        # only files with library changes after them
    python results_age.py --match chip           # only files whose path matches a regular expression

A file that records `provenance` (tools/compiler_core/chip/provenance.py) is dated by its own run day and library
commit. Older files carry no date: they are dated by their last commit, which is when the number reached the
repository, not when it was measured (it may be some days older); a
rename (results/ reorganized into method folders) is followed back and does not count as a change. INT8 / TFLite, low-bit, random forest and
ultracompactBNN and earlier integer (int4_ / int8_) files do not run lib_symbex: their library column is "-".
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIB = "Symbex1_PIO/lib/lib_symbex"
NOT_SYMBEX = re.compile(r"(^|/)(tflite|lowbit|rf|ucbnn|int\d+)(/|_)|/tflite_[^/]*$")   # earlier/int4_pc.json: MAC


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout


def last_commits():
    """{file: (unix time, day)} of the last commit that changed the content of each JSON under benchmarks/. A pure
    rename (folders reorganized) is followed back to the older name, not counted as a change."""
    out, alias, when = {}, {}, None
    for line in git("log", "-M", "--format=@%ct %cs", "--name-status", "--", "benchmarks").splitlines():
        if line.startswith("@"):
            t, day = line[1:].split()
            when = (int(t), day)
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[-1].endswith(".json"):
            continue
        status, path = parts[0], parts[-1]
        current = alias.get(path, path)
        if status.startswith("R"):
            alias[parts[1]] = current
            if status == "R100":
                continue
        if status[0] in "AMR" and current not in out:
            out[current] = when
    return out


def lib_after(lib_times, t):
    return sum(1 for x in lib_times if x > t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", action="store_true", help="one line per file instead of one per folder")
    ap.add_argument("--stale", action="store_true", help="only SYMBEX files with lib_symbex commits after them")
    ap.add_argument("--match", help="only paths matching this regular expression")
    args = ap.parse_args()

    tracked = [f for f in git("ls-files", "benchmarks").splitlines()
               if f.endswith(".json") and "/results/" in f and (not args.match or re.search(args.match, f))]
    if not tracked:
        sys.exit("[X] no result files match")
    dates = last_commits()
    lib_times = [int(t) for t in git("log", "--format=%ct", "--", LIB).split()]

    rows = []
    for f in tracked:
        t, day = dates.get(f, (None, "uncommitted"))
        symbex = not NOT_SYMBEX.search(f)
        after, source = None, "commit"
        try:
            prov = json.load(open(os.path.join(REPO, f))).get("provenance")
        except (ValueError, AttributeError, OSError):
            prov = None
        if isinstance(prov, dict) and prov.get("date"):
            day, source = prov["date"], "run"
            if symbex and prov.get("lib_commit"):
                try:   # a commit lost in a squash merge or a shallow clone: fall back to the file's commit date
                    after = int(git("rev-list", "--count", f"{prov['lib_commit']}..HEAD", "--", LIB).strip())
                except subprocess.CalledProcessError:
                    after = None
        if symbex and after is None and t is not None:
            after = lib_after(lib_times, t)
        if args.stale and not after:
            continue
        rows.append((f, day, source, after if symbex else None))

    if args.files:
        print(f"{'day':<11} {'from':<7} {'lib commits after':>17}  file")
        for f, day, source, after in sorted(rows, key=lambda r: (r[1], r[0])):
            print(f"{day:<11} {source:<7} {'-' if after is None else after:>17}  {f}")
        return
    groups = {}
    for f, day, source, after in rows:
        key = f.split("/results/")[0]
        g = groups.setdefault(key, dict(n=0, days=[], stale=0, symbex=0))
        g["n"] += 1
        g["days"].append(day)
        g["symbex"] += after is not None
        g["stale"] += bool(after)
    print(f"{'folder':<46}{'files':>6}{'oldest':>12}{'newest':>12}{'SYMBEX files':>14}{'with lib changes after':>24}")
    for key, g in sorted(groups.items()):
        days = sorted(d for d in g["days"] if d != "uncommitted")
        print(f"{key:<46}{g['n']:>6}{days[0] if days else '-':>12}{days[-1] if days else '-':>12}"
              f"{g['symbex']:>14}{g['stale']:>24}")


if __name__ == "__main__":
    main()
