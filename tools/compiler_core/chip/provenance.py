"""
Where a measurement came from, written into every result file so a table can say how current it is.

    provenance(**extra) -> dict(date, commit, lib_commit, lib_dirty, **extra)

date        the day of the run (ISO, local time)
commit      the repository's HEAD when it ran
lib_commit  the last commit that changed lib_symbex (the engine the chip ran)
lib_dirty   True when lib_symbex had uncommitted changes: the commit alone does not describe that build

Each field is None when git is not available (a copy of the repository without .git).
"""
import datetime
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LIB = os.path.join("Symbex1_PIO", "lib", "lib_symbex")


def _git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(**extra):
    dirty = _git("status", "--porcelain", "--", LIB)
    return dict(date=datetime.date.today().isoformat(),
                commit=_git("rev-parse", "--short=12", "HEAD"),
                lib_commit=_git("log", "-1", "--format=%h", "--abbrev=12", "--", LIB),
                lib_dirty=None if dirty is None else bool(dirty),
                **extra)
