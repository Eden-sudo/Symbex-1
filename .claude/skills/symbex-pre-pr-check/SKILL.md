---
name: symbex-pre-pr-check
description: Checks a SYMBEX-1 branch before a pull request is opened or updated, against what the team's reviewers keep rejecting. Use before every `gh pr create`, before pushing new commits to an open PR, and when asked whether a branch is ready.
---

# SYMBEX-1 pre-PR check

Run it on your own branch, read-only, and report each point as passed, failed or not applicable. Fix only what your
human asked you to fix. Context first: [AGENTS.md](../../../AGENTS.md), [docs/roadmap.md](../../../docs/roadmap.md),
[.agents/memory/conventions.md](../../../.agents/memory/conventions.md).

## 1. Scope

```sh
git fetch origin master
git diff --stat origin/master...HEAD
git diff --check origin/master...HEAD
```

- Only files this PR is about; no measurement scripts that were not run, no files from other sessions (#22).
- Nothing that `.gitignore` keeps out: `runs/`, `models/`, datasets, `*.pth`, `*.tflite`, `*.u8`.
- Result data above ~4 MiB of diff goes to its own data PR (#23 -> #24).
- The title says everything the PR changes, fixes included (#19).

## 2. Figures

For every number added or changed in a README, `docs/` or the PR body:

- It comes from a committed `results/**/*.json` or log in this branch or `master` (#8, #11, #15). Name the file.
- Accuracy comes from validation; the test set appears once, on the final models (`docs/methodology.md`, "Data").
- Three seeds, mean +- standard deviation; a single seed is called a ranking.
- Anything not measured is called an estimate and says where it comes from.

## 3. Documentation pairs

- `README.md` and `README.es.md` say the same; each changed `docs/<name>.md` has its `docs/<name>.es.md` updated.
- `UPDATES.md` has this PR's section (`## PR #N: title (Month YYYY)`, newest first).
- `docs/roadmap.md` moves the items this PR changes (done / open / next / proposed).
- If you learned a durable fact, update the matching file in `.agents/memory/`.

## 4. Tests

Run what the change touches; report each one as run, not run (and why), or needs a board.

- Python: `python -m pytest -q` from the repository root.
- C engine: `python3 Symbex1_PIO/lib/lib_symbex/tests/run_tests.py` (`--only <regex>` to narrow, `--stress` for the
  sanitizers). Both need gcc and numpy; benchmarks use `benchmarks/env_bench`.
- On a board: [symbex-exactness-check](../symbex-exactness-check/SKILL.md). A shared S3 run needs a board request
  ([symbex-sync](../symbex-sync/SKILL.md), section 2).

## 5. Report

A short list: each point, its result and the evidence (command output, file). Do not open the PR yourself unless
your human asked you to.
