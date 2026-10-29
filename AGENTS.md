# SYMBEX-1 agent instructions

SYMBEX-1 compiles neural networks into fully binary networks for small microcontrollers (`lib_symbex`, the CDT in
`tools/cdt.py`, and the reproducible comparisons in `benchmarks/`). See `README.md`.

Three people work on this repo, each with their own agent. Every agent, whatever tool runs it (Claude Code, Codex,
MendCode), follows the team skill at the start and end of each session:

**[.claude/skills/symbex-sync/SKILL.md](.claude/skills/symbex-sync/SKILL.md)**: one roadmap (`docs/roadmap.md`), one
shared ESP32-S3 board on Eden's PC, one status thread (the **Team sync** issue) and pull-request mentions as requests.

Benchmark rules are in `benchmarks/README.md`; the per-pull-request log is `UPDATES.md`.
