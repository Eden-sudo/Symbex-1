---
name: symbex-sync
description: Keeps every SYMBEX-1 agent (Obed's, Eden's and Abraham's) on one roadmap, one shared ESP32-S3 board and one status thread. Use at the start and end of every SYMBEX-1 session, before picking work, before asking for a board run, and whenever a pull request or comment mentions you.
---

# SYMBEX-1 team sync

Three people work on SYMBEX-1, each with their own coding agent. This skill is the contract between those agents:
one roadmap, one board, one status thread, and pull-request mentions as the way to hand work to each other.

| Person | GitHub | Track |
|---|---|---|
| Obed | `Obed0101` | native 1-bit LLMs on the chip (`benchmarks/edge_llm_probe`) |
| Eden | `Eden-sudo` | convolutional networks (`conv1d/`, `conv2d/`), `lib_symbex` kernels; owns the repo, merges, hosts the board |
| Abraham | `abrahandejesusaparicio-code` | FOMO detection (`conv2d/fomo_voc_person`), the classic ESP32 without PSRAM |

Find out whose agent you are from `gh api user --jq .login` and work on that person's track unless they say otherwise.

## 1. Start of every session

1. `git fetch origin` and read `docs/roadmap.md` on `origin/master`.
2. Read the status thread: the open issue titled **Team sync** in `Eden-sudo/Symbex-1`.
   ```
   SYNC=$(gh issue list -R Eden-sudo/Symbex-1 --state open --json number,title --jq '.[] | select(.title == "Team sync") | .number')
   gh issue view "$SYNC" -R Eden-sudo/Symbex-1 --comments
   ```
   Read at least the last update of each person. If the issue does not exist, tell your human and do not create it
   yourself.
3. Look for work addressed to you: every pull request, issue or comment of the last 14 days that mentions your login.
   (`gh search --mentions` misses mentions made in comments, so this reads the comments directly; it works in bash and
   zsh, on Linux and macOS.)
   ```
   LOGIN=$(gh api user --jq .login)
   SINCE=$(date -u -d '14 days ago' +%FT%TZ 2>/dev/null || date -u -v-14d +%FT%TZ)
   for endpoint in issues issues/comments pulls/comments; do
     gh api --paginate "repos/Eden-sudo/Symbex-1/$endpoint?since=$SINCE&per_page=100" \
       --jq ".[] | select((.body // \"\") | test(\"@$LOGIN\\\\b\")) | .html_url"
   done
   ```
   Open each link and check whether it was already answered. A request that mentions you comes before new work of your
   own, unless your human says otherwise.
4. Before starting something, check that nobody else is doing it: the roadmap's owner column, the open pull requests
   (`gh pr list -R Eden-sudo/Symbex-1`) and the last status updates.

## 2. One board for everyone

From now on every chip measurement is made on **Eden's ESP32-S3-DevKitC-1 N16R8** (16 MB flash, 8 MB octal PSRAM),
which stays connected to Eden's PC. Numbers from different people are then comparable without asking which board or
which run made them.

- Code that runs on the chip must build for the S3: the `esp32-s3-devkitc-1` env in `Symbex1_PIO/platformio.ini`, or
  an env with the same board, memory type (`qio_opi`) and flash size in the benchmark's own `platformio.ini`.
  Run `pio run -e <env>` locally before asking for a board run; a request that does not build wastes the board.
- Only Eden's agent flashes the board. Everyone else asks for a run with a board request (below).
- Other chips (the classic ESP32, ESP32-CAM, Pico 2) are still welcome as extra results; say which board produced
  each number. The S3 result is the one the roadmap tracks.
- One job at a time. Flashing replaces whatever was on the board, so no request may assume an earlier firmware or
  file is still there.
- Do not change partitions, `sdkconfig` options or clock settings for one benchmark without saying so in the request;
  the next person's numbers depend on them.

### Board request format

Post it as a comment on the pull request whose branch holds the code:

```
@Eden-sudo board run
commit: <sha>  (pushed, the branch of this pull request)
dir: benchmarks/<benchmark>
command: pio run -e <env> -t upload -t monitor
capture: <which lines of the serial log matter, and when the run is finished>
pass if: <bit-exact checks or numbers expected; what counts as a failure>
return: the raw serial log as benchmarks/<benchmark>/results/<name>.log, committed to this branch
```

Keep "Allow edits by maintainers" on for the pull request so Eden's agent can push the log to its branch. Eden's agent
gets the code with `gh pr checkout <number>`, checks that the head is the requested commit (if not, it asks), runs it, and replies on the same pull request
with the commit that adds the log, the board, the env and anything
that went wrong (did not build, did not flash, failed a check). It reports a failure as a failure, with the log.

## 3. Handing work to each other

- **A mention is a request.** Mentioning `@Obed0101`, `@Eden-sudo` or `@abrahandejesusaparicio-code` in a pull request
  or its comments asks that person's agent to act on it. Say exactly what is asked, the commit it applies to and how to
  tell it is done. A mention without a request is just a mention; use the name without `@` when you only refer to
  someone.
- Answer on the same pull request, even if the answer is "not now" or "this is not my track".
- Today the agents find mentions with the searches in section 1. When MendCode v2 is released as open source, each of
  the three runs a MendCode pull-request babysitter that watches the repo and starts working when its person is
  mentioned; the request format stays the same.

## 4. Rules everyone already agreed on

These come from the team's Discord and the merged pull requests. They are not optional:

- **Reproducible scripts.** Every published number comes from a script in the repo that anyone can re-run. No tables or
  JSON written by hand.
- **Same architecture, independent INT8 baseline, measured on the chip, exact, several seeds, test set once.** The
  details are in `benchmarks/README.md` ("The rules every benchmark follows").
- **Choose by validation, report test once.** Checkpoints, thresholds and variants are chosen on a validation split
  taken from the training data.
- **Flash is the main configuration**, weights in RAM the alternative.
- **Precision per layer and per chip.** Binary does not always win; say where it loses.
- **No energy claims without a measurement.** No power meter reading, no mJ figure.
- Results in the repo are raw logs and the `.json` files the scripts write; generated models and firmware stay out
  (`.gitignore`).

## 5. End of every session

1. Post one comment on the **Team sync** issue, in this shape:
   ```
   ### <Name> - <YYYY-MM-DD>
   Done: <what changed, with pull requests and commits>
   Measured: <numbers, board, env, and where the log is; "nothing" if nothing was measured>
   Blocked: <what is missing, and who can unblock it (@mention only if you are asking them for something)>
   Next: <what this agent will do next>
   ```
   Report what was verified and what was not. A number that was not measured on the chip is labelled as such.
2. If the work changes the roadmap (something finished, a new item, a change of owner), update `docs/roadmap.md` in
   the same pull request as the work. Changes to another person's track need that person's agreement in the pull
   request.
3. A pull request that is ready also adds its section to `UPDATES.md` (one section per pull request, newest first).

## 6. Git

- Push branches to your own fork and open pull requests against `Eden-sudo/Symbex-1`, `master`. Eden merges.
- One branch per task; follow-ups go on the same branch while its pull request is open.
- Never force push a branch someone else has commented on. Stage files by name.
- Do not close, merge or delete another person's pull requests, issues or branches.
