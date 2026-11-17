# damage_probe -- logbook

Every entry: date, question, what was run, numbers, errors, decision. Newest last.

### 2026-11-16: pre-registration
- Question, pairs, the four measures and the rule are in README.md, written before any measurement. Branch
  exp/damage-probe (folder ~/Proyectos/Symbex1_damage, the main folder was busy with merges).
- Ground truth = the tiny_llm runs already trained (runs/ in the main folder, gitignored): no new full training.
- Edén's classic ESP32 checked the same day: no PSRAM (probe firmware: "PSRAM ID read error"), 4 MB flash, 346 KB
  free heap. Not used here; it matters for the on-chip LLM design that this tool should serve.

### 2026-11-16/17: full run (damage_probe.py, 34 rows, ~2.9 h on the GPU, results/damage_probe.json)
- No errors. Smoke test before (3 pairs, 20 fine-tune steps, /tmp, not kept).
- Spearman per seed 42 / 43 / 44: cos 0.670 / 0.867 / 0.818, loss0 0.618 / 0.927 / 0.952, fisher 0.007 / 0.200 /
  0.248, recover 0.815 / 0.988 / 0.976.
- Rule: recover passes the correlation, FAILS the cost (210-226 s per pair + a control per model A vs 136-201 s =
  10 % of a full run). Nothing passes. Written as is in README.
- My expectation (written before) was right for cos (fails on seed 42) and loss0 (overrates large changes) and
  wrong for fisher (I expected it close to loss0; it is the worst: 1-bit logits break the second-order estimate).
- Where they fail: the four single-seed input pairs of the W1A1 models (o, q k v, w1 w3, w2 2->1). Without them
  every measure but fisher is >= 0.90 on seed 42. The fine choice (which group of inputs) is what nobody sees.
- My mistake to note: the cost limit was set against a compiled full run; the fine-tune runs without compile and
  evaluates three times. Making it cheaper (compile, one evaluation, fewer windows) is the next step, and it must be
  judged on new pairs (ResNet-8), not on these.
- Next (Edén decides): (a) cheaper recover + ResNet-8 pairs from training_gap, same rule; (b) use it for the on-chip
  LLM: coarse choices with cos / loss0 (seconds), the short fine-tune only between close candidates.

### 2026-11-17: MendCode review (PR #100, 2 findings, both right)
- Major: resuming from an existing --out reused rows and controls without checking the settings (another
  --ft-steps / lr / data would mix measurements, or KeyError on missing report steps). Fix: the script now refuses
  an --out measured with other settings (seeds, pairs and --no-recover may differ). Checked: same settings resume
  (no GPU work, same Spearman, JSON unchanged); --ft-steps 20 on the same file is refused.
- Minor: a bare --out file name crashed os.makedirs(""). Fixed; smoke run with --out smoke_dp.json writes the file.
- The published results were made in one clean run with fixed settings: not affected.

### 2026-11-17: part 2 pre-registration (per-layer accuracy table, ResNet-8)
- Edén asked for a per-layer accuracy-drop tool (Nov 17). Searched first: AIMET QuantAnalyzer (per-layer
  enable / disable sweep, INT8) and HAWQ (Hessian trace) exist; no BNN library (Larq) has one.
- Question, layers, both directions, the fine-tune check and the rule written in README part 2 before any number.
- Design choice to note: BatchNorm recalibration after every change (needed: 1-bit weights change the conv output
  scale by ~1/|w|); reference networks recalibrated too.
