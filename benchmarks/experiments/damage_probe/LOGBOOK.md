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

### 2026-11-17: part 2 run (layer_table.py, 3 seeds, results/layer_table.json)
- Smoke test first (seed 42, no fine-tune, /tmp): the switch reproduces training_gap's twin and binary outputs
  exactly (torch.equal on 1,000 validation images, asserted in the script). Table 12 s per seed.
- Full run ~9 min, no errors. Rule FAILS: one-shot recovery vs fine-tuned gain -0.04 / 0.16 / 0.24. Cost passes (12 s).
- Drop is stable (seed-vs-seed 0.68-0.83) and finds the stem (37 points; fine-tune +2.2 on every seed, int8 stem
  trained +4.5). Activations cost more than weights inside the blocks.
- The check itself is weak: fine-tuned gains 0-1.3 points below the stem, seed-vs-seed 0.29-0.53. Not a truth for
  the small layers. Noted in README limits.
- Recalibration lowers the twin 83.6 -> 81.7 (stored statistics come from augmented batches; recalibrated without
  augmentation) and moves the binary 70.0 -> 69.9. Same for every row, so differences are fair; noted.
- Expectation: stem first right; fc / last stage second wrong; shortcut convs last right only for the drop.
- Next (Edén decides): (a) a real truth for ResNet-8 (full training with each layer real or int8, 30 runs, ~4.7 h
  on the GPU, the binary net's recipe) to judge the drop column below the stem; (b) the same table for the MLP; (c)
  drop table in a tool (tools/dev) only after (a).

### 2026-11-17: MendCode review of part 2 (PR #100 at c89330f, 2 findings, both right)
- Major: resuming layer_table.py did not check the inputs (--runs, --data were excluded from the settings check):
  other checkpoints or data would be skipped or mixed silently. Fix: every seed row stores its provenance (sha256 of
  twin_s<seed>.pth, binary_s<seed>.pth and of the split: validation images + labels, training labels); a cached seed
  is recomputed and compared, a mismatch is refused. The committed rows got their provenance afterwards (the
  checkpoints are unchanged since Nov 4, before the run; noted in the JSON as provenance_note). Checked: same inputs
  resume with no GPU work and the JSON byte-identical; a Bop checkpoint copied as binary_s42.pth is refused.
- Minor: README said b4 was the only layer after the stem with a fine-tuned gain above its sd; b2 (+1.19 ± 0.90) and
  fc (+0.43 ± 0.33) are too. My mistake reading the table; fixed (b4 is the only clear one, sd 0.24).

### 2026-11-17: literature (Edén's deep search; checked by me with web search, the rest NOT verified)
- Verified:
  - NNCF uses HAWQ-V2 for mixed precision: layer sensitivity = mean Hessian trace (Hutchinson) x squared L2 norm of
    the weight quantization perturbation (NNCF paper, arXiv 2002.08679).
  - NNCF `quantize_with_accuracy_control` (max_drop): keeps the most harmful operations in the original precision
    until the drop is met. INT8 PTQ only, no retraining. The class name "QuantizationAccuracyRestorer" given by the
    report was not found (only AdvancedAccuracyRestorerParameters).
  - NNCF binarization (PyTorch, Conv2D only, XNOR or DoReFa scale, staged training: activations then weights). No
    per-layer damage table for binary networks.
  - FIT (Zandonati et al., ICLR 2023, arXiv 2210.08502): layer-wise Fisher trace x expected quantization noise,
    weights and activations in one metric, claims to predict the final accuracy without retraining (authors'
    claim; their tables not checked).
- Not verified (report only): BD-BNN per-layer cosine against the FP network, C-PTQ 2026 (per-channel
  Fisher-weighted), BDM + entropy 2025-26, NNCF v3.3.0 HAWQ aggregator fix, QAIRT / ONNX Runtime / ESP-PPQ /
  TFLite debuggers (all PTQ, compare tensors without retraining).
- What it means here: the report's thesis (no-retrain measures overstate the damage because the rest of the network
  co-adapts; nobody measures damage after retraining because it costs O(N) trainings) is what parts 1 and 2 found
  (loss0 overrates large changes; one-shot recovery is noise). FIT's Fisher is on the WEIGHTS, not the layer
  outputs we tried in part 1 (which broke on 1-bit logits): not ruled out. Our networks have no scale alpha (pure
  sign) and Bop has no latent weights: a FIT-style column needs adapting, not copying.
- Proposed next (Edén decides): a real truth for ResNet-8 (full training with each layer real, 10 layers x 3 seeds,
  ESTIMATE ~4.7 h), then judge on it, with a rule written first: drop, FIT on weights, activation entropy (IR-Net),
  cosine binary vs twin maps (BD-BNN). Risk: at 3 seeds the small layers may stay inside the seed spread.
