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

### 2026-11-17: FIT paper read (arXiv 2210.08502, full text) + two BNN sources (search snippets only)
- FIT = sum over layers of Tr(empirical Fisher of the layer) / n(l) x ||quantization noise||^2, weights and
  activations (derivatives w.r.t. activations for the activation term); EF trace by sampling until tolerance 0.01
  (82 iterations on U-Net). Computed on the trained FULL-PRECISION model (our "drop" direction, not "recovery").
- How they validate (the part we need): their truth is many RANDOM mixed-precision configurations (every layer gets
  a random bit width from {8, 6, 4, 3}), each trained 30 epochs from the same full-precision checkpoint; metric vs
  final accuracy by rank correlation over the configurations ("hundreds"; U-Net 50 configs: 0.86). Table 2: FIT
  ~0.6-0.9; activation-only terms much worse with BatchNorm (0.33-0.39). Never tested at 1 bit.
- Lesson for us: (1) their truth changes many layers per configuration, so the accuracies spread widely and a rank
  correlation is meaningful; our one-layer-at-a-time gains (0-1.3 points, sd up to 1.2) are mostly seed noise;
  (2) their truth is a short training from the full-precision model (30 epochs), not a full run; (3) BatchNorm hurts
  activation sensitivity measures: ResNet-8 has BN after every conv.
- Search snippets, NOT read: BNN survey (arXiv 2004.03333) describes switching one layer between 1 bit and full
  precision as an importance test (= our drop column); arXiv 2208.00398 says the first layer matters more for
  inference, the last for training, and keeping both real is not always best; RBNN (arXiv 2009.13055) plots
  per-layer cosine between real and binarized WEIGHT vectors (not feature maps). The report's "BD-BNN per-layer
  feature cosine" was not found.

### 2026-11-17: part 3 pre-registration (random mixed configurations, FIT's protocol at 1 bit)
- Edén agreed to try FIT's validation protocol. README part 3: 40 random configurations, 30 epochs from the twin,
  measures drop_sum / fit_w / fit_a / fit / acc0 / cos_end, rule coarse (≥ 0.8 all) + fine (≥ 0.6 inside each stem
  group), each seed.
- Smoke test before writing the rule (config_table.py --limit 1 --epochs 2, seed 42, not saved): speed only, 2.8 s
  per epoch -> ~85 s per configuration, ESTIMATE ~2.8 h for 40 x 3. It printed the per-layer FIT values of seed 42
  (a measure, not the truth; no trained ranking seen). Noted for honesty.
- Bug caught by the smoke design, not by a crash: retain_grad on activations must be off outside the Fisher pass
  (no-grad recalibration / evaluation); the Recorder has a record flag.

### 2026-11-17: literature round 2 (Edén's deep search, checked by me) + one exploratory measure, before any analysis
- Written while config_table.py runs (2 of 120 configurations logged; no Spearman computed or looked at).
- Verified:
  - PQuantML (github.com/cern-nextgen/PQuantML, CERN group of FIT's authors): src/pquant/core/torch/fit_compress.py
    exists. Its empirical Fisher uses MINI-BATCH gradients squared x batch size (its docstring says this differs from
    the paper's per-sample definition); an activation term exists (gradients w.r.t. each layer's input, first layer
    skipped) but FITCompress's search uses weights only. Ours is per-sample (batch of 1) = the paper's definition, on
    the BatchNorm OUTPUT (the tensor that is binarized). pruning_methods/fitcompress.py not checked.
  - Saturation Makes Quantization Error Additive (J. Hill, arXiv 2607.12266, Jul 2026): coverage model
    f(S) = c (1 - prod_{i in S} (1 - a_i)) from per-layer break rates. Variance claims not checked.
  - When Does Blockwise Additivity Break? (arXiv 2611.07487): after fine-tuning the additive vs pairwise gap shrinks
    to <= 1.8 points on Swin-T in 500 steps, 3.3-9.9 remain on DeiT after 10,000 (unequal budgets). The report's
    "fine-tuning repairs it on Swin, not DeiT" is a stronger reading than the abstract.
  - MixBin (arXiv 2211.06739): compares amplitude loss (BNAS), Adaptive MBQ, HAWQ-V2, network slimming to choose
    which layers to binarize; judged only by accuracy / FLOPs of the trained networks, NO rank correlation.
  - Pandey et al. (arXiv 2302.05397): Kendall tau of FIT / SQNR / accuracy sensitivity lists vs a ground-truth list
    (per-layer accuracy, no retraining, W4A8 / W8A8). PTQ, not a trained truth.
  - Sakr 2022 = Charbel Sakr et al., ICML 2022 (arXiv 2206.06501, OCTAV / magnitude-aware differentiation).
- Not verified: BMPQ, Hybrid-Net, GMPQ, TASA / CASQ, Spike-aware SNN (Oct 2026), DepthQuant, Down-projection
  fragility, CLADO, CoopQ, QEP, the FIT OpenReview quote on BatchNorm folding, Fouad Sakr EDGE 2022.
- Consequence: nothing found contradicts the part 3 design; still no paper validates a measure against TRAINED
  1-bit mixed configurations (MixBin trains them but reports no correlation).
- Exploratory measure added NOW (not in the pre-registered rule, reported apart): drop_cov = sum over the binary
  layers of -log(1 - drop_l / c), c = the recalibrated twin's accuracy - 10 (chance), Hill's coverage model; drop_l
  clipped below c. Computed afterwards from the saved JSON (no rerun). It is NOT a monotone transform of drop_sum
  across configurations, so it can rank differently (it weighs the stem's 37-point drop more).
