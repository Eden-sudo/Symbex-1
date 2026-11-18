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

### 2026-11-17: MendCode review of part 3 (PR #100 at 061b04d, 3 findings)
- Major, right: config_table.py read the per-layer drops from --table without checking they belong to the same
  checkpoints / split, and resume did not check the table. Fix: the seed's table row must carry the same provenance
  as the checkpoints in --runs (else refused); each result row stores table_drop_sha256 and resume compares it.
  Checked: wrong provenance and a missing seed are refused; smoke run (--limit 1, not saved) gives the same drop_sum
  (95.3) as the running job.
- Minor, right: --configs > 1024 (= 2^10 distinct subsets) looped forever. Now refused.
- Minor, not changed: "twin-only runs need the binary checkpoint". drop_sum comes from part 2, measured with the
  binary checkpoint; hashing it is what ties the drops to the same networks. Answered on the PR.
- The running job (started with the old code, same inputs) will save rows without table_drop_sha256: backfilled
  after it ends, noted in the JSON, as for part 2.

### 2026-11-17: part 3 run (config_table.py, 3 seeds x 40 configurations, results/config_table.json)
- ~2.7 h, no errors (the watcher scripts hung, not the run: `pgrep -f` matched itself; waits by PID now).
  Seed 44 ran slower (~100 s per configuration vs ~73): the GPU was shared.
- Rule: drop_sum, fit_a, acc0, cos_end pass; fit_w (0.70 on s43) and fit (0.76 on s43) fail the coarse test. Cost
  passes for all (Fisher 3-6 s, per-configuration measures ≤ 0.5 s). Noise ceiling 0.97-0.98: the truth is stable
  (unlike part 2's one-layer gains).
- Found AFTER the run (config_analysis.py, results/config_analysis.json): counting the binary layers passes the rule
  too (0.84-0.87 all, 0.91-0.98 in groups). The pre-registered rule did not ask the useful question. With the count
  held fixed: drop_sum 0.82 / 0.80 / 0.73, acc0 0.81 / 0.75 / 0.64, cos_end 0.81 / 0.73 / 0.73, drop_cov 0.80 /
  0.77 / 0.73, fit_a 0.49 / 0.64 / 0.57, fit_w 0.02 / -0.38 / 0.18; ceiling 0.86-0.93. Written as found, labelled
  post-hoc in README.
- drop_cov (declared before the analysis): no better than the plain sum.
- table_drop_sha256 added to the three rows after the run (the job used the code before the MendCode fix); each
  seed's provenance was checked equal to its layer_table.json row first. Noted in the JSON (table_drop_note).
- Expectation: wrong on fit_a being weakest (fit_w is) and on drop_sum / acc0 failing the fine test.
- Next (Edén decides): pre-register the "same count" test on another network (MLP or tiny_llm) before moving the
  drop table to tools/dev.

### 2026-11-17: MendCode review of part 3 results (PR #100 at 63fe400, 2 findings, both right)
- Major: the "same count" statistic removed only a LINEAR trend of the count from the ranks; counting itself kept
  0.10-0.17, so the column did not hold the count fixed. Fix: ranks centered inside each group of equal count
  (counting now gives 0.00). New numbers at the same count: drop_sum 0.83 / 0.82 / 0.72, acc0 0.84 / 0.78 / 0.67,
  cos_end 0.84 / 0.78 / 0.75, drop_cov 0.82 / 0.80 / 0.72, fit_a 0.47 / 0.64 / 0.54, fit_w -0.06 / -0.45 / 0.11,
  ceiling 0.86-0.92. The claim changes: drop_sum is NOT the best, it ties with acc0 and cos_end. README fixed.
- Minor: config_analysis.py did not check that --table belongs to --configs. Now it compares provenance and
  table_drop_sha256; a table with one drop changed is refused (checked).

### 2026-11-17: quick check, before vs after binarizing (quick_check.py, seed 42, 96 s, a ranking)
- Edén's question: the measures so far all start from the full-precision network; does the predicted per-layer drop
  name the layers that cost the most once the network IS binary?
- Twin 81.26 -> every layer binary, one shot 13.40 -> fine-tuned 30 epochs (B) 53.72 (far below the binary trained
  from scratch, ~70: 30 epochs are not enough for all layers at once).
- After-measure 1, one layer of B made real again without training: stem -8.94, b0 -1.04, fc -1.56, others 0 to
  +1.36. Spearman with the prediction -0.29. Making a layer real HURTS B: its other layers adapted to that layer
  being binary (same problem as part 2's recovery column). Not a valid after-measure.
- After-measure 2, cosine of each layer's BatchNorm output B vs twin: grows with depth (stem 0.17, ds4 0.57),
  accumulated error, not the layer's own. Spearman with the prediction -0.30.
- Next: a LOCAL error per layer in B (each layer fed the twin's input), then decide whether the slow truth
  (training with one layer real) is worth it.

### 2026-11-17: tiny_llm, before vs after binarizing + propagation (llm_layers.py, 3 seeds, ~33 s each, no training)
- Edén: which LLM layers hurt at 1 bit, how the damage travels through the blocks, and does the prediction on the
  fp32 model match the trained binary (W1A8) model? Units: 8 blocks x (qkv, o, w13, w2).
- BEFORE (fp32, one unit W1A8): b0.w13 +0.289, b0.qkv +0.151, qkv of every block +0.06-0.12, b7.w13 +0.105;
  o and w2 of the middle blocks +0.01-0.03. All 32 units at once +4.36 / +4.82 / +4.71 vs sum of singles ~1.9:
  the damage is 2.2-2.5x more than additive.
- Propagation: the error does NOT grow through the blocks; the residual stream carries it and it fades slowly
  (b0.w13 0.59 -> 0.40 at block 7, b0.w2 0.49 -> 0.24). Block 0 counts most because its error lives through all 8.
- AFTER (trained binary, one unit back to int8 latent weights, no training): EVERY unit makes the loss worse
  (-0.02 to -0.38), as in ResNet-8 (co-adaptation). But how much it hurts follows the prediction: Spearman -0.84 /
  -0.84 / -0.85 (sign: larger predicted damage = larger loss when changed back). Same top: b7.w13, b0.w13, b0.qkv.
  Local error (binary vs int8 output, same input): 0.74-0.77, but it mostly sorts by group (qkv ~1.5, w13 ~1.1,
  o / w2 ~0.5), not by block.
- Against the trained truth already known: prediction b0+b7 ~0.80 vs b3+b4 ~0.38 (2x), but trained int8 blocks
  gained 0.102 vs 0.095 (LOGBOOK of tiny_llm): the prediction names the sensitive layers, it does not say how much
  training recovers.

### 2026-11-17: does it improve when we change them? (training test, rule written BEFORE the runs)
- Question: llm_layers.py names b0.w13, b0.qkv, b7.w13 as the most sensitive units. If those 3 keep int8 weights
  (the rest 1-bit, W1A8, same recipe as binary_d384l8, 10,000 steps), is the trained model better than with 3
  middle units of the SAME shapes and bytes at int8 (b3.w13, b3.qkv, b4.w13)? 2,015,232 int8 weights each.
- train.py `--int8-units 0.w13 0.qkv 7.w13` (name d384l8_utop) vs `--int8-units 3.w13 3.qkv 4.w13` (d384l8_umid).
  Reference: binary_d384l8 s42 1.5252 (3 seeds 1.5224 ± 0.0026).
- Rule: seed 42 first (a ranking). If utop's validation loss is lower than umid's by > 0.01 (~4x the seed sd), the
  selection helps: then seeds 43, 44 and a paired result. Otherwise the answer is "no, the bytes matter, not which
  units" (as b07 vs b34 already hinted: 1.4219 vs 1.4262). The test split is not read.
- RESULT (seed 42, a ranking): utop 1.4820, umid 1.4832 (binary base 1.5252): utop is better by 0.0012 < 0.01.
  The rule says NO: with the same int8 bytes, the units the table names do not train better than middle units of the
  same shapes. Both gain ~0.043 over the base: the bytes matter, not which units (as b07 vs b34: 1.4219 vs 1.4262).
  Seeds 43 / 44 not run (the rule did not pass). The table names the sensitive units without training, but in this
  model, after training, protecting them is worth no more than protecting others.
- Context (Edén, Nov 17): the goal is LLMs on the ESP32 and Pico 2 too (Obed's: llama2.c 260K / 15M). The current
  15.7 M model does not fit there (RAM ~1.9 MB, the KV cache 1.67 MB; ESTIMATE from tinylm.c's allocations). Next
  (agreed in principle): llama2.c's 260K shape (dim 64, 5 layers, vocab 512), fp32 vs binary, 3 seeds, to the chips.

### 2026-11-17: are there LLM designs made for the edge, or are tiny LLMs small because of their data? (reading only)
- Both. Small because of the data: llama2.c 260K / 15M and our tiny_llm are a standard Llama on TinyStories (simple
  stories, short vocabulary). Designs for the edge: MobileLLM (arXiv 2402.14905, ICML 2024: deep and thin, shared
  input / output embedding, GQA, block-wise weight sharing, 125-350 M, phones); Super Tiny LMs (2405.14159, weight
  sharing, 10-100 M; abstract only); Opt.Gear-1M (2608.01034, Cortex-M7 STM32H747, W4A32, 20 tok/s by the authors,
  convolutional key-value gated mixer + local-global attention to cut the KV cache; abstract only); slvDev/esp32-ai
  (per-layer embeddings in flash, S3); RWKV-type recurrent models (constant state, no KV cache; blogs only, no MCU
  deployment found).
- For the ESP32 / Pico 2 our limit is the KV cache in RAM, not the weights (already 1-bit). What attacks it, easiest
  first: GQA (llama2.c 260K already has 4 kv heads of 8; tinylm.c does not support it yet), sliding-window attention
  (fixed RAM), deep and thin; a recurrent / convolutional mixer is a new architecture (later).
- Proposal (not started, Edén decides): the 260K shape with GQA + a window, binary, 3 seeds. Before citing
  Opt.Gear / Super Tiny LMs / RWKV in the paper: read the PDFs.

### 2026-11-17: part 4, do layer interactions rank part 3's configurations better than the sum? (3 seeds)
- interactions.py: 1,024 subsets x 3 seeds on the GPU, 3 runs in parallel (~10 min, RTX 4000 Ada laptop), BatchNorm
  recalibrated on 10,000 training images, validation only. acc0 reproduced exactly (0.00).
- Rule (README, written first): beat additive's same-count Spearman by > 0.05 on each seed. Additive 0.83 / 0.82 /
  0.72; shapley 0.84 / 0.81 / 0.72, popcount 0.79 / 0.76 / 0.79, pairs 0.59 / 0.58 / 0.47, acc0 0.84 / 0.78 / 0.67.
  None passes.
- ResNet-8 is sub-additive (sum of singles ~148 vs all binary ~68: floor at chance), unlike tiny_llm's 2.2-2.5x.
  Largest interactions: stem with b0 (-8 to -10). popcount group spread 2.08 / 2.08 / 2.96 vs seed spread 2.32.

### 2026-11-17: part 4b, the same interaction table in validation loss (3 seeds)
- Question (Edén): was part 4's sub-additivity only the 10 % accuracy floor? interactions.py --metric loss records the
  cross-entropy in the same pass (accuracies identical to part 4). 3 runs in parallel, ~11 min.
- Rule (README, written first): ratio = all-10 increase / sum of singles; > 1 on every seed = super-additive.
  Result: 0.92 / 0.97 / 0.98 -> sub-additive by the rule, but almost exactly additive (accuracy: 0.45-0.47). Pair
  interactions positive 42-43 of 45 (accuracy ~half). Loss all binary ~7 nats (chance 2.30): no saturation.
- Reading: the floor explained most of part 4's sub-additivity; ResNet-8 in loss ~ additive, tiny_llm 2.2-2.5x: not
  the same story, but much closer. Interaction models still do not beat the sum in ranking.
- Hypothesis for later (Edén, Nov 17, NOT measured): in the LLM everything goes through a residual that adds the
  errors of its 32 units, which accumulate; ResNet-8's shortcuts are OR, which can mask errors. Would explain
  super-additive (LLM) vs ~additive (ResNet-8).
- Edén on 4b's caveats: the ratio has no units, comparable across models (only task and architecture differ); the
  extreme-image caveat is worth checking -> loss_tails.py (median, clipped at 10), rule in README first.

### 2026-11-17: part 4b check, median / clipped loss (3 seeds, 57 subsets each, ~2 min GPU)
- Ratio by statistic: mean 0.92-0.98 (= 4b), median 2.10 / 2.47 / 3.14, clipped at 10 0.81 / 0.84 / 0.86. Rule
  (0.85-1.05 for both) FAILS: "close to additive" is not solid; the ratio depends on the statistic.
- Not a few images: worst 5 % = 13.6 % of the all-binary loss; 24-28 % of images exceed 10 (the clip is a floor).
  The median is a threshold statistic (single layers leave the median image right). Like-for-like vs tiny_llm is
  mean vs mean only: 0.95 vs 2.2-2.5.
- Synthesis (Edén, Nov 17): additivity = network + statistic (floor -> looks sub-, threshold -> looks super-,
  mean loss closest to neutral). For the paper: "claiming per-layer damage is additive requires naming the
  statistic". For the CDT: use the sum for ORDER (Spearman 0.72-0.83 at same count), never for MAGNITUDE.
  README part 4 summary paragraph added.

### 2026-11-17: part 5, fan-in (A) and distance to the output (B) vs the one-shot drop (no training)
- Hypotheses from Edén. fanin_depth.py on part 2 (ResNet-8) and llm_layers (tiny_llm) tables; rule in README first.
- Fan-ins from the checkpoints: ResNet-8 stem 216 (not 54: no BPR here; more than b0-b2's 144), ds2 16, ds4 32, fc 4096.
- A fails: ResNet-8 wa ~0, weights only +0.60..+0.70 (opposite); tiny_llm -0.31 and w2 < o only 2-3 / 8.
- B passes on ResNet-8 only (wa 0.54 / 0.80 / 0.78, from the activations; confounded with stage / stem); tiny_llm 0.03.
- Model LOO R^2 -0.20 / 0.13 / 0.59 (ResNet-8), ~-0.1 (tiny_llm): not useful.
- Post hoc, untested: tiny_llm groups ordered by number of matrices binarized (qkv 3 > w13 2 > w2 / o 1), then size.

### 2026-11-17: part 5 checks (rules in README first)
- ResNet-8 B: without the stem 0.36 / 0.72 / 0.70 (FAILS on s42), given fan-in 0.69 / 0.88 / 0.81.
- llm_matrices.py (56 single matrices x 3 seeds, ~1 min GPU, base = llm_layers'): w2 > mean(w1, w3) in 8 / 8 blocks
  every seed -> A rejected the other way (as ResNet-8 weights-only). q / k / v vs o 0.95-1.09: matrix count
  explains the group order, not the role. Unit = 1.31-1.38x the sum of its matrices (super-additive inside a block).
