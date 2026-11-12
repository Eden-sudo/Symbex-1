# experiments -- ideas measured before they reach the CDT

New ideas start here: a study on the PC, a small firmware on the chips. They go into the CDT or `lib_symbex` only
once the experiments show they really work. One folder per question, each with its own README, scripts and
`results/`:

| Folder | Question | What we found |
|---|---|---|
| [`pair_bits/`](pair_bits/) | does a *pair* of bits per neuron help a fully binary network learn? | yes, on the activations: +2 to +3 points at the same memory (MLP and ResNet-8) |
| [`training_gap/`](training_gap/) | how much of the binary ResNet-8's loss is training, and how much capacity? | mostly capacity; Bop +2.5, twostep +2.35; width, depth (1-3-3: 77.3 %), shortcuts (addpool +1.5, the 1×1 costs ~3.2 ms on the chip), int8 classifier +2.8; padding, BatchNorm, block order: nothing |
| [`chip_kernels/`](chip_kernels/) | which counting and int8 kernels are fastest on each chip? | `pie_swar`, AND + PIE, the vertical count, the int8 stem (faster only with SIMD) |
| [`tiny_llm/`](tiny_llm/) | does a small language model still write stories with 1-bit weights (int8 activations), integers only, exact in C? | yes on the PC: 15.7 M parameters, 90 % of them 1-bit, perplexity 4.58 (fp32 3.67), integer model = float, C exact on every score; 3.55 MB; ESP32-S3: exact, 5.25 tokens/s (seed 42) |
| [`other_probes/`](other_probes/) | single questions: PopLUT weight codebooks, arithmetics on the GPU | |

Every study follows the rules of the benchmarks: seeds 42, 43 and 44, choices made on a validation split of the
training data (the test set is never read here), means ± standard deviation; single-seed numbers are rankings. Chip
measurements check every method against plain C on the chip.

Run each script from its own folder, in `benchmarks/env_bench` (or the main `.venv`), for example:

```
cd training_gap && python train_gap_study.py --model binary --seeds 42 43 44
cd chip_kernels && python int8_first_bench.py --port /dev/ttyACM0
```
