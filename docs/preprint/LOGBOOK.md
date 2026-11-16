# Preprint logbook

Inconsistencies found while reading `draft.md`, so none is forgotten. Each one: what, where, source of truth, status.
Newest at the bottom of each list.

## Fixed

| # | Where | What was wrong | Fix | Commit |
|---|---|---|---|---|
| 1 | Authors | "Obed" | Obed Gonzales (draft and outline) | 2c970f6 |
| 2 | Abstract, intro, §2.2, conclusions, outline | "four instruction sets" / "five cores of four ISAs": LX6 and LX7 are one ISA (Xtensa; the LX7 adds PIE) | three commercial chips, four cores, three ISA families (Xtensa, Arm, RISC-V); the Uno's AVR apart, a fourth ISA, Digits only | 2c970f6 |
| 3 | Abstract, intro, §3.2, conclusions | ESP32 "5-13×" | 5-15× (ResNet-8 vertical compact 14.8×, PR #48) | 2c970f6 |
| 4 | §3.2, conclusions | Hazard3 inside "without SIMD 5-13×" | apart: 39-153× (ECG), ~98× ResNet-8, reference kernels only | 2c970f6 |
| 5 | §3.2 | M33 "2-3×" | 2-8× (ResNet-8 6.7, HAR 3.3-7.9, MLP 1.9-6.9, ECG 2.0-3.0) | 2c970f6 |
| 6 | §3.1 ResNet-8 table | no ESP32 vertical row; TFLite "chip equal 98.1 %" hid Hazard3 | row ESP32 vertical compact 48.6 ms (14.8×); 96 % on Hazard3, 100 images | 2c970f6 |
| 7 | §3.1 text | "on the ESP32 and the Pico 2 the standard engine stays the fastest" | measured: vertical wins on ESP32 (54.7 / 48.6 vs 62.5) and M33 (66.9 / 61.0 / 59.1 vs 67.8); standard wins on Hazard3 | 2c970f6 |
| 8 | §2.2 | ESP-DL "100 test inputs" | 200 (MLP) / 100 (ResNet-8), from the ESP-DL check's input files (N_IMG) | 2c970f6 |
| 9 | §3.2 | clipping "+1.7" alone | +1.7 in the training-gap study; no gain in the benchmark recipe (PR #47, 70.9 ± 1.0 vs 71.3 ± 0.8) | 2c970f6 |
| 10 | §3.1 HAR table | population std (ddof 0), the README moved to sample std (PR #43) | sample std from har_ultracompact/README.md | this commit |
| 11 | §3.1 HAR | names min / max-10 % ... while the README uses xs / s / m / l | both names given once | this commit |
| 12 | §3.1 HAR | int8 first layer "faster on every board" | about as fast as the fastest binary first layer (-1.8 % to +2.4 %; review: min on the M33 0.84 vs 0.82), slower on max on the ESP32 (14.07 vs 12.78) | this commit |
| 13 | §3.2 | M33 "2-8×" left out ECG l in flash (12.6×, 30.36 vs 383.44 ms: the flash effect, like the MLP's 28× on the S3) | named in the flash-effect sentence | this commit |

## Open

| # | Where | What | Needs |
|---|---|---|---|
| A | §3.1 ResNet-8 table | "fully binary" column (S3 39.2, M33 74.5) is the Oct 28 / Nov 3 library; "int8 stem" column the Nov 6 one | one re-measurement of the whole table on current master, or a note with the library date per column |
| B | §3.1 MLP | S3 0.420 ms is the standard engine; the dense vertical engine (PR #46, merged) gives 0.292 ms | add the vertical row (S3 rule: claims use the vertical engine) |
| C | §3.1 HAR / ECG, §5 future work | S3 numbers are the standard engine (1.04-1.90×); the 1D vertical engine (PR #84) gives 1.72-3.51× vs TFLite; §5 still lists it as future work | after #84 merges: S3 rows from #84, drop it from future work |
| D | §3.3 energy table | ResNet-8 rows use the vertical engine, MLP / HAR / ECG / FOMO the standard one (ENERGY.md of each); only ResNet-8 says so | engine named on every row; vertical energy for the others is a new measurement |
| E | §3.1 vs §3.3 HAR | speed table = fully binary; energy = max with int8 first layer | say it in the energy row (or measure the same variant) |
| F | §3.2, abstract, conclusions | "with SIMD the standard engine wins 1.0-2×" is the standard engine; on the S3 the vertical gives 5.0-5.4× on ResNet-8 | rewrite once A-C are settled |
| G | §2.2, §3.1 Digits | Uno: TFLite Micro needs 32 bits (official docs), but AIfES (Fraunhofer) documents an INT8 (Q7) dense path on the Uno; emlearn runs float MLPs on AVR. Text reworded (this commit) | measure Digits SYMBEX vs AIfES Q7 on the Uno (same 64-256-10 net) before any claim |
| H | §3.1 Digits | the Uno runs one benchmark only; it was the first target chip | ESTIMATE (to verify): HAR min (0.7 KB weights) and ECG s (1.3 KB) may fit 32 KB flash / 2 KB RAM; MLP Fashion does not (4.6 KB RAM) |
| J | §3.1 FOMO, §3.3 | FOMO speed / energy use the standard engine; no vertical FOMO measured | say so or measure |

## Verified against the source (no change)

- ResNet-8 latencies, flash and accuracy (3.1); S3 vertical 9.70 / 7.07 / 6.02; FOMO 5.7×; ECG int8-first 67.9 / 65.4 / 68.8;
  76.1 % (int8 stem + Bop, 76.08); powers ESP32 0.33-0.46 W, S3 0.69-0.99 W.
- HAR speed table (flash, ±1 bytes column) and the 9 / 23 point gaps: equal to har_ultracompact/README.md.
- ECG table (accuracy, S sensitivity 74-85 vs 53-63 %, latency on 4 cores, sizes): equal to ecg_mitbih/README.md.
- Low-bit study (MLP 784-256-128-10): 87.64 / 89.95 / 90.04 %, 28.7 / 114.6 / 229.2 KB, 10.3 / 13.4 ms flash,
  0.556 / 0.525 ms RAM: equal to mlp_fashion/README.md.
