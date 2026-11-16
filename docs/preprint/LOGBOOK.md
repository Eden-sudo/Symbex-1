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
| 14 | Abstract, intro, conclusions | the M33 inside "with SIMD or DSP 1.0-2×" (it is 2-8×) | the 1.0-2.0× is the S3 standard engine only; M33 2-8× and S3 vertical 5.0-5.4× named apart | this commit |
| 15 | Abstract, §2.2, contributions | "five benchmarks" with six in the table | five tasks against INT8, plus Digits on the Uno | this commit |
| 16 | §3.2 clip | 71.3 ± 0.8 called validation: it is the TEST accuracy of the Oct 28 model (no validation recorded); PR #47's comment had the same mix | validation of the same recipe without clip: 71.17 ± 0.32 (symbex_seeds_pc_w1_base.json) against 70.93 ± 1.02 | this commit |
| 17 | §3.1 ResNet-8 | ESP-DL ResNet-8 latency missing (only the MLP's) | row 14.8 ms flash / 9.8 ms RAM (quick check, Nov 2) + the 1.43 / 1.53× and 1.06× sentence | this commit |
| 18 | §3.1 ECG | 12.6× M33 in flash cited but the table showed the RAM row | table cell: flash 30.36 / 383.44 (12.6×), RAM 3.0× beside it | this commit |
| 19 | §3.3 energy | prepared 1.51× vs a table default of 7.2 (7.2 / 4.5 = 1.60) | the 1.51 uses the Nov 15 default (6.86 mJ); 7.24 is the Nov 14 run; both named with dates | this commit |
| 20 | §3.1 low-bit | 89.3 − 87.6 = 1.7, text 1.6 | two decimals (89.26 − 87.64 = 1.62) | this commit |
| 21 | §3.1 cost table | Digits' 94.4 % (FP32) in the "INT8" column; FOMO "none" without caveat | column "Reference (INT8 unless noted)"; FOMO "none measured" + caveat, also abstract, intro, conclusions | this commit |
| 22 | Abstract, §2.2 | "exact on 100 % of its test inputs": the check uses validation inputs (PC up to 2,000, board 1,000) | "inputs checked (validation inputs)"; title "bit-exact" defined | this commit |
| 23 | §2.2 | QAT (MLP) and PTQ (rest) mixed without a reason; SIMD use of the INT8 side not stated | one paragraph: converter PTQ everywhere, MLP accuracy from QAT (the higher, so the cost is not understated); ESP-NN uses PIE, CMSIS-NN the M33 DSP (fashion_tflite.py builds TARGET_ARCH=cortex-m33, "DSP on") | this commit |
| 24 | §3.3 | RNN bullet with no result file in the repository | reduced to "no recurrent layers in the chain" | this commit |
| 25 | §3.1 HAR table | no flash for max-10 / max-5; "ms" only in the first cells | 2.9 / 20.5 and 5.6 / 41.8 KB (README sec. on flash, INT8 Pico 2 build); units in the headers | this commit |
| 26 | §1.1 | Larq quotes without a source | GitHub discussion #775 (CNugteren, Jul 5 2023), quotes checked word for word; bib nugteren2023lce775 | this commit |
| 27 | §3.2 | "9-16 points" for fine tasks (HAR reaches 23) | 9-23 | this commit |
| 28 | Conclusions | "four cores of three commercial chips and the Arduino's AVR" (the AVR is commercial too) | "four 32-bit cores of three chips and the 8-bit AVR of the Arduino Uno" | this commit |
| 29 | Every results table | no measurement date | a "Measured" line per table (result files' commit dates, renames followed: tools/dev/results_age.py, PR #98) + one sentence in §2.2 | this commit |
| 30 | §2.2 Baselines | "the S3 comparison uses the vertical engine": only ResNet-8 does (FOMO, HAR, ECG on the S3 = standard) | "the S3 comparison of ResNet-8 also uses…", the others named | this commit |
| 31 | §3.1 ResNet-8 | ESP-DL RAM missing, so "the vertical uses less RAM" could not be checked; the 63 KB was the flash run | ESP-DL 147 KB (docs/en/results.md); vertical default 87 KB with weights in RAM, 63 KB in flash | this commit |
| 32 | §3.1 ECG | 12.6× put on "68.6 KB does not fit the cache", but m (23.7 KB) does not fit 16 KB either and gains 3.0× | "mostly"; likely reason (NOT measured): largest INT8 layer of m 9.2 KB fits the cache, l's 18.4 / 36.9 KB do not; only l measured in RAM | this commit |
| 33 | Limitations, Not written yet | re-measurements promised in the date notes were in no list | both lists: one library version, Figures 2-4, abstract ranges; MLP two-model note in Limitations | this commit |
| 34 | Abstract | ~330 words (venues ask ≤ 250) | 248 words: cores list, MLP 28×, capacity clause and the chain's steps moved to the body | this commit |
| 35 | Title | "Bit-exact" defined in a parenthesis | parenthesis removed; definition in §2.1 | this commit |

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
| K | §3.1 MLP, ESP32 flash | 3.55 ms (Nov 7, swar, flash) but the energy loop ran the same model, same counting, flash at 2.33 ms (Nov 15): likely the short-first-layer change (f127456); hence its "0.26 W" | re-measure the ESP32 row on current master (chip_test.py, 3 seeds); power is 0.39 W with the energy loop's own time |
| L | All tables | dates are commit dates (the day the number reached the repository); old files carry no library commit | measuring scripts write `provenance` (after #83 / #84 merge), then one re-measurement on one library version |
| M | §2.2, §3.1 MLP | INT8 accuracy from the QAT (Brevitas) model, latency from the converter's PTQ model of the same shape: a reviewer may ask for one model | export the QAT model to `.tflite` and measure its latency (S3, ESP32, M33) |

## Verified against the source (no change)

- ResNet-8 latencies, flash and accuracy (3.1); S3 vertical 9.70 / 7.07 / 6.02; FOMO 5.7×; ECG int8-first 67.9 / 65.4 / 68.8;
  76.1 % (int8 stem + Bop, 76.08); powers ESP32 0.33-0.46 W, S3 0.69-0.99 W.
- HAR speed table (flash, ±1 bytes column) and the 9 / 23 point gaps: equal to har_ultracompact/README.md.
- ECG table (accuracy, S sensitivity 74-85 vs 53-63 %, latency on 4 cores, sizes): equal to ecg_mitbih/README.md.
- Low-bit study (MLP 784-256-128-10): 87.64 / 89.95 / 90.04 %, 28.7 / 114.6 / 229.2 KB, 10.3 / 13.4 ms flash,
  0.556 / 0.525 ms RAM: equal to mlp_fashion/README.md.
