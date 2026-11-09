# ecg_mitbih — binary 1D convolutions for heartbeat classification

Arrhythmia detection on the **MIT-BIH Arrhythmia Database** (PhysioNet), the standard ECG benchmark: every heartbeat
of the MLII lead is classified into the four AAMI classes **N** (normal), **S** (supraventricular ectopic), **V**
(ventricular ectopic) and **F** (fusion), with the **inter-patient split** of de Chazal et al. (2004): the network
never sees a test patient while training. SYMBEX's fully binary 1D networks (`symbex_bool1d`) are compared with
**TensorFlow Lite Micro INT8** of the same architectures, trained with the same recipe, on four cores: ESP32,
ESP32-S3 and both cores of the Raspberry Pi Pico 2.

## Results in short

- **Accuracy: binary and INT8 are tied.** Balanced accuracy on the test patients (DS2, 3 seeds): small network
  67.0 ± 4.6 % (binary) vs 70.9 ± 6.1 % (INT8), medium 71.2 ± 3.3 vs 71.9 ± 4.6, large **75.0 ± 5.5 vs 72.4 ± 8.8**.
  Every gap is inside one standard deviation. The binary networks find more **S** beats (74-85 % vs 53-63 %), the INT8
  ones more N and V beats.
- **The limit is the data, not the network.** The rare classes rest on very few patients: 75 % of the test S beats
  come from one record (232), and F is learned from one patient (208) and tested on another (213). Bigger networks
  and one detector per class do not help (section 5), and INT8 meets the same wall. Every inter-patient MIT-BIH study
  faces it; what would help is a patient-adaptive input or more patients (future work).
- **Faster on every board** (weights in flash, 3 seeds): **6.0-13.4×** on the ESP32, **2.0-3.0×** on the Pico 2's
  Cortex-M33 against CMSIS-NN (3.0× for the large network with both models in RAM), **1.04-1.59×** on the ESP32-S3
  against ESP-NN. On the Pico 2's Hazard3 core, where TFLite Micro only has its reference kernels, 39-153×.
- **Much smaller:** 1.3-10.1 KB of weights and thresholds (TFLite model 11.4-68.6 KB), 3.4-5.7 KB of RAM in total
  (TFLite arena alone 7.3-45.9 KB; on the M33 SYMBEX adds its 64 KB bit-count table).
- **Exact:** on every board, every seed and every network, 1,000 beats per run, the chip gives 100 % the classes of
  the integer simulation.
- **Rhythm features help both methods.** Three constant input channels with the RR intervals (the time between
  beats, what a cardiologist reads first) add 2-5 points to both, chosen by cross-validation inside the
  training patients.
- **Two operating points.** The default recipe catches more arrhythmic beats but raises many false alarms (PPV of S
  24-33 %). A second recipe chosen by cross-validation (softer class weights, checkpoint on macro F1) cuts the false
  S alarms of the binary networks by 40-60 %: test accuracy 85-91 %, macro F1 52-57 % (binary) vs 51-60 % (INT8), at the cost of fewer S beats found.
  Both are shown (section 3b): the scores swing with very few patients (75 % of the test S beats come from one).

## 1. The task

| | |
|---|---|
| Data | MIT-BIH Arrhythmia Database (Moody & Mark 2001), 48 records, lead MLII, 360 Hz averaged to 180 Hz |
| Beats | one window per annotated beat: 256 samples (1.42 s) around the R peak, median removed, int8 (4 mV full scale) |
| Classes | AAMI EC57: N, S, V, F; class Q and the 4 paced records (102, 104, 107, 217) dropped |
| Split | inter-patient (de Chazal 2004): DS1 (22 records) trains, **DS2 (22 records) is the test set** |
| Validation | 5 DS1 records (`ecg_data.py`); for choosing the recipe, 4 patient-wise folds inside DS1 |
| Rhythm (RR) | 3 constant int8 input channels: RR before / local mean, RR after / local mean, local mean (10 beats) |
| Metric | **balanced accuracy** (mean of the four sensitivities): N is 89 % of the test beats, plain accuracy hides S and F; **macro F1** (sensitivity and false alarms together) for the second operating point |

| Split | N | S | V | F |
|---|---|---|---|---|
| train (17 DS1 records) | 35,019 | 664 | 2,644 | 399 |
| validation (5 DS1 records) | 10,813 | 280 | 1,142 | 15 |
| test (DS2) | 44,224 | 1,837 | 3,219 | 388 |

The networks (`ecg_data.ARCHS`): three binary convolutions of width 9 and two max pools, then one dense layer to the
4 classes. The same layers in both methods.

| Network | Layers (channels) |
|---|---|
| s | conv 8, conv 16, pool 4, conv 16, pool 4 |
| m | conv 16, conv 32, pool 4, conv 32, pool 4 |
| l | conv 32, conv 64, pool 4, conv 64, pool 4 |

The recipe, the same for SYMBEX (`ecg_train.py`, PyTorch) and INT8 (`ecg_tflite.py`, Keras, then full-integer
post-training quantization): Adam, learning rate 5e-3 with cosine decay, 100 epochs, batches of 128, cross-entropy
weighted by the inverse class frequency, checkpoint at the best validation balanced accuracy, seeds 42, 43, 44.

## 2. Choosing the recipe: patient-wise cross-validation

**Question.** Do augmentation or rhythm features help, and do they help both methods alike?

**What.** One validation split of 5 patients is noisy (it holds only 15 F beats), so the recipe was chosen with
4-fold patient-wise cross-validation inside DS1 (`--cv`, seed 42, whole records per fold, the test patients never
in a fold). Metric: the mean of the four folds' balanced accuracy. Three recipes: base, `--augment` (random
amplitude, baseline wander, noise and shift, a new draw every epoch) and `--rr`.

**Result** (mean of the fold balanced accuracies, %, seed 42: a ranking):

| Network | binary base | binary augment | **binary RR** | INT8 base | INT8 augment | **INT8 RR** |
|---|---|---|---|---|---|---|
| s | 58.4 | 55.6 | **60.7** | 55.8 | 59.3 | 58.1 |
| m | 60.7 | 58.3 | **63.5** | 56.4 | 58.7 | **61.5** |
| l | 61.1 | 58.1 | 63.3 | 57.2 | **60.8** | 60.5 |

Binary augment + RR: 61.0 / 60.9 / 61.8 (s / m / l), below RR alone.

**Answer.** RR helps both methods (+2-5 points). Augmentation helps INT8 (+2-4) but hurts binary (-2 to -3): the
binary networks are small and augmentation makes the task harder, not the data larger. The rule is the same recipe
for both, so the comparison uses **RR without augmentation**, the best binary recipe and the best INT8 recipe for m.
On s and l INT8 would gain 1.2 and 0.3 points with augmentation instead of RR (seed 42 only), less than the seed spread of the test.

## 3. Accuracy on the test patients

Balanced accuracy on DS2, RR recipe, 3 seeds (mean ± standard deviation); accuracy and sensitivities are means.

| Network | Method | **Test balanced %** | Accuracy % | N | S | V | F |
|---|---|---|---|---|---|---|---|
| s | SYMBEX binary | 67.0 ± 4.6 | 71.1 | 70.7 | 74.2 | 78.4 | 44.6 |
| s | TFLite INT8 | 70.9 ± 6.1 | 82.2 | 82.7 | 60.2 | 91.5 | 49.2 |
| m | SYMBEX binary | 71.2 ± 3.3 | 76.3 | 75.8 | 84.8 | 81.8 | 42.4 |
| m | TFLite INT8 | 71.9 ± 4.6 | 84.0 | 85.0 | 53.2 | 90.7 | 58.9 |
| l | SYMBEX binary | **75.0 ± 5.5** | 81.0 | 80.9 | 83.9 | 84.1 | 51.0 |
| l | TFLite INT8 | 72.4 ± 8.8 | 80.8 | 81.2 | 63.3 | 88.8 | 56.4 |

SYMBEX with an int8 first layer (`--first int8`, the option `symbex_bool1d` offers): 67.9 ± 7.2 / 65.4 ± 3.4 /
68.8 ± 3.9 %. Here it does not pay.

- **Tied within the spread.** No gap is larger than one standard deviation. The large binary network has the best
  mean.
- **Different errors.** The binary networks catch more supraventricular beats (S, the hardest class across patients);
  INT8 keeps more normal beats right (higher plain accuracy) and more V beats.
- **Validation does not predict the test.** Validation balanced accuracy (binary / INT8): s 61.4 / 51.4, m 58.2 /
  51.6, l 53.9 / 50.2; on the test the order is different. Inter-patient ECG varies a lot from patient to patient, F
  has only 15 validation beats, and the standard deviations of 3-9 points say the same.
- **The test set was scored three times**, once per recipe (the third in section 3b): first the base recipe (binary s / m / l 56.6 ± 4.0, 59.8 ±
  5.7, 60.9 ± 2.1 %; INT8 61.7 ± 12.5, 58.5 ± 6.2, 59.8 ± 7.7 %), then, after the cross-validation chose RR, this
  final one. The choice of RR was made only on the folds inside DS1; the base numbers are kept for the record.

## 3b. A second operating point: fewer false alarms

**Question.** The default recipe finds most S beats but most of its S alarms are wrong (S PPV 24-33 %, F 2-5 %). Can
a recipe with fewer false alarms be chosen without looking at the test?

**What.** Two new options in both scripts: `--weight-power p` (class weights = inverse frequency ** p, mean 1) and
`--select f1` (checkpoint at the best validation macro F1). Four recipes in the same 4-fold cross-validation (seed 42),
the criterion fixed before: the best mean fold macro F1 over the 3 networks × 2 methods.

| Recipe | default (p 1, balanced) | p 1, F1 | **p 0.5, F1** | p 0, F1 |
|---|---|---|---|---|
| mean fold macro F1 % | 47.7 | 52.2 | **53.5** | 52.9 |

**Result** on the test patients (the third and last scoring of DS2), recipe p 0.5 + F1, 3 seeds; next to it the
macro F1 of the default recipe (section 3). Sensitivity / PPV of S are means:

| Network | Method | **Macro F1 %** | (default recipe) | Balanced % | Accuracy % | S sens. | S PPV |
|---|---|---|---|---|---|---|---|
| s | SYMBEX binary | 52.2 ± 2.8 | 47.5 ± 2.2 | 64.3 ± 5.1 | 84.8 | 60.6 | 30.4 |
| s | TFLite INT8 | 55.8 ± 5.5 | 58.1 ± 5.0 | 60.3 ± 8.6 | 92.3 | 23.5 | 50.1 |
| m | SYMBEX binary | 57.4 ± 3.1 | 51.1 ± 0.6 | 62.5 ± 3.9 | 90.5 | 49.1 | 35.2 |
| m | TFLite INT8 | 59.5 ± 3.2 | 57.2 ± 3.8 | 65.8 ± 7.0 | 89.6 | 54.1 | 52.4 |
| l | SYMBEX binary | 56.3 ± 7.2 | 54.7 ± 1.3 | 60.1 ± 8.0 | 90.7 | 43.2 | 39.9 |
| l | TFLite INT8 | 50.8 ± 4.2 | 53.7 ± 8.7 | 51.2 ± 1.9 | 91.5 | 18.6 | 40.2 |

Exact as before: the 9 binary networks give 100 % in C on the PC (HAL row functions and `-mpopcnt`); the integer
simulation equals PyTorch on 99.99-100 % of the test beats.

**Why the scores swing.** The test classes rest on very few patients. Of the 1,837 S beats of DS2, 1,382 (75 %) come
from record 232; F is learned almost only from record 208 (372 of 399 training beats) and tested almost only on 213
(362 of 388). A recipe that moves how one patient is read moves the whole S column. The class weights do not teach
the network anything new: they choose a point on the same trade-off between beats found and false alarms.

**Answer.** Yes for the binary networks: +1.6 to +6.3 points of macro F1, accuracy up to 85-91 %, half the false
S alarms (from the mean sensitivity and PPV: 4,234 -> 1,660 on m), but fewer S beats found (43-61 % instead of 74-85 %). INT8 gains on m and loses on s and l. The two
methods stay tied within the spread. Which point is right depends on the use (screening wants sensitivity, a
wearable alarm wants few false alarms), so both are kept.

## 4. Speed and memory on the boards

**Question.** Does the binary network run faster than INT8 of the same shape on real microcontrollers, and does it
stay exact?

**What.** Each seed's binary network exported with `ecg_export.py` (CDT 1D exporter) and run with `tools/chip_test.py`
(1,000 validation beats, drawn with numpy seed 0); the TFLite model of the same seed run with `ecg_tflite.py --board`
(TFLite Micro, the firmware of har_ultracompact: ESP-NN on the ESP32 / S3, CMSIS-NN on the M33, reference kernels on
Hazard3) on the same beats. Weights in flash, median latency per beat, 3 seeds.

**Result** (ms per beat, mean over 3 seeds; standard deviations ≤ 0.02 ms everywhere):

| Board | Network | SYMBEX | TFLite INT8 | **Speed-up** |
|---|---|---|---|---|
| ESP32-S3 (PIE, ESP-NN) | s | 3.03 | 3.14 | **1.04×** |
| | m | 6.06 | 6.86 | **1.13×** |
| | l | 13.55 | 21.59 | **1.59×** |
| ESP32 (ESP-NN) | s | 5.47 | 32.95 | **6.0×** |
| | m | 11.69 | 94.48 | **8.1×** |
| | l | 30.55 | 408.46 | **13.4×** |
| Pico 2 Cortex-M33 (CMSIS-NN) | s | 5.70 | 11.66 | **2.0×** |
| | m | 11.74 | 35.10 | **3.0×** |
| | l, both in RAM | 29.60 | 89.52 | **3.0×** |
| Pico 2 Hazard3 (reference kernels) | s | 5.75 | 225.48 | 39× |
| | m | 10.55 | 799.95 | 76× |
| | l | 23.25 | 3554.52 | 153× |

On the M33 with weights in flash the large network gives 30.36 vs 383.44 ms (12.6×): the 68.6 KB TFLite model does
not fit the Pico 2's 16 KB flash cache, so the fair number is the RAM one above. Hazard3 only has TFLite's reference
kernels (no optimized RISC-V kernels for it): a measurement of what runs today, not of the best INT8 can do.

Memory (bytes; SYMBEX RAM = buffers + input + stack + tables):

| Network | SYMBEX weights + thresholds | TFLite model | SYMBEX RAM (S3 / ESP32 / Hazard3) | TFLite arena (S3 / ESP32) |
|---|---|---|---|---|
| s | 1,360 | 11.4 KB | 3,676 / 3,384 / 3,884 | 8.7 / 7.3 KB |
| m | 3,696 | 23.7 KB | 4,300 / 4,000 / 4,508 | 18.0 / 13.3 KB |
| l | 10,352 | 68.6 KB | 5,532 / 5,232 / 5,740 | 45.9 / 25.3 KB |

On the M33 SYMBEX adds its 64 KB bit-count table (lut16) in RAM: 69-71 KB in total.

**Exactness.** Every SYMBEX run: 100 % equal to the integer simulation, beat by beat, on all 4 cores, 3 seeds, 3
networks, plus the M33 RAM variant of l (39 runs). On the PC the compiled C gives 100 % with the HAL row functions and with `-mpopcnt`. The integer
simulation gives the class of PyTorch for 100 % of the test beats, except m seed 44 (99.4 %) and l seed 44
(99.99 %). TFLite Micro gives the class of the PC interpreter for 100 % of the beats, except l seed 44 (99.9 %).

**The cost of the rhythm channels.** Four input channels instead of one: m goes from 5.56 to 6.06 ms on the S3, 9.43
to 11.69 on the ESP32, 9.88 to 11.74 on the M33, 7.72 to 10.55 on Hazard3.

**Answer.** Yes, on every board, and exact. The gain is largest where there is no SIMD (ESP32: 6-13×) and grows with
the network. On the ESP32-S3, whose ESP-NN kernels use the PIE vector unit, the small network is only 4 % faster:
there the gain of SYMBEX is mostly memory (6.6-8.4× less flash, 2.4-8.5× less RAM).

## 5. Experiment: one detector per class

**Question.** Binary networks are cheap. Is it better to split the task into one small binary detector per class (N,
S, V, F: "this class or not"), each free to specialize (S lives in the rhythm, V in the beat shape), than to train one
4-class network of about the same cost?

**What.** `ecg_detectors.py`: four detectors of size s (together about the cost of one m, ESTIMATE from the weight
count) and four of size m (about one l), recipe p 0.5 + F1 with RR, trained side by side with ONE checkpoint for the
four (the same freedom of choice as one network); the class is the detector with the largest margin, no thresholds
tuned. Cross-validation inside DS1 only, 3 seeds; the test set was not used.

**Result** (mean fold macro F1 %, 3 seeds; pooled out-of-fold sensitivities):

| Model | Macro F1 % | S sens. | F sens. |
|---|---|---|---|
| one m network | 52.8 ± 0.9 | 29.6 | 5.9 |
| **4 detectors s** (≈ m) | 53.2 ± 2.0 | 29.0 | 23.0 |
| one l network | 52.2 ± 1.2 | 33.1 | 3.6 |
| 4 detectors m (≈ l) | 51.4 ± 0.8 | 30.7 | 14.3 |
| one s network | 52.4 ± 1.0 | 31.1 | 10.7 |

**Answer.** No. Four detectors s tie with one m network (+0.4, inside the spread) and four detectors m lose to one l
(-0.9). S does not move: the limit is the patients, not the way the task is split. Only F improves, and F rests on
one training patient. The single network stays.

## Future work

- More seeds or a k-fold test protocol: with 3 seeds and inter-patient variance, accuracy differences of 1-4 points
  cannot be called.
- A fresh test set never used here (INCART, or SVDB resampled from 128 Hz): DS2 has now been scored three times.
- Scores per test patient, and inputs that adapt to the patient (the beat minus a running template of that patient).
- N / S / V without F, which is one training patient against one test patient.
- The INT8 side with augmentation (its best recipe on s and l in the cross-validation), shown next to this one.
- The vertical engine (`symbex_bool_vertical`) for 1D, and a PIE first-layer kernel for 4-channel inputs on the S3.
- Energy per beat with the current meter: the metric a wearable cares about most.

## How to reproduce

From `benchmarks/conv1d/ecg_mitbih/`, with `benchmarks/env_bench` (the 48 records `.hea / .dat / .atr` in
`benchmarks/data/mitbih/`; TensorFlow needs the `nvidia/cusolver/lib` folder of env_bench in `LD_LIBRARY_PATH`):

```bash
python ecg_data.py                                          # -> data/ecg_mitbih.npz
python ecg_train.py --arch m --cv --rr                      # recipe choice (also --augment, and without --rr)
python ecg_tflite.py --arch m --cv --rr --gpu-memory 6144
python ecg_train.py --arch m --rr --seed 42                 # final training, scores the test once
python ecg_tflite.py --arch m --rr --seed 42
python ecg_export.py runs/m_binary_e100_s42_rr              # integers, model_symbex.h, EXACTNESS on the PC
python ../../../tools/chip_test.py runs/m_binary_e100_s42_rr --board s3 --port /dev/ttyACM0 --popcount swar --usb
python ecg_tflite.py --arch m --rr --seed 42 --board s3 --port /dev/ttyACM0 --usb
python ecg_train.py --arch m --rr --weight-power 0.5 --select f1 --seed 42   # second operating point (also --cv)
python ecg_tflite.py --arch m --rr --weight-power 0.5 --select f1 --seed 42
python ecg_detectors.py --arch s --rr --weight-power 0.5 --select f1 --cv --seed 42   # one detector per class (CV only)
```

Boards: `--board esp32 --popcount swar`, `--board pico2_arm|pico2_riscv --popcount lut16`, `--weights ram` for the
RAM variant (both scripts).

Results: `results/symbex/<run>/` (train.json, export.json, chip_<board>_<popcount>_<weights>_mask.json),
`results/symbex/cv/` (cross-validation; `det_*` the detectors), `results/tflite/` (tflite_pc_*, tflite_<board>_*, cv_*).
Second operating point: the same names with `_wp0.5_self1`.
