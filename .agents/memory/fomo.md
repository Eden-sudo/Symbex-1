# Vision (FOMO): state and gaps

Code: `benchmarks/conv2d/fomo_voc_person/`. Owner: Abrahan.

## What exists

- Person detection on Pascal VOC, 96 x 96 RGB input, 8-level thermometer encoding, 12 x 12 grid of centers.
- Binary F1 31.2 +- 0.7 % against INT8 30.1 +- 0.2 % (3 seeds); ESP32-S3 126.3 ms against 256.9 ms; runs on the classic
  ESP32, where INT8 does not fit (README.md, "Person detection").
- Chip measurement: `fomo_symbex.py --board {esp32,s3,pico2_arm,pico2_riscv}`; `fomo_tflite.py` and
  `tools/dev/fine_profile.py` only accept `esp32` and `s3`.
- The October 23 figures must be re-measured on the fixed validation split (PR #10).

## Gaps for "more classes and a camera on the board"

Found by reading the code (November 2026); check again before relying on a line number.

- Two classes only (person / background): `model.py` (`classes=2`), `fomo_train.py` (two class weights),
  `fomo_data.py` (keeps only `person`).
- The exporter passes `n_classes` (`tools/compiler_core/export/boolean.py`), not yet tried with more than two.
- One threshold, for the person class only: `fomo_export.py`.
- No camera driver in the repository; images reach the chip over serial from the PC
  (`tools/compiler_core/chip/measure.py`), and `Symbex1_PIO/src/main.cpp` runs a fixed demo input.
- Data loader is VOC only (`torchvision.datasets.VOCDetection`); no loader for own images.
