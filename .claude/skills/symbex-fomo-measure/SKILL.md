---
name: symbex-fomo-measure
description: Trains, sizes and measures the binary FOMO person detector (benchmarks/conv2d/fomo_voc_person) on a board against its INT8 twin, and updates its figures. Use for any FOMO run, sizing study, re-measurement on the validation split, or a new board for FOMO.
---

# SYMBEX-1 FOMO measurement

Owner: Abrahan. State and gaps: [.agents/memory/fomo.md](../../../.agents/memory/fomo.md). Work from
`benchmarks/conv2d/fomo_voc_person/`; general rules in [symbex-benchmark-run](../symbex-benchmark-run/SKILL.md).

## 1. Binary detector

```sh
python3 fomo_symbex.py --channels 32-32-128 --blocks 1-2-2 --seeds 3 --board <board> --port <port>
python3 fomo_symbex.py --channels 32-32-128 --blocks 1-2-2 --seeds 3 --board <board> --port <port> --weights ram
python3 fomo_symbex.py --summary          # no training, export or chip run: summarize the run folders
```

- Boards: `esp32`, `s3`, `pico2_arm`, `pico2_riscv`; `--usb` for the S3 native port; `--popcount {lut16,swar}`
  (`swar` for minimum RAM, used on the classic ESP32 without PSRAM).
- `--redo` retrains seeds already trained; `--epochs` (default 100), `--weight` (person cell weight, default 40),
  `--images` (default 1000) for the C check and the chip.
- Every seed must be exact: export and chip equal to the simulation on 100 % of the images.

## 2. INT8 twin

```sh
python3 fomo_tflite.py --board <esp32|s3> --port <port>
python3 fomo_tflite.py --model reference/<file>.tflite --board <esp32|s3> --port <port>   # measure as it is
```

Only `esp32` and `s3` here (and in `tools/dev/fine_profile.py`).

## 3. Record

- Sizing and ranking by validation F1, never by test (#9); one fixed validation split (#10).
- Results JSON in `results/`, figures in the FOMO README, `README.md` and `README.es.md`, each with its file (#8, #15).
- Per-step flash / RAM captures are committed with the claim they support (#15).
- Label each result with board, owner and commit; classic ESP32 and shared S3 results are kept apart (`docs/roadmap.md`).

Finish with [symbex-pre-pr-check](../symbex-pre-pr-check/SKILL.md).
