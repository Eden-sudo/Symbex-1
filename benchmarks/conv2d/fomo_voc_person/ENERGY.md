# fomo_voc_person — energy per inference (classic ESP32)

**Question.** How much energy does the binary FOMO person detector use per frame on the classic ESP32?

**Answer.** **96 mJ per frame** (226 ms, 0.43 W). TFLite Micro INT8 of this detector does not fit the classic ESP32 (its arena is about 230 KB), so there is no INT8 number on this board: SYMBEX is the only one of the two that runs here.

## What was measured

- **Board:** classic ESP32 (no SIMD), weights in flash, powered and read through its USB-UART port.
- **Meter:** USB power meter between the PC and the board (V, A, mAh, Wh), read by eye; whole mAh.
- **SYMBEX:** runs/c32-32-128_b1-2-2_w40_e100_s42, lut16 popcount, flash; exact against its simulation on the chip, 100 %.
- **Phases:** one 10-minute run per method, meter zeroed before each. `tools/energy_loop.py` keeps the model busy on one frame and counts the inferences. Idle (8 mAh in 600 s, 0.25 W) is the Nov 14 reading of the same board (`../../conv2d/cnn_cifar_resnet8/ENERGY.md`).
- **Energy:** mAh × V × 3.6 J; the idle rate is subtracted for the seconds the meter timer ran past 600 s. Per inference = energy / runs. "Above idle" subtracts the idle 8 mAh first.

## Result

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Current seen | Power | **mJ per inference** | above idle |
|---|---|---|---|---|---|---|---|
| SYMBEX | 2,656 | 225.94 | 14 / 0.07 / 10:12 | not noted | 0.43 W | **96.4** | 40.7 |

- Voltage 5.14 V. Detection: binary and INT8 detect alike (F1 31.2 vs 30.1 %, 3 seeds; README).
- Raw readings and every derived number: `results/energy/energy_esp32.json`.

## Limits

- One 10-minute run per method on one board; the meter counts whole mAh (±1 mAh per reading).
- The idle draw was not measured again: the Nov 14 reading is reused. The classic ESP32 board does not warm up like
  the S3 (at most slightly warm by touch), so its idle draw is not expected to drift much; the total-energy ratios do
  not depend on it.
- Whole board at the USB port (chip, regulator, USB bridge, LED), not the chip alone: what a battery would feed.

## Reproduce

```bash
# firmware on the board (chip_test.py for SYMBEX, the benchmark's TFLite script for INT8), meter zeroed, then:
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 27648 --seconds 600
```
