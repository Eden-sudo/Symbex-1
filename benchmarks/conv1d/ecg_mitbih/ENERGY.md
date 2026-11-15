# ecg_mitbih — energy per inference (classic ESP32)

**Question.** Does the binary ECG net m (RR channels) also use less energy per beat than the same network on TFLite Micro INT8?

**Answer.** Yes. On the classic ESP32 one beat costs **5.04 mJ with SYMBEX against 40.1 mJ with TFLite Micro INT8: 8.0× less energy** (worst case with ±1 mAh: 6.9-9.2×). SYMBEX is 8.1× faster and the board draws about the same power with both (0.43 against 0.43 W): the energy gain comes from speed.

## What was measured

- **Board:** classic ESP32 (no SIMD), weights in flash, powered and read through its USB-UART port.
- **Meter:** USB power meter between the PC and the board (V, A, mAh, Wh), read by eye; whole mAh.
- **SYMBEX:** runs/m_binary_e100_s42_rr, swar popcount, flash; exact against its simulation on the chip, 100 %.
- **TFLite Micro:** INT8, m seed 42 with RR (same architecture, same recipe), ESP-NN, flash.
- **Phases:** one 10-minute run per method, meter zeroed before each. `tools/energy_loop.py` keeps the model busy on one beat and counts the inferences. Idle (8 mAh in 600 s, 0.25 W) is the Nov 14 reading of the same board (`../../conv2d/cnn_cifar_resnet8/ENERGY.md`).
- **Energy:** mAh × V × 3.6 J; the idle rate is subtracted for the seconds the meter timer ran past 600 s. Per inference = energy / runs. "Above idle" subtracts the idle 8 mAh first.

## Result

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Current seen | Power | **mJ per inference** | above idle |
|---|---|---|---|---|---|---|---|
| TFLite Micro INT8 | 6,374 | 94.14 | 14 / 0.07 / 10:15 | not noted | 0.43 W | **40.1** | 16.8 |
| SYMBEX | 51,440 | 11.66 | 14 / 0.07 / 10:00 | not noted | 0.43 W | **5.04** | 2.16 |

- Voltage 5.14 V. Accuracy: balanced accuracy of binary and INT8 is tied within the standard deviation on net m (README).
- Above idle: 7.8×; only a few mAh above idle, so ±1 mAh moves it a lot.
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
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 1024 --seconds 600
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware tflite --input <export>/validation_x.u8 --n-inputs 1024 --seconds 600
```
