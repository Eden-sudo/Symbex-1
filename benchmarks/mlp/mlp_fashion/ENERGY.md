# mlp_fashion — energy per inference (classic ESP32)

**Question.** Does the binary 784-128-512-10 MLP also use less energy per image than the same network on TFLite Micro INT8?

**Answer.** Yes. On the classic ESP32 one image costs **0.91 mJ with SYMBEX against 10.3 mJ with TFLite Micro INT8: 11.3× less energy** (worst case with ±1 mAh: 9.5-13.4×). SYMBEX is 13.4× faster and the board draws about the same power with both (0.39 against 0.33 W): the energy gain comes from speed.

## What was measured

- **Board:** classic ESP32 (no SIMD), weights in flash, powered and read through its USB-UART port.
- **Meter:** USB power meter between the PC and the board (V, A, mAh, Wh), read by eye; whole mAh.
- **SYMBEX:** runs/sizing/h128_512_s42, swar popcount, flash; exact against its simulation on the chip, 100 %.
- **TFLite Micro:** INT8, runs/tflite/fashion_int8_h128_512.tflite (same shape, INT8 QAT), ESP-NN, flash.
- **Phases:** one 10-minute run per method, meter zeroed before each. `tools/energy_loop.py` keeps the model busy on one image and counts the inferences. Idle (8 mAh in 600 s, 0.25 W) is the Nov 14 reading of the same board (`../../conv2d/cnn_cifar_resnet8/ENERGY.md`).
- **Energy:** mAh × V × 3.6 J; the idle rate is subtracted for the seconds the meter timer ran past 600 s. Per inference = energy / runs. "Above idle" subtracts the idle 8 mAh first.

## Result

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Current seen | Power | **mJ per inference** | above idle |
|---|---|---|---|---|---|---|---|
| TFLite Micro INT8 | 19,289 | 31.11 | 11 / 0.05 / 10:20 | 0.06 A, stable | 0.33 W | **10.3** | 2.62 |
| SYMBEX | 258,056 | 2.33 | 13 / 0.07 / 10:20 | 0.05-0.07 A, peaks 0.08 A | 0.39 W | **0.913** | 0.339 |

- Voltage 5.14 V. Accuracy differs: binary 87.5 % against INT8 QAT 89.8 % (same shape, labels only, 3 seeds; README).
- Above idle: 7.7×; only a few mAh above idle, so ±1 mAh moves it a lot.
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
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 784 --seconds 600
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware tflite --input <export>/validation_x.u8 --n-inputs 784 --seconds 600
```
