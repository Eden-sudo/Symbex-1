# fomo_voc_person — energy per inference (classic ESP32, ESP32-S3)

**Question.** How much energy does the binary FOMO person detector use per frame on the classic ESP32?

**Answer.** **96 mJ per frame** (226 ms, 0.43 W). TFLite Micro INT8 of this detector does not fit the classic ESP32 (its arena is about 230 KB), so there is no INT8 number on this board: SYMBEX is the only one of the two that runs here.

**On the ESP32-S3** one frame costs **96.7 mJ with SYMBEX against 192.5 mJ with TFLite Micro INT8: 2.0× less energy** (worst case with ±1 mAh: 1.8-2.2×). SYMBEX is 1.96× faster and the board draws about the same power with both (0.74 against 0.75 W): the gain comes from speed. On the S3 the INT8 detector fits, so both run there.

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

## ESP32-S3

Same method on the ESP32-S3 (Nov 15), weights in flash, powered and read through its native USB port, through the meter's **USB-A** port (see Limits).

- **SYMBEX:** runs/c32-32-128_b1-2-2_w40_e100_s42, standard engine, lut16, flash; exact against its simulation on the chip, 100 %.
- **TFLite Micro:** models/fomo_int8_c32-32-128_b1-2-2_w20_e60.tflite (same shape, the README's INT8 detector), ESP-NN, flash; 95 % equal to the PC interpreter on 20 images (88 % on 100 in the README: ESP-NN rounding).
- Idle (19 mAh in 600 s, 0.59 W) is the Nov 14 reading of the same board and meter port (`../../conv2d/cnn_cifar_resnet8/ENERGY.md`); only its rate is used, to subtract the seconds the meter timer ran past 600 s.

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Power | **mJ per inference** |
|---|---|---|---|---|---|
| TFLite Micro INT8 | 2,338 | 256.65 | 25 / 0.12 / 10:20 | 0.75 W | **192.5** |
| SYMBEX | 4,592 | 130.75 | 25 / 0.12 / 10:30 | 0.74 W | **96.7** |

- Voltage 5.13 V. Ratio, total energy: **2.0×** (range 1.8-2.2×); speed 1.96×.
- On the classic ESP32 the INT8 detector does not fit (energy_esp32.json); on the S3 it does.
- Raw readings and every derived number: `results/energy/energy_s3.json`.

## Limits

- One 10-minute run per method on one board; the meter counts whole mAh (±1 mAh per reading).
- The idle draw was not measured again: the Nov 14 reading is reused. The classic ESP32 board does not warm up like
  the S3 (at most slightly warm by touch), so its idle draw is not expected to drift much; the total-energy ratios do
  not depend on it.
- **ESP32-S3:** the board reads about 20 % more through the meter's USB-C port than through its USB-A port (idle 0.14 against 0.11 A); the classic ESP32 and the Pico 2 do not. The cause is not known (probably the board's USB circuit); every S3 number uses the USB-A port, the lower reading. The S3 board is always warm, which raises its idle draw; its above-idle numbers (in the JSON) are not used.
- Whole board at the USB port (chip, regulator, USB bridge, LED), not the chip alone: what a battery would feed.

## Reproduce

```bash
# firmware on the board (chip_test.py for SYMBEX, the benchmark's TFLite script for INT8), meter zeroed, then:
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 27648 --seconds 600
# ESP32-S3: chip_test.py ... --board s3 --usb (SYMBEX) or the TFLite script with --board s3 --usb, then:
python tools/energy_loop.py --port /dev/ttyACM0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 27648 --seconds 600
```
