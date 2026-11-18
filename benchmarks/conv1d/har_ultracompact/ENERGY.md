# har_ultracompact — energy per inference (classic ESP32, ESP32-S3)

**Question.** Does the binary HAR max network (int8 first layer) also use less energy per window than the same network on TFLite Micro INT8?

**Answer.** Yes. On the classic ESP32 one window costs **5.46 mJ with SYMBEX against 47.7 mJ with TFLite Micro INT8: 8.7× less energy** (worst case with ±1 mAh: 7.5-10.1×). SYMBEX is 8.7× faster and the board draws about the same power with both (0.42 against 0.42 W): the energy gain comes from speed.

**On the ESP32-S3** one window costs **2.97 mJ with SYMBEX against 5.43 mJ with TFLite Micro INT8: 1.8× less energy** (worst case with ±1 mAh: 1.7-2.0×). SYMBEX is 1.77× faster and the board draws about the same power with both (0.76 against 0.79 W): the gain comes from speed.

## What was measured

- **Board:** classic ESP32 (no SIMD), weights in flash, powered and read through its USB-UART port.
- **Meter:** USB power meter between the PC and the board (V, A, mAh, Wh), read by eye; whole mAh.
- **SYMBEX:** runs/max_int8_e100_s42, lut16 popcount, flash; exact against its simulation on the chip, 100 %.
- **TFLite Micro:** INT8, max seed 42 (same architecture, same recipe), ESP-NN, flash.
- **Phases:** one 10-minute run per method, meter zeroed before each. `tools/energy_loop.py` keeps the model busy on one window and counts the inferences. Idle (8 mAh in 600 s, 0.25 W) is the Nov 14 reading of the same board (`../../conv2d/cnn_cifar_resnet8/ENERGY.md`).
- **Energy:** mAh × V × 3.6 J; the idle rate is subtracted for the seconds the meter timer ran past 600 s. Per inference = energy / runs. "Above idle" subtracts the idle 8 mAh first.

## Result

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Current seen | Power | **mJ per inference** | above idle |
|---|---|---|---|---|---|---|---|
| TFLite Micro INT8 | 5,306 | 113.08 | 14 / 0.07 / 10:25 | ~0.08 A, up and down | 0.42 W | **47.7** | 19.8 |
| SYMBEX | 46,136 | 13.01 | 14 / 0.07 / 10:30 | 0.08-0.09 A | 0.42 W | **5.46** | 2.25 |

- Voltage 5.14 V. Accuracy differs: on HAR max INT8 is about 9 points more accurate (README); this energy buys speed, not equal accuracy.
- Above idle: 8.8×; only a few mAh above idle, so ±1 mAh moves it a lot.
- Raw readings and every derived number: `results/energy/energy_esp32.json`.

## ESP32-S3

Same method on the ESP32-S3 (Nov 15), weights in flash, powered and read through its native USB port, through the meter's **USB-A** port (see Limits).

- **SYMBEX:** runs/max_int8_e100_s42, standard engine, lut16, flash; exact against its simulation on the chip, 100 %.
- **TFLite Micro:** max seed 42 (same architecture, same recipe), ESP-NN, flash.
- Idle (19 mAh in 600 s, 0.59 W) is the Nov 14 reading of the same board and meter port (`../../conv2d/cnn_cifar_resnet8/ENERGY.md`); only its rate is used, to subtract the seconds the meter timer ran past 600 s.

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Power | **mJ per inference** |
|---|---|---|---|---|---|
| TFLite Micro INT8 | 86,740 | 6.92 | 26 / 0.13 / 10:15 | 0.79 W | **5.43** |
| SYMBEX | 153,544 | 3.91 | 25 / 0.12 / 10:10 | 0.76 W | **2.97** |

- Voltage 5.13 V. Ratio, total energy: **1.8×** (range 1.7-2.0×); speed 1.77×.
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
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 453 --seconds 600
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware tflite --input <export>/validation_x.u8 --n-inputs 453 --seconds 600
# ESP32-S3: chip_test.py ... --board s3 --usb (SYMBEX) or the TFLite script with --board s3 --usb, then:
python tools/energy_loop.py --port /dev/ttyACM0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 453 --seconds 600
```
