# mlp_fashion — energy per inference (classic ESP32, ESP32-S3)

**Question.** Does the binary 784-128-512-10 MLP also use less energy per image than the same network on TFLite Micro INT8?

**Answer.** Yes. On the classic ESP32 one image costs **0.91 mJ with SYMBEX against 10.3 mJ with TFLite Micro INT8: 11.3× less energy** (worst case with ±1 mAh: 9.5-13.4×). SYMBEX is 13.4× faster and the board draws about the same power with both (0.39 against 0.33 W): the energy gain comes from speed.

**On the ESP32-S3** one image costs **0.323 mJ with SYMBEX against 8.18 mJ with TFLite Micro INT8: 25.4× less energy** (worst case with ±1 mAh: 23.3-27.6×). SYMBEX is 28.5× faster; here it draws more power than TFLite (0.78 against 0.69 W) and still uses far less energy.

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

## ESP32-S3

Same method on the ESP32-S3 (Nov 15), weights in flash, powered and read through its native USB port, through the meter's **USB-A** port (see Limits).

- **SYMBEX:** runs/sizing/h128_512_s42, standard engine, lut16, flash; exact against its simulation on the chip, 100 %.
- **TFLite Micro:** runs/tflite/fashion_int8_h128_512.tflite (same shape, INT8 QAT), ESP-NN, flash.
- Idle (19 mAh in 600 s, 0.59 W) is the Nov 14 reading of the same board and meter port (`../../conv2d/cnn_cifar_resnet8/ENERGY.md`); only its rate is used, to subtract the seconds the meter timer ran past 600 s.

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Power | **mJ per inference** |
|---|---|---|---|---|---|
| TFLite Micro INT8 | 50,845 | 11.80 | 23 / 0.11 / 10:15 | 0.69 W | **8.18** |
| SYMBEX | 1,447,344 | 0.415 | 26 / 0.13 / 10:23 | 0.78 W | **0.323** |

- Voltage 5.13 V. Ratio, total energy: **25.4×** (range 23.3-27.6×); speed 28.46×.

### Weights in RAM (S3)

**Question.** TFLite drew less power than SYMBEX in flash (0.69 against 0.78 W). Is that because it waits on the flash? Then, with its weights in RAM, it should draw more.

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh / timer) | Power | **mJ per inference** |
|---|---|---|---|---|---|
| TFLite Micro INT8, RAM | 680,380 | 0.882 | 25 / 0.13 / 10:00 | 0.77 W | **0.679** |
| SYMBEX, RAM | 1,447,336 | 0.415 | 30 / 0.15 / 10:07 | 0.92 W | **0.38** |

**Answer.** Not supported. With the weights in RAM **both** drew more: TFLite 0.69 → 0.77 W (and 13× faster), SYMBEX 0.78 → 0.92 W at exactly the same 0.415 ms. Why is not measured. In RAM against RAM SYMBEX uses 1.8× less energy (1.7-1.9×); SYMBEX in flash against TFLite in RAM, 2.1×. A first SYMBEX RAM run was read 380 s late and was discarded.

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
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 784 --seconds 600
python tools/energy_loop.py --port /dev/ttyUSB0 --firmware tflite --input <export>/validation_x.u8 --n-inputs 784 --seconds 600
# ESP32-S3: chip_test.py ... --board s3 --usb (SYMBEX) or the TFLite script with --board s3 --usb, then:
python tools/energy_loop.py --port /dev/ttyACM0 --firmware symbex --input <export>/validation_x.u8 --n-inputs 784 --seconds 600
```
