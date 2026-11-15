# cnn_cifar_resnet8 — energy per inference on the ESP32-S3

**Question.** Being faster is one thing; does the fully binary ResNet-8 also use less energy per image than the
official MLPerf Tiny INT8 model running on TFLite Micro?

**Answer.** Yes. On the ESP32-S3, one CIFAR-10 image costs **7.2 mJ with SYMBEX against 47.7 mJ with TFLite Micro
INT8: 6.6× less energy**. Counting only the energy above the board's idle draw, SYMBEX uses about **10× less** (1.7 vs
17.5 mJ). The meter has coarse readings, so this second number has a wide range (6-18×).

## What was measured

- **Board:** ESP32-S3 with the weights in flash, powered and read through its native USB port.
- **Meter:** a USB power meter between the PC and the board. It shows V, A, mAh and Wh and is read by eye.
- **Models:**
  - SYMBEX: `int8stem_pool2`, seed 42, vertical engine (default), swar popcount. Exact against its simulation on the
    chip, 100 %.
  - TFLite Micro: the official MLPerf ResNet-8 with ESP-NN kernels, the same firmware as `results/tflite/tflite_s3.json`.
- **Three phases of 10 minutes each:**
  - TFLite busy;
  - SYMBEX busy;
  - idle (the SYMBEX firmware waiting on the serial port).

  For each phase the meter was zeroed, then `tools/energy_loop.py` started the loop. The firmware runs the same image
  again and again until the time is up, then reports how many inferences it did.
- **Energy:** mAh × V × 3.6 J. Per inference, that energy is divided by the number of runs. "Above idle" subtracts
  the idle phase's mAh before dividing.

## Result

| Phase | Inferences in 600 s | ms each | Meter (mAh / Wh) | Current seen | Power | **mJ per inference** | above idle |
|---|---|---|---|---|---|---|---|
| TFLite Micro INT8 | 11,626 | 51.61 | 30 / 0.15 | 0.17-0.18 A (peaks 0.20) | 0.92 W | **47.7** | 17.5 |
| SYMBEX (vertical) | 63,736 | 9.41 | 25 / 0.13 | 0.14-0.15 A (low 0.13) | 0.77 W | **7.2** | 1.74 |
| Idle | — | — | 19 / 0.09 | 0.11 A | 0.58 W | — | — |

- Voltage was 5.13 V in every phase.
- The two Wh readings agree with the mAh readings: 0.15 Wh = 540 J, against 554 J from 30 mAh.
- **Ratio, total energy:** 6.6×. With every reading off by 1 mAh in the worst direction it is still 6.1-7.1×.
- **Ratio, above idle:** 10×, worst-case range 6.2-17.8×. Above idle the difference is only 6-11 mAh, so ±1 mAh
  matters a lot.
- **Two things explain the gap:**
  - SYMBEX finishes each image 5.5× sooner.
  - The board also draws less while SYMBEX runs: 0.77 W against 0.92 W. Likely cause, not measured: the binary
    engine keeps fewer multipliers and less memory traffic busy.
- **The loop's latency matches the 1,000-image chip runs:**
  - SYMBEX: 9.41 ms in the loop against 9.72 ms per image. The loop repeats one image, which keeps the cache warm.
  - TFLite: 51.61 ms in the loop against 51.96 ms.
- **Seeds 43 and 44 (calculated, NOT measured):** SYMBEX's measured power × each seed's chip latency
  (`results/symbex/symbex_seeds_s3_swar_w1_int8stem_pool2_vertical.json`) gives 7.48 / 7.46 / 7.46 mJ. All three seeds
  have the same shape, engine and code, and their latencies differ by 0.3 %.

The raw readings and every derived number are in `results/energy/energy_s3.json`.

## Limits (to keep in mind)

- **One 10-minute run per phase, on one board.** The board is always slightly warm, which raises its idle draw. A
  second board would be a cleaner check.
- **The meter is read by eye and counts whole mAh.** The total-energy ratio is solid (±7 %); the above-idle ratio
  is not.
- **The power is measured at the USB port, so it is the whole board:** chip, regulator, USB bridge and LEDs, not the
  chip alone. This is what a battery would feed.
- **Accuracy differs:** TFLite INT8 88.0 % against SYMBEX 76.9 % on the same 1,000 chip images (see README). This is the energy of each
  model's own trade-off, not of equal accuracy.

## Reproduce

```bash
# SYMBEX firmware on the board (chip_test.py), then, with the meter zeroed:
python tools/energy_loop.py --port /dev/ttyACM0 --firmware symbex \
    --input <export>/validation_x.u8 --n-inputs 3072 --seconds 600
# TFLite Micro firmware: python resnet8_tflite.py --board s3 --port /dev/ttyACM0 --usb, then:
python tools/energy_loop.py --port /dev/ttyACM0 --firmware tflite \
    --input <export>/validation_x.u8 --n-inputs 3072 --seconds 600
# Idle: firmware on the board, no traffic
python tools/energy_loop.py --idle --seconds 600
```

`chip_test.py --energy S` does the same three phases in one go for CDT exports: it asks for the meter readings and
writes the result into the chip JSON.

## Future work

- A second board, and the classic ESP32 and the Pico 2.
- The other engines (compact, prepared) and RAM weights.
- A meter the PC can read, for finer numbers than whole mAh.
