# edge_llm_probe

A hardware probe that **measures** the two numbers that decide whether a natively trained 1-bit LLM (BitNet-style:
+-1 weights, low-bit activations; for example a 15-30M-parameter TinyStories model with a ~4k vocabulary) can run
usefully on an **ESP32-S3** and on a classic **ESP32** (AI-Thinker ESP32-CAM):

1. **memory read bandwidth per tier**: internal SRAM, PSRAM, memory-mapped flash, SD card;
2. **cycles per weight of 1-bit matrix-vector kernels** (`kernels/onebit_gemv.c`), per weight location.

It also contains an energy schedule for a USB power meter, a camera smoke test, and a host-side parser that projects
tokens/s. The design reasoning is in [`docs/llm_1bit_embedded.md`](../../docs/llm_1bit_embedded.md).

> **Status: nothing has been measured on silicon yet.** The firmware builds for both boards and the kernels are
> bit-exact against numpy on the host (`tests/test_edge_llm_kernels.py`); every table below says "pending". The parser's
> PROJECTION is arithmetic on measurements, never a measurement.

| What | File |
|---|---|
| kernels (portable C99): `gemv_ref`, `gemv_mask`, `gemv_lut4`, `gemv_bitserial4`, `gemv_xnor` | `kernels/onebit_gemv.{c,h}` |
| firmware (Arduino framework, PlatformIO) | `platformio.ini`, `partitions_probe.csv`, `src/` |
| log -> JSON + markdown + projection | `parse_log.py` |
| host tests | `tests/test_edge_llm_kernels.py`, `tests/test_edge_llm_parse.py` |

## What the firmware prints

Machine-readable lines (comma separated) plus human comments starting with `#`:

| Line | Meaning |
|---|---|
| `INFO,<key>,<value>` | chip, revision, cores, CPU MHz, free heap, PSRAM and flash size, SD card or `none`, camera `ok` / `fail:<err>` / `skipped` |
| `BW,<tier>,<test>,<bytes>,<MBps_best>,<MBps_median>` | sequential 32-bit read-sum (and `memcpy_to_sram`, `file_read_buf4k/32k`), 7 repetitions (SD: 3). MB = 10^6 bytes |
| `KERN,<kernel>,<mem>,<M>,<N>,<cycles_per_weight>,<us_per_call>,<OK\|FAIL>` | one kernel, weights in `sram`, `psram` or `flash` (memory-mapped), for 256x1024, 1024x1024 and 2048x2048 where memory allows. `OK` = identical integers to `gemv_ref` on the chip |
| `CAM,<w>,<h>,<jpeg_bytes>,<fps>` | camera smoke test |
| `PHASE,<n>,<name>,start` / `PHASE,<n>,<name>,end,<iterations>` | energy schedule |

Kernel lines: `lut4_build` (once per input vector) and `lut4_lookup` are timed separately, `lut4` is their sum,
`lut4_2core_lookup` splits the rows over both cores (wall-clock cycles, FreeRTOS task on the other core + semaphores),
`bitserial4_pack` is the activation preparation of `bitserial4`, `xnor` is the 1-bit-activation lower bound. Cycles per
weight = cycles / (M x N), so the per-vector build/pack cost is amortised over the M rows. Weights in `psram` and
`flash` are **cache-cold** (the data cache is flushed before each timed call), like a layer last used a whole token ago;
`sram` has no cache in the path. Reported value: the best of 3 calls. Activations, the LUT and the outputs are always
in internal SRAM. `ref` is checked against the independent `mask` algorithm on the chip and against numpy on the host.

The probe partition (`probe`, 1 MB + a 4 KB tag sector) is filled **once** with a reproducible pseudo-random pattern
if its tag is missing; later boots do not rewrite flash. The SD test file (8 MB) is written once if absent.

## Build and flash

Needs PlatformIO. The first build downloads the toolchains.

```
cd benchmarks/edge_llm_probe
pio run -e esp32s3                     # or: -e esp32cam
pio run -e esp32s3 -t upload -t monitor
```

No `upload_port` is set: PlatformIO picks the only serial device, or pass `--upload-port /dev/ttyUSB0`.
The boot prints for 3 s of start delay first; **press reset after opening the monitor** so the whole log is captured.

### ESP32-S3 (esp32-s3-devkitc-1 definition, octal PSRAM)

`board_build.arduino.memory_type = qio_opi` and `psram_type = opi` match N8R8/N16R8 modules. The log goes to UART0
(the USB-UART connector). For the native USB connector add `-DARDUINO_USB_CDC_ON_BOOT=1` to the env's `build_flags`.
The board definition assumes 8 MB flash; for 16 MB add `board_upload.flash_size = 16MB` (the partition table needs only
4 MB; the reported `flash_size` is the real chip size).

### ESP32-CAM (AI-Thinker, classic ESP32, 4 MB PSRAM mapped)

There is no USB. Use a USB-TTL adapter (3.3 V logic): adapter TX -> board U0R, RX -> U0T, GND -> GND, and power the
board from a **5 V** supply with enough current (the camera and PSRAM brown out on a weak 3.3 V adapter pin).
**To flash: connect IO0 to GND**, press reset, run `pio run -e esp32cam -t upload`, then **disconnect IO0 from GND**
and press reset again to run; open the monitor (`pio device monitor`). The 4 MB PSRAM allocation test (`psram_4MB`)
will probably be skipped on this board (the heap holds slightly less than 4 MB); the log says so.

## Saving the log and the report

```
pio device monitor -e esp32s3 | tee probe_s3.log                 # or: pio device monitor -e esp32s3 --filter log2file
python parse_log.py probe_s3.log                                 # writes results/<chip>_<timestamp>.json
python parse_log.py --port /dev/ttyUSB0 --duration 600           # capture (needs pyserial) and parse
```

The monitor loops forever through the energy schedule; the BW and KERN sections are complete after a few minutes
(2048x2048 kernels from PSRAM/flash dominate). If the log holds several boots, the parser uses the last.

`parse_log.py` prints the measurements, then a section titled **PROJECTION (not a measurement)**: for 15M, 30M, 110M
and 200M 1-bit parameters (bytes = params / 8), for every tier that holds the model (SRAM about 300 KB usable, PSRAM
minus 1 MB, flash minus 3 MB, SD only if nothing else fits):
`t_mem = bytes / BW`, `t_cpu = params x best verified cycles-per-weight / f_cpu`; optimistic s/token = max(t_mem, t_cpu),
pessimistic = t_mem + t_cpu. The assumptions are printed with the table. Attention, KV-cache traffic and activation
quantisation are not included.

With meter readings:

```
python parse_log.py probe_s3.log --watts idle=1.10 sram-read=1.25 lut4-gemv=1.62 lut4-2core=1.95 mask-gemv=1.60
```

adds a projected energy per token = watts x projected s/token (total, and above idle).

## Measuring power with a USB meter

The firmware runs an **energy schedule** forever: 1 idle, 2 sram-read, 3 psram-read, 4 flash-read, 5 sd-read (if an SD
card is present), 6 lut4-gemv (PSRAM weights, 1 core), 7 lut4-2core, 8 mask-gemv, 9 camera (if the camera works).
Every phase runs one steady workload for `PROBE_PHASE_SECONDS` (default 20 s, a build flag) and prints
`PHASE,<n>,<name>,start` ... `PHASE,<n>,<name>,end,<iterations>` plus a countdown (`# phase N name: 15 s left, read the meter now`) every
5 s. WiFi and Bluetooth are never started.

1. Put the USB meter **between the USB supply and the board** (ESP32-CAM: in the 5 V line to the adapter/board).
2. For each phase, note the watts in the **last 10 s** (slow meters need ~20 s to settle; raise
   `-DPROBE_PHASE_SECONDS=40` if yours is slower).
3. **Record the idle baseline** and report both the **total** and the **above-idle** figure.
4. These are **board-level** numbers: they include the LDO/DC-DC, the USB-UART chip, LEDs, PSRAM and flash, not only the
   chip. Do not quote them as chip power. The ESP32-CAM white LED (GPIO4) lights in SD 4-bit mode; a camera left powered
   changes every phase, so the camera is initialised only for its own phase.
5. For sample-level data use an INA219/INA226 on a second MCU, or a scope on a shunt synced to `MARKER_GPIO`:
   `-DPROBE_MARKER_GPIO=<pin>` drives the pin high during every non-idle phase (default -1 = disabled).

Reading the schedule: sram/psram/flash/sd phases show the cost of moving bytes; lut4-gemv vs lut4-2core shows the second
core's cost; mask-gemv is today's SYMBEX first-layer approach for comparison; camera shows what the sensor adds.

## Camera smoke test

At boot the camera is initialised (QVGA JPEG), 30 frames are grabbed and the camera is released. `INFO,camera,ok` plus a
`CAM,...` line means the pin map, sensor and PSRAM frame buffers work. `INFO,camera,fail:0x<code>` is an `esp_err_t`
from `esp_camera_init` (commonly wrong pins or an unpowered sensor). It says nothing about image quality and is not
part of the LLM measurement; it exists because both boards carry a camera and the energy phase 9 needs one.

## Pin and board overrides

All in `src/probe_config.h`; override with `-D` in `build_flags`:

| Macro | Default | Meaning |
|---|---|---|
| `PROBE_SD_CLK` / `PROBE_SD_CMD` / `PROBE_SD_D0` | 39 / 38 / 40 | ESP32-S3 SD_MMC pins, 1-bit mode (Freenove ESP32-S3-WROOM CAM and ESP32-S3-EYE style boards). `PROBE_SD_CLK=-1` disables SD on an S3 board without a slot |
| `PROBE_SD_FREQ_KHZ` | 40000 | SDMMC clock |
| `PROBE_SD_TEST_4BIT` | 1 on classic ESP32, 0 on S3 | test 4-bit mode too. **It drives GPIO4 = the flash LED** of the ESP32-CAM |
| `PROBE_CAMERA_AI_THINKER`, `PROBE_CAMERA_ESP32S3_EYE`, `PROBE_CAMERA_NONE` | per env | camera pin map (`src/camera_pins.h`, copied from the arduino-esp32 CameraWebServer example) |
| `PROBE_MARKER_GPIO` | -1 | GPIO high during non-idle phases |
| `PROBE_PHASE_SECONDS` | 20 | seconds per energy phase |
| `PROBE_CPU_MHZ` | 240 | CPU clock set at boot (the log's cycle counts are at this clock) |

The classic ESP32 has a fixed SDMMC slot (CLK 14, CMD 15, D0 2, D1 4, D2 12, D3 13): no pin flags. For another S3 board:
set the three SD pins, and either pick the matching camera map or add a block to `camera_pins.h` from your board's
schematic. The Freenove ESP32-S3-WROOM CAM is documented as using the ESP32-S3-EYE GPIOs; the S3 pin defaults here were
**not** checked against a schematic: if SD or the camera fails on your board, look at its pin table first.

## Results

| Board | Memory BW | Kernels (cycles/weight) | Power per phase | Camera |
|---|---|---|---|---|
| ESP32-S3 | measured, see below | measured, see below | pending: needs the USB meter | not on this board |
| ESP32-CAM | pending: run on hardware | pending: run on hardware | pending: run on hardware | pending: run on hardware |

Commit the JSON files produced by `parse_log.py` under `results/` once they exist.

### ESP32-S3-DevKitC-1 N16R8 (October 2026)

Env `esp32s3_devkit` (16 MB flash, 8 MB octal PSRAM, no SD slot, no camera, log on the native USB port), 240 MHz.
Raw log: `results/capture_s3_devkit.log`; parsed: `results/esp32s3_20261027_010134.json`. Every kernel line is `OK`
(identical integers to `gemv_ref` on the chip, 72 of 72).

Sequential read bandwidth (MB = 10^6 bytes, best of 7):

| Tier | read32_seq | memcpy to SRAM |
|---|---|---|
| SRAM 16 KB / 128 KB | 422 / 424 MB/s | 379 / 380 MB/s |
| PSRAM 16 KB (fits the cache) | 422 MB/s | 379 MB/s |
| PSRAM 1 MB / 4 MB | 61.8 / 61.9 MB/s | 60.4 / 60.4 MB/s |
| flash, memory-mapped, 1 MB | 26.5 MB/s | 26.5 MB/s |

Cycles per weight, 2048 x 2048 (256 x 1024 for SRAM; PSRAM and flash cache-cold, best of 3):

| Kernel | SRAM | PSRAM | flash |
|---|---|---|---|
| `ref` | 16.12 | 16.43 | 16.56 |
| `mask` | 10.44 | 10.74 | 10.87 |
| `lut4` (build + lookup) | 1.89 | 1.94 | 2.07 |
| `lut4_2core_lookup` (wall clock, both cores) | 0.81 | 0.96 | 1.11 |
| `bitserial4` | 2.67 | 2.98 | 3.11 |
| `xnor` (1-bit activations, lower bound) | 0.68 | 0.99 | 1.20 |

Against the kill criteria of the design note (section 9: more than 8 cycles per weight, or less than ~7 MB/s, re-plan):
the best kernel is 0.96-1.11 cycles per weight and PSRAM / flash read at 62 / 26 MB/s, so the 15M model is not ruled out.
`parse_log.py` PROJECTS (arithmetic, not a measurement; no attention, KV cache or int8 head) 7-17 tok/s for 15M and
3.6-8.4 tok/s for 30M. `mask` is the portable C of the multi-bit-input path; lib_symbex's PIE version of it is about
0.65 cycles per weight from RAM (design note, section 2), not measured by this probe.

The first run of `read32_seq` read each buffer once per call instead of once per pass (GCC treated the read as pure
across passes): fixed with a memory barrier in `bw_read_buffer`; the numbers above are after the fix.

## Host tests

```
python -m pytest -q tests/test_edge_llm_kernels.py tests/test_edge_llm_parse.py     # from the repository root
```
