# The 260K tiny_llm on a second classic ESP32 (#210 reproduction)

Eden's 260K runs (`runs/binary_v512d64l5kv4_s4*`, `runs/binary_v512d64l5kv4_lr3e-3_20k_k1trv2_s4*`,
`runs/fp32_v512d64l5kv4_lr3e-3_20k_s4*`, all 35 sha256 matching his zip) run on a second classic ESP32 with the
unmodified `chip_run.py` / `chip_fp32.py`. One folder per PlatformIO platform, so these files never mix with the
`../chip_esp32_*` results (espressif32@7.1.3). The JSONs are the scripts' own output, unchanged; only the 7.1.3 ones
carry `"platform"` (added in 4c0103c).

| folder | platform (Arduino-ESP32 / ESP-IDF) | PR commit | runs |
|---|---|---|---|
| `espressif32-8.0.0/` | 8.0.0 (3.3.12 / 5.5.5), unpinned default | 8169461 | 9: int8 cache, `k1trv2`, fp32; seeds 42-44 |
| `espressif32-6.13.0/` | 6.13.0 (2.0.17 / 4.4), pinned locally | 8169461 | 2: int8 cache and fp32, seed 42 |
| `espressif32-7.1.3/` | 7.1.3 (2.0.17 / 4.4), the pin of 4c0103c | 4c0103c | 2: int8 cache and fp32, seed 42 |

**Exact on every run:** binary 1024 / 1024 window outputs and 120 / 120 story tokens each; fp32 120 / 120 story
tokens each; every story text equal to the one in the matching `../chip_esp32_*.json`.

**Speed and RAM depend on the core** (story tokens/s; ms per token of the binary int8-cache model; seed 42 unless
3 seeds are given):

| | `../` (Eden, 7.1.3) | 8.0.0 | 6.13.0 | 7.1.3 |
|---|---|---|---|---|
| binary int8 cache, tokens/s | 40.15 ± 0.06 | 39.52 ± 0.04 (3 seeds) | 40.60 | 40.60 |
| binary `k1trv2`, tokens/s | 35.88 ± 0.08 | 35.44 ± 0.07 (3 seeds) | not run | not run |
| fp32 `run.c`, tokens/s | 6.48 ± 0.00 | 6.71 ± 0.00 (3 seeds) | 6.48 | 6.48 |
| binary / fp32 | 6.20x | 5.89x | 6.26x | 6.26x |
| ms per token at position 0 / 511 | 19.85 / 49.07 | 19.86 / 53.96 | 19.74 / 48.79 | 19.74 / 48.79 |
| free RAM at start / after load, binary | 350,764 / 71,672 | 331,076 / 34,420 | 350,764 / 71,672 | 350,764 / 71,672 |
| run.c state: bytes taken from the heap | 171,232 | 172,712 | 171,232 | 171,232 |

On 8.0.0 the first token costs the same and the gap grows with the position (attention side); the cause is not
measured (the classic ESP32 firmware has no part timers). 6.13.0 and 7.1.3 give the same numbers to 0.01 ms.

**`faults/`, the first board runs of 831ffda and 05e5b6b** (8169461, platform 8.0.0, only a generated header edited,
`scripts/fault_test.sh`):

- `fault_l2c.log`: `chip_l2c/src/gen_data.h` `GEN_IDS 127 -> 512` (run.c's state at context 512 = 676,192 bytes):
  the board prints `ERROR: run.c needs 676192 bytes of RAM, 331472 free`, `chip_fp32.py` raises.
- `fault_step.log`: `chip_mcu/src/check_data.h` `check_tokens[0][5] = 512` (= vocab): `H:0,0..4` ok, then
  `ERROR: step`, `chip_run.py` raises.

**How it was run** (`scripts/`): our board's CP2102 drops transfers at pio's 460800 baud, and the first port open
after a pio upload can jam it. So each run is: `chip_run.py --no-upload` (writes the data headers; the old firmware
stops on the ID check), `pio run` (build only) + esptool at 115200 until all 4 parts verify, a throwaway port open,
then `chip_run.py --no-upload` reads the board. The firmware is the one `chip_run.py` would build. One step-1 line per
log is the expected ID-check stop. The 6.13.0 runs have no logs (overwritten by later runs; the JSONs are complete).

Board: ESP32-D0WD-V3 rev 3.1, 240 MHz, one core used, 4 MB flash (DIO, 40 MHz), no PSRAM, CP2102. PC: Python 3.14,
torch 2.14.1+cpu (no GPU; `chip_run.py` needed a local CPU fallback at 8169461, in the PR since 4c0103c). Energy not
measured.
