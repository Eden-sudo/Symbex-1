# N16R8 chat board preparation — host evidence only

This lane adds planning and integration scaffolding, not runnable ESP32 firmware.
Target: ESP32-S3 N16R8, 16 MiB flash and 8 MiB PSRAM. Model flash starts at
`0x310000`, matching the existing `chip_run.py`. Eden owns physical board runs.
No board access, PlatformIO, firmware builds, downloads, push or PR publication
were performed for this lane. Existing C model, chip main and runner are unchanged.

## Flash plan

`board_flash_plan.py` accepts an ordered JSON object of complete container sizes,
with `model` first. Include metadata and alignment internal to each container.
It preserves NVS at `0x9000` and the factory app at `0x10000` (3 MiB), rounds
each data partition to 4096 bytes, and rejects overlap with the flash end.
It emits a partition CSV and JSON offsets; it does not install either artifact.

```sh
python board_flash_plan.py containers.json --csv partitions.csv --offsets offsets.json
```

The partition-table binary must later be built from this CSV by the firmware
owner. Do not flash a container at a planned offset against an unrelated table.
This is a single factory application layout, not an OTA or rollback scheme.

## RAM plan

`board_ram_plan.py` takes five **disjoint byte counts**: `weights`, `kv`, `luts`,
`activations`, `scratch`. Weights and KV go in PSRAM; the other regions in SRAM.
Offsets and total arena allocations are aligned to 16 bytes by default.
The allocator must align both arena bases too. Capacity includes padding and
explicit reserved headroom; arithmetic fit does not prove contiguous allocation.

```sh
python board_ram_plan.py sizes.json --sram-bytes 262144 --sram-reserved 65536 --header ram_layout.h
```

Those SRAM numbers are an illustrative budget, **not measured free RAM**.
The separate architecture-lane planner was absent from this checkout. Its future
integration must supply these five disjoint counts, including guards and unpacking
workspace. `legacy_sizes(**architecture)` delegates to existing `memory_budget.py`:
the entire serialized model (including constant LUTs) is counted as weights, and
the existing combined internal allocation as activations. Separate LUT/scratch
regions are zero in that adapter to avoid counting the same bytes twice.
This adapter does not establish GQA/LittleBit loaded-layout compatibility.

Reserve stack, framework state, structs, allocator overhead, token tables,
sampler scratch, logits and chat buffers when not already included. In particular,
`c/board_chat.c` uses roughly 35 KiB of automatic turn/prompt storage; it needs a
suitably sized task stack, or a future caller-owned workspace adaptation.

## Reference and C entry

`board_chat_reference.py` creates a C header and JSON from a trusted local
checkpoint, tokenizer and JSON turns such as `[["user", "hello"]]`.

```sh
python board_chat_reference.py runs/local --tokenizer tok.model --turns turns.json --max-tokens 16 --header chat_reference.h --json chat_reference.json
```

The CLI uses the existing legacy `intsim` CPU forward; it is not a combined
GQA/LittleBit loader. The reusable `reference` function accepts a full-history
forward callback. It records prompt IDs, emitted greedy IDs and the legacy
uint64 score/scale hash per generation decision, including terminal EOS.
It does not hash each prefill step. Ties choose the lowest ID. No rolling cache,
presence penalty or stochastic sampling is used for these greedy references.
Generated C arrays remain valid even with zero emitted tokens.

`c/board_chat.h` exposes `board_chat_turn`: preencoded role/content spans,
existing sampler options, reset/forward/write callbacks and caller-owned logits
and sampler scratch. It preserves BOS/system and drops oldest complete turns to
reserve the requested generation length, then resets and replays the backend.
Only the first turn may be system. At most 32 turns and 256 context IDs are
supported; a single formatted turn must fit its 256-ID temporary buffer.
Eviction units are individual turns, not automatically paired exchanges.
This entry streams token IDs, not decoded text or score hashes. It does not
implement interactive serial input; firmware must arrange the prepared request.

Protocol lines are ASCII, newline terminated:

```text
CHAT1 BEGIN 42
CHAT1 TOKEN 0 6
CHAT1 TOKEN 1 6
CHAT1 END 3 2
```

BEGIN carries the reference ID. TOKEN carries zero-based sequence index and ID.
END carries stop reason and emitted count: 1 EOS, 3 token limit, 4 context full.
EOS is not emitted. Backend/write failure returns -1 without a successful END.
The reference ID is a fixture identifier, not proof of the loaded model identity.

## Runner and eventual /chips use

`board_chat_run.py` defaults to no implicit device action: choose a recording
or explicitly choose hardware mode. Offline example:

```sh
python board_chat_run.py --reference chat_reference.json --recording serial.bin --result result.json
```

On Eden's host only, after approval for the exact commit, images and exclusive
board turn, the script can use `--hardware --port /dev/board_s3`. Repeat
`--flash OFFSET:FILE` for prebuilt bootloader, partition table, application and
model images at their verified build offsets. No firmware is built by this script.
It invokes installed `python -m esptool`, then reads pyserial at 115200 baud.
Without `--flash`, the operator must independently establish resident image identity.
Firmware must emit CHAT1 after reset/open without a host prompt handshake.
USB reset/enumeration behavior and bot invocation compatibility remain untested.

The decoder bounds lines, checks reference ID/order/count/ranges and compares
IDs and stop reason to the JSON reference. Failure returns exit 1 and a failure
result for framing, incomplete streams or mismatches. Boot noise is allowed only
before BEGIN. Recorded bytes never set `board_verified`; score hashes are always
marked `output_hashes_verified: false`. Even a serial token match is not full
score exactness, image attestation, or evidence of useful chat quality.

## Focused checks and remaining acceptance

Run each named file separately from repository root using
`/Users/obed/repos/Symbex-1/.venv/bin/python -m pytest -q tests/<file>.py`.

- `test_board_flash_plan.py`: 10 passed; CSV, offsets, alignment, exact fit, rejection.
- `test_board_ram_plan.py`: 3 passed; arena sums, alignment, existing budget adapter.
- `test_board_chat_reference.py`: 2 passed; tiny random integer model, EOS/ties/limits.
- `test_board_chat_entry.py`: 1 passed; host clang stub, retention, replay, stream, errors.
- `test_board_chat_run.py`: 10 passed; recorded fake bytes, CLI, flash command planning only.
- `test_board_prep_doc.py`: validates linked assets, example CLI syntax and explicit limitations.

Still required: architecture-planner schema integration, real loaded-size audit,
firmware adapter and build by its owner, task-stack/free-heap measurements,
real tokenizer/model references, full score-hash comparison, approved board flash,
serial capture, model identity verification, and physical speed/power measurements.
No full-model GQA/LittleBit C forward, chat quality, latency or energy claim follows
from these host tests. Team-sync publication is deliberately deferred under the
local-only brief; no remote coordination or mention-discovery result is claimed.
