#!/usr/bin/env bash
# One PR #210 fp32 run (chip_l2c, llama2.c run.c) on the classic ESP32, in three steps so that chip_fp32.py never
# reads the board right after a pio upload (the CP2102 of our board jams there):
#   1. chip_fp32.py --no-upload: writes the data headers (the board still holds the previous firmware,
#      so it stops on the ID check, expected)
#   2. pio run (build only), then esptool at 115200 until all 4 parts verify (pio's 460800 upload drops on our
#      CP2102)
#   3. chip_fp32.py --no-upload: reads the board, writes results/chip_esp32_l2c_<run>.json
# Usage (from anywhere, repo venv active): PORT=/dev/ttyUSB0 LOGDIR=. run_l2c.sh <run name>
set -u
RUN=$1
LOG=${LOGDIR:-$PWD}/esp32_l2c_$RUN.log
T=$(cd "$(dirname "$0")/../../.." && pwd)   # benchmarks/experiments/tiny_llm
PORT=${PORT:-/dev/ttyUSB0}   # python and pio: the repo venv, activated
cd "$T" || exit 1
{
  echo "== $RUN  $(date -Is)  commit $(git -C "$T" rev-parse --short HEAD)"
  echo "-- step 1: data files"
  timeout 120 python -u chip_fp32.py "runs/$RUN" --board esp32 --port $PORT --no-upload 2>&1 | tail -2
  echo "-- step 2: build (pio), flash at 115200 (esptool; pio's 460800 fails on this CP2102)"
  PLATFORMIO_BUILD_FLAGS="" timeout 1500 pio run -d chip_l2c -e esp32 2>&1 | grep -E 'RAM:|Flash:|SUCCESS|FAILED|ERROR'
  B=chip_l2c/.pio/build/esp32
  BA=$(find ~/.platformio/packages/framework-arduinoespressif32 -name boot_app0.bin | head -1)
  for try in 1 2 3 4; do   # the CP2102 drops long transfers now and then: flash until all 4 parts verify
    n=$(timeout 300 pio pkg exec -p tool-esptoolpy -- esptool.py --chip esp32 --port $PORT --baud 115200 \
      write_flash -z 0x1000 $B/bootloader.bin 0x8000 $B/partitions.bin 0xe000 "$BA" 0x10000 $B/firmware.bin 2>&1 \
      | grep -c 'Hash of data verified')
    echo "flash try $try: $n/4 verified"; [ "$n" = 4 ] && break; sleep 3
  done
  sleep 2
  echo "-- step 2b: throwaway open + reset + drain (the first open after an upload floods)"
  timeout 30 python - "$PORT" <<'EOF'
import serial, sys, time
s = serial.Serial(sys.argv[1], 115200, timeout=0.5)
s.dtr = False; s.rts = True; time.sleep(0.2); s.rts = False
t, n = time.time(), 0
while time.time() - t < 6:
    n += len(s.read(4096))
print(f"drained {n} bytes")
s.close()
EOF
  echo "-- step 3: board"
  timeout 300 python -u chip_fp32.py "runs/$RUN" --board esp32 --port $PORT --no-upload 2>&1
  echo "EXIT $?"
} > "$LOG" 2>&1
grep -E '"(story_equal|first_diverging_story_token|tokens_per_second_story|loaded|ready)"|EXIT|Error|ERROR' "$LOG"
