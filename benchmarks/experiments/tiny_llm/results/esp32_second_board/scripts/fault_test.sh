#!/usr/bin/env bash
# The two error paths of PR #210 that were only built, never run on a board, each triggered by editing a generated
# data header (no source file of the PR is changed):
#   l2c : chip_l2c/src/gen_data.h GEN_IDS 127 -> 512 (run.c's state at context 512 = 676,192 bytes > the heap):
#         expected "ERROR: run.c needs 676192 bytes of RAM, ... free" (831ffda), raised by chip_fp32.py
#   step: chip_mcu/src/check_data.h check_tokens[0][5] -> 512 (= vocab, out of range: tinylm_step returns -1):
#         expected "ERROR: step" after H lines 0..4 (05e5b6b), raised by chip_run.py
# Usage (repo venv active): PORT=/dev/ttyUSB0 LOGDIR=. fault_test.sh l2c|step
set -u
WHICH=$1
T=$(cd "$(dirname "$0")/../../.." && pwd)   # benchmarks/experiments/tiny_llm
LOG=${LOGDIR:-$PWD}/fault_$WHICH.log
PORT=${PORT:-/dev/ttyUSB0}   # python and pio: the repo venv, activated
cd "$T" || exit 1
if [ "$WHICH" = l2c ]; then
  PROJ=chip_l2c; RUN=runs/fp32_v512d64l5kv4_lr3e-3_20k_s42; SCRIPT=chip_fp32.py
else
  PROJ=chip_mcu; RUN=runs/binary_v512d64l5kv4_s42; SCRIPT=chip_run.py
fi
{
  echo "== fault $WHICH  $(date -Is)  commit $(git rev-parse --short HEAD)"
  echo "-- step 1: data files for $RUN (the board stops on the ID check, expected)"
  timeout 120 python -u $SCRIPT $RUN --board esp32 --port $PORT --no-upload 2>&1 | tail -1
  if [ "$WHICH" = l2c ]; then
    sed -i 's/^#define GEN_IDS 127$/#define GEN_IDS 512/' $PROJ/src/gen_data.h
    grep -n '#define GEN_IDS' $PROJ/src/gen_data.h
  else
    python - "$PROJ/src/check_data.h" <<'EOF'
import re, sys
p = sys.argv[1]; t = open(p).read()
i = t.index("check_tokens[CHECK_WINDOWS][CHECK_TOKENS] = {") ; j = t.index("{", t.index("\n", i)) + 1
row_end = t.index("}", j); row = t[j:row_end].split(",")
print("check_tokens[0][0..7] before:", row[:8]); row[5] = "512"
open(p, "w").write(t[:j] + ",".join(row) + t[row_end:])
print("check_tokens[0][0..7] after: ", row[:8])
EOF
  fi
  echo "-- step 2: build + flash"
  PLATFORMIO_BUILD_FLAGS="" timeout 400 pio run -d $PROJ -e esp32 2>&1 | grep -E 'SUCCESS|FAILED|ERROR'
  B=$PROJ/.pio/build/esp32
  BA=$(find ~/.platformio/packages/framework-arduinoespressif32 -name boot_app0.bin | head -1)
  for try in 1 2 3 4; do
    n=$(timeout 300 pio pkg exec -p tool-esptoolpy -- esptool.py --chip esp32 --port $PORT --baud 115200 \
      write_flash -z 0x1000 $B/bootloader.bin 0x8000 $B/partitions.bin 0xe000 "$BA" 0x10000 $B/firmware.bin 2>&1 \
      | grep -c 'Hash of data verified')
    echo "flash try $try: $n/4 verified"; [ "$n" = 4 ] && break; sleep 3
  done
  sleep 2
  timeout 30 python - "$PORT" <<'EOF'
import serial, sys, time
s = serial.Serial(sys.argv[1], 115200, timeout=0.5)
s.dtr = False; s.rts = True; time.sleep(0.2); s.rts = False
t, n = time.time(), 0
while time.time() - t < 6:
    n += len(s.read(4096))
s.close()
EOF
  echo "-- step 3: read the board with the unmodified $SCRIPT (regenerates the header; the firmware keeps the fault)"
  timeout 300 python -u $SCRIPT $RUN --board esp32 --port $PORT --no-upload 2>&1 | tail -4
  echo "EXIT ${PIPESTATUS[0]}"
  echo "-- raw serial after a reset (12 s)"
  timeout 30 python - "$PORT" <<'EOF'
import serial, sys, time
s = serial.Serial(sys.argv[1], 115200, timeout=0.5)
s.reset_input_buffer(); s.dtr = False; s.rts = True; time.sleep(0.2); s.rts = False
time.sleep(1.5); s.reset_input_buffer(); s.write(b"\x55")
t, out = time.time(), b""
while time.time() - t < 12:
    out += s.read(4096)
for l in out.decode(errors="replace").splitlines()[:14]:
    print("   ", l)
EOF
  git -C "$T" status --short -- $PROJ/src
} > "$LOG" 2>&1
cat "$LOG"
