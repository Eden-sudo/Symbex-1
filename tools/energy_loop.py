"""
energy_loop.py -- keeps a model busy on a board for S seconds, so a USB power meter can measure its energy.

For firmware that is already on the board: SYMBEX's (chip_test.py, model_test.c) or TFLite Micro's (the benchmarks'
*_tflite.py scripts, fashion_tflite.MAIN_TFLM). Sends one input, then the energy-loop command (0xE5, or 0xE6 for the
INT8 reference inside SYMBEX's firmware) with the time in ms; the board runs that input again and again and answers
"L:count,elapsed_us,class". Prints the start and end times (read the meter at both) and the inferences counted.

    python energy_loop.py --port /dev/ttyACM0 --firmware symbex --input runs/resnet8/validation_x.u8 --n-inputs 3072 --seconds 600
    python energy_loop.py --port /dev/ttyACM0 --firmware tflite --input runs/resnet8/validation_x.u8 --n-inputs 3072 --seconds 600
    python energy_loop.py ... --idle                  # no model: only waits, for the board's idle reading

Energy per inference = meter mAh x volts x 3.6 J / count; above idle = (mAh - idle mAh over the same time) x volts
x 3.6 / count. The meter shows whole mAh: run long enough (10 min) that +-1 mAh is small.
--json FILE adds the run (count, time) to a list in FILE; the meter readings go next to it by hand.
ESP32 boards only (esp_timer); the S3 through its native USB port needs the 0x55 hello, which this script sends.
"""
import argparse
import json
import os
import sys
import time


def open_board(port, baud, ready):
    import serial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, 1
    s.dtr = False
    s.rts = False
    s.open()
    time.sleep(1.5)
    s.reset_input_buffer()
    s.write(b"\x55")
    t0 = time.time()
    while time.time() - t0 < 30:
        line = s.readline().decode(errors="replace")
        if ready in line:
            return s
        if not line.strip():
            s.write(b"\x55")                             # ask for the ready line again
    sys.exit(f"[X] no '{ready}' line from {port}: is the firmware on the board?")


def main():
    ap = argparse.ArgumentParser(description="Keep a model busy on a board for a power meter")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--firmware", choices=["symbex", "tflite"], default="symbex")
    ap.add_argument("--int8", action="store_true", help="symbex firmware: loop its INT8 reference (0xE6), not SYMBEX")
    ap.add_argument("--input", help="raw uint8 file of inputs (e.g. a CDT export's validation_x.u8): the first is sent")
    ap.add_argument("--n-inputs", type=int, help="bytes of one input")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--idle", action="store_true", help="no board traffic: only wait, for the idle reading")
    ap.add_argument("--json", help="append the run to this JSON list")
    ap.add_argument("--label", default="", help="name of the run in the JSON")
    args = ap.parse_args()

    run = dict(label=args.label, firmware="idle" if args.idle else args.firmware + ("_int8" if args.int8 else ""),
               seconds=args.seconds)
    if args.idle:
        print(f"[*] IDLE START {time.strftime('%H:%M:%S')} ({args.seconds:g} s)", flush=True)
        time.sleep(args.seconds)
        print(f"[*] IDLE END {time.strftime('%H:%M:%S')}")
    else:
        if not args.input or not args.n_inputs:
            sys.exit("[X] give --input and --n-inputs (or --idle)")
        x = open(args.input, "rb").read(args.n_inputs)
        if len(x) != args.n_inputs:
            sys.exit(f"[X] {args.input} holds fewer than {args.n_inputs} bytes")
        ready, answer = ("TFLITE ready", "T:") if args.firmware == "tflite" else ("RAM:", "R:")
        s = open_board(args.port, args.baud, ready)
        s.write(bytes([0xA5]) + x)                       # one normal inference: the input the loop repeats
        t0 = time.time()
        while not s.readline().decode(errors="replace").startswith(answer):
            if time.time() - t0 > 10:
                sys.exit("[X] the board did not answer the input")
        print(f"[*] LOOP START {time.strftime('%H:%M:%S')} ({args.seconds:g} s): read the meter", flush=True)
        s.write(bytes([0xE6 if args.int8 else 0xE5]) + int(1000 * args.seconds).to_bytes(4, "little"))
        t0 = time.time()
        while True:
            line = s.readline().decode(errors="replace").strip()
            if line.startswith("L:"):
                break
            if time.time() - t0 > args.seconds + 60:
                sys.exit("[X] no L: line after the loop")
        count, us, cls = (int(v) for v in line[2:].split(","))
        if cls < 0:
            sys.exit("[X] the firmware has no INT8 reference (--int8)")
        print(f"[*] LOOP END {time.strftime('%H:%M:%S')}: {count} runs in {us / 1e6:.1f} s = "
              f"{us / count / 1000:.3f} ms per run: read the meter")
        run.update(runs=count, elapsed_us=us, ms_per_run=round(us / count / 1000, 4), seconds=round(us / 1e6, 3))
    if args.json:
        runs = json.load(open(args.json)) if os.path.exists(args.json) else []
        runs.append(run)
        with open(args.json, "w") as f:
            json.dump(runs, f, indent=1)
        print(f"[*] -> {args.json}")


if __name__ == "__main__":
    main()
