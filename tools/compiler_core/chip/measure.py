"""
measure.py -- tests an exported model on the board (the CDT chip stage and tools/chip_test.py).

    generate_project(...)   ESP-IDF project with model_symbex.h, the lib_symbex sources and firmware/model_test.c
    measure_on_chip(...)    flashes it, streams the validation images over serial and reports what the chip
                            measured: equality with the integer simulation (must be 100%), accuracy, latency
                            (total and per layer) and RAM; with an INT8 reference, the same for it on the same images
"""
import numpy as np

from .core import AVR_BOARDS, BOARDS, flash, lib_sources, open_port, write_project

AVR_BAUD = 115200                          # an ATmega328P at 16 MHz is reliable at this speed, not at 921600


def _build_flags(popcount, weights, first_layer):
    """Compiler flags of the model test: popcount method, where the weights live, first-layer kernel (XNOR
    headers only; the AND arithmetic has a single one)."""
    return ((["-DSYMBEX_POPCOUNT_NO_LUT16"] if popcount == "swar" else [])
            + (["-DSYMBEX_ROM="] if weights == "ram" else [])
            + (["-DSYMBEX_U8_SIMPLE"] if first_layer == "simple" else [])
            + (["-DSYMBEX_U8_PLANES"] if first_layer == "planes" else []))


USB_BOARDS = {"s3"}                        # boards with a native USB-Serial-JTAG port (usb=True)


def generate_project(project_dir, header_path, lib_path, board, port, baud, popcount="lut16",
                     weights="flash", first_layer="mask", int8_path=None, usb=False):
    """PlatformIO project: the exported header, the lib_symbex sources it needs, and the firmware (ESP-IDF on the
    ESP32 boards; Arduino on AVR boards, with the weights in flash: PROGMEM, read by the library).
    usb=True: the firmware talks over the chip's native USB-Serial-JTAG port instead of UART0 (the ESP32-S3)."""
    if usb and board not in USB_BOARDS:
        raise ValueError(f"chip: --usb needs a board with a native USB-Serial-JTAG port ({sorted(USB_BOARDS)})")
    extra = {"model_symbex.h": header_path}
    if board in AVR_BOARDS:
        write_project(project_dir, board, port, lib_sources(lib_path, ("symbex_dense.c", "symbex_hal_generic.c")),
                      "model_test_avr.cpp", {"BAUD_RATE": AVR_BAUD}, extra, lib_path=lib_path)
        return
    defines = {"BAUD_RATE": baud}
    if int8_path:
        extra["model_int8.h"] = int8_path
        defines.update(SYMBEX_WITH_INT8=1, SYMBEX_WITH_ESP_NN=1)
    if usb:
        defines.update(SYMBEX_USB_SERIAL_JTAG=1)
    write_project(project_dir, board, port, lib_sources(lib_path), "model_test.c", defines, extra,
                  _build_flags(popcount, weights, first_layer), ["espressif/esp-nn"] if int8_path else (),
                  "CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n" if usb else "", lib_path=lib_path)


def measure_on_chip(header_path, x, y, expected, project_dir, lib_path, board="esp32", port="/dev/ttyUSB0",
                  baud=921600, n=1000, upload=True, popcount="lut16", model_flash=None, verbose=True,
                  weights="flash", first_layer="mask", int8=None, usb=False):
    """Flashes the exported model and measures it on the board: equality with the integer simulation (must be
    100%), accuracy, latency (total and per layer) and RAM. x / y / expected: validation images (uint8), labels
    and the classes of the integer simulation. Returns a dict, or raises RuntimeError with what went wrong.
    int8 (optional): dict(header=..., expected=classes of its integer simulation, flash=bytes) to run the INT8
    reference of the original model on the same chip and the same images (Espressif's ESP-NN kernel).
    usb=True: through the board's native USB-Serial-JTAG port (port is then e.g. /dev/ttyACM0), not its UART."""


    import time
    generate_project(project_dir, header_path, lib_path, board, port, baud, popcount, weights, first_layer,
                     int8["header"] if int8 else None, usb)
    if upload:
        flash(project_dir, verbose, f"{BOARDS[board]} on {port} (popcount {popcount}, weights in {weights}, "
                                    f"first layer {first_layer}" + (", native USB" if usb else "") + ")")
    n = min(n, len(x))
    pred, totals, per_step, names, ram, stack = [], [], [], {}, None, None
    pred8, totals8, per_step8, info8 = [], [], [], None
    if board in AVR_BOARDS:
        baud = AVR_BAUD
        if verbose and weights == "ram":
            print("    [!] AVR: the weights always stay in flash (2 KB of RAM); --weights ram is ignored")
        weights, int8 = "flash", None
    with open_port(port, baud, "dtr" if board in AVR_BOARDS else "usb" if usb else "rts") as s:
        t0, warned, hal = time.time(), False, "?"
        while True:
            l = s.readline().decode(errors="replace").strip()
            if usb and not l and hal == "?":
                s.write(b"\x55")                       # native USB: ask again for the ready lines
            if "SYMBEX ready" in l:
                hal = l.split("|")[1].strip() if "|" in l else "?"
            elif l.startswith("STEP:"):
                i, label = l[5:].split("|", 1)
                names[int(i)] = label
            elif l.startswith("RAM:"):
                ram = [int(v) for v in l[4:].split(",")]
                if not int8:
                    break
            elif l.startswith("INT8:"):
                info8 = [int(v) for v in l[5:].split(",")]
                break
            if not warned and time.time() - t0 > 8:
                print("    >>> the board did not answer the reset: press EN/RST once")
                warned = True
            if time.time() - t0 > 60:
                raise RuntimeError("chip: the board did not start (no 'SYMBEX ready')")
        if ram is None:
            raise RuntimeError("chip: the board sent 'INT8:' without its 'RAM:' line (garbled serial?): retry")
        if verbose:
            print(f"    board ready | popcount: {hal}")
        for i in range(n):
            s.write(bytes([0xA5]) + np.ascontiguousarray(x[i], dtype=np.uint8).tobytes())
            deadline = time.time() + 10
            while True:
                l = s.readline().decode(errors="replace").strip()
                if l.startswith("R:"):
                    v = [int(t) for t in l[2:].split(",")]
                    pred.append(v[0])
                    totals.append(v[1])
                    per_step.append(v[2:])
                    if not int8:
                        break
                elif l.startswith("Q:"):
                    v = [int(t) for t in l[2:].split(",")]
                    pred8.append(v[0])
                    totals8.append(v[1])
                    per_step8.append(v[2:])
                    break
                if time.time() > deadline:
                    raise RuntimeError(f"chip: no answer for image {i}")
            if verbose and (i + 1) % max(1, n // 5) == 0:
                print(f"      {i + 1}/{n} | equal to the simulation {100 * np.mean(np.array(pred) == expected[:len(pred)]):.2f}%"
                      f" | median {np.median(totals) / 1000:.3f} ms")
        s.write(bytes([0x5A]))
        deadline = time.time() + 5
        while time.time() < deadline:
            l = s.readline().decode(errors="replace").strip()
            if l.startswith("STACK:"):
                stack = int(l.split(":", 1)[1])
                break
    pred, per_step = np.array(pred), np.array(per_step)
    medians = np.median(per_step, axis=0) / 1000
    buffers, input_bytes, table, stack_total = ram
    stack_used = None if stack is None else stack_total - stack
    out = dict(board=board, hal=hal, popcount=popcount, weights=weights, first_layer=first_layer, n=n,
               equal_simulation=100 * float((pred == expected[:n]).mean()),
               accuracy=100 * float((pred == y[:n]).mean()), median_ms=float(np.median(totals)) / 1000,
               min_ms=float(np.min(totals)) / 1000,
               layers=[dict(step=names.get(i, f"step {i}"), median_ms=float(m)) for i, m in enumerate(medians)],
               ram=dict(buffers=buffers, input_bytes=input_bytes, popcount_table=table, stack_used=stack_used,
                        weights_in_ram=(model_flash or 0) if weights == "ram" else 0,
                        total=buffers + input_bytes + table + (stack_used or 0)
                        + ((model_flash or 0) if weights == "ram" else 0)),
               model_flash=model_flash)
    if int8:
        p8 = np.array(pred8)
        out["int8"] = dict(equal_simulation=100 * float((p8 == int8["expected"][:n]).mean()),
                           accuracy=100 * float((p8 == y[:n]).mean()), median_ms=float(np.median(totals8)) / 1000,
                           layers_ms=[float(m) for m in np.median(np.array(per_step8), axis=0) / 1000],
                           ram_buffers=info8[1] if info8 else None, flash=int8.get("flash"))
    if verbose:
        status = "OK" if out["equal_simulation"] == 100 else "[X] MISMATCH between the chip and the simulation"
        mhz = 16 if board in AVR_BOARDS else 240
        print(f"\n    ON THE CHIP  ({BOARDS[board]}, {mhz} MHz, one core, {hal}, weights in {weights}, first layer {first_layer})")
        print(f"    images        : {n} | equal to the integer simulation {out['equal_simulation']:.2f}% {status}")
        print(f"    accuracy      : {out['accuracy']:.2f}%")
        print(f"    latency       : median {out['median_ms']:.3f} ms | min {out['min_ms']:.3f} ms")
        print(f"\n    {'step':<52}{'ms':>9}{'%':>7}")
        print("    " + "-" * 68)
        total_sum = float(medians.sum()) or 1.0
        for c in out["layers"]:
            bar = "#" * int(round(20 * c["median_ms"] / total_sum))
            print(f"    {c['step']:<52}{c['median_ms']:>9.3f}{100 * c['median_ms'] / total_sum:>6.1f}%  {bar}")
        print("    " + "-" * 68)
        print(f"    {'total (sum of the steps)':<52}{total_sum:>9.3f}")
        r = out["ram"]
        print(f"\n    RAM (measured on the chip)")
        print(f"    model buffers (static)    : {r['buffers'] / 1024:7.2f} KB")
        print(f"    input image (static)      : {r['input_bytes'] / 1024:7.2f} KB")
        print(f"    popcount table (heap)     : {r['popcount_table'] / 1024:7.2f} KB")
        if weights == "ram":
            print(f"    weights + thresholds (RAM): {r['weights_in_ram'] / 1024:7.2f} KB")
        print(f"    stack used by inference   : " + ("   n/a" if stack_used is None else f"{stack_used / 1024:7.2f} KB")
              + f"  (of {stack_total / 1024:.0f} KB reserved)")
        print(f"    TOTAL                     : {r['total'] / 1024:7.2f} KB")
        if model_flash:
            print(f"    flash (weights+thresholds): {model_flash / 1024:7.2f} KB")
        if int8:
            r8 = out["int8"]
            status8 = "OK" if r8["equal_simulation"] == 100 else "[X] MISMATCH"
            print(f"\n    INT8 REFERENCE ON THE SAME CHIP  (original model, ESP-NN esp_nn_fully_connected_s8, weights in flash"
                  + (": --weights ram moves only the SYMBEX weights" if weights == "ram" else "") + ")")
            print(f"    equal to its integer simulation {r8['equal_simulation']:.2f}% {status8}")
            print("    " + "  ".join(f"layer {i}: {m:.3f} ms" for i, m in enumerate(r8["layers_ms"])))
            print(f"\n    {'model':<26}{'accuracy':>9}{'latency ms':>12}{'speed vs INT8':>15}{'flash KB':>10}{'RAM KB':>9}")
            print("    " + "-" * 81)
            ram8 = (r8["ram_buffers"] or 0) + out["ram"]["input_bytes"]
            print(f"    {'INT8 (ESP-NN)':<26}{r8['accuracy']:>8.2f}%{r8['median_ms']:>12.3f}{1.0:>14.2f}x"
                  f"{(r8['flash'] or 0) / 1024:>10.1f}{ram8 / 1024:>9.2f}")
            print(f"    {'SYMBEX':<26}{out['accuracy']:>8.2f}%{out['median_ms']:>12.3f}"
                  f"{r8['median_ms'] / out['median_ms']:>14.2f}x{(model_flash or 0) / 1024:>10.1f}{out['ram']['total'] / 1024:>9.2f}")
            print("    (speed vs INT8 > 1: SYMBEX is faster. RAM: model buffers + input (+ table, stack, weights for SYMBEX))")
    return out
