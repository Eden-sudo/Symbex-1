"""
core.py -- what every chip tool shares: the boards, the PlatformIO / ESP-IDF project, flashing, the serial port.

    write_project(project_dir, board, port, sources, firmware, defines, ...)   writes an ESP-IDF project
    flash(project_dir, verbose)                                              builds and uploads it (pio)
    open_port(port, baud)                                                    opens the serial port + reset pulse
    wait_ready(serial, token, ...)                                           waits for the board's ready line
    lib_sources(lib_path, names)                                             paths of the lib_symbex sources
"""
import os
import shutil
import subprocess
import time

BOARDS = {"esp32": "esp32dev", "s3": "esp32-s3-devkitc-1", "uno": "uno", "pico2_arm": "rpipico2",
          "pico2_riscv": "rpipico2"}
AVR_BOARDS = {"uno"}                       # Arduino framework, ATmega328P: no ESP-IDF, no lib_symbex sources
PICO_BOARDS = {"pico2_arm", "pico2_riscv"}  # Raspberry Pi Pico 2 (RP2350), Arduino-Pico core: Cortex-M33 or Hazard3
BOARD_MHZ = {"esp32": 240, "s3": 240, "uno": 16, "pico2_arm": 150, "pico2_riscv": 150}
FIRMWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firmware")

# lib_symbex sources by short name: name in the project -> path inside lib_symbex
LIB_SOURCES = {
    "symbex_dense.c": ("src", "layers", "dense", "symbex_dense.c"),
    "symbex_and.c": ("src", "ops", "symbex_and.c"),
    "symbex_hal_generic.c": ("src", "hal", "generic", "symbex_hal_impl.c"),
    "symbex_hal_esp32s3.c": ("src", "hal", "esp32s3", "symbex_hal_impl.c"),
    "symbex_hal_rp2350.c": ("src", "hal", "rp2350", "symbex_hal_impl.c"),   # Pico 2 first layer (empty elsewhere)
    "symbex_bool.c": ("src", "layers", "conv", "symbex_bool.c"),      # fully binary conv networks (method boolean)
    "symbex_bool1d.c": ("src", "layers", "conv", "symbex_bool1d.c"),  # fully binary 1D conv networks (time series)
    "symbex_ternary.c": ("src", "layers", "dense", "symbex_ternary.c"),  # ternary dense networks (method ternary)
}


def lib_sources(lib_path, names=tuple(LIB_SOURCES)):
    """{name in the project: path in lib_symbex} for the given sources; raises ValueError if one is missing."""
    out = {}
    for n in names:
        p = os.path.join(lib_path, *LIB_SOURCES[n])
        if not os.path.exists(p):
            raise ValueError(f"chip: {p} not found (check [output] lib_path)")
        out[n] = p
    return out


def write_project(project_dir, board, port, sources, firmware, defines=None, extra_files=None, build_flags=(),
                  components=(), sdkconfig_extra="", lib_path=None):
    """Writes an ESP-IDF project for PlatformIO.
    sources: {name: path} copied to src/; firmware: file name in chip/firmware/ (becomes src/main.c, after the
    `defines` as #define lines); extra_files: {name in src/: path} (e.g. exported headers); build_flags: compiler
    flags; components: ESP-IDF managed components (e.g. "espressif/esp-nn"); lib_path: copies its include/."""
    if board not in BOARDS:
        raise ValueError(f"chip: board must be one of {sorted(BOARDS)}")
    if board in AVR_BOARDS:
        return _write_project_avr(project_dir, board, port, sources, firmware, defines, extra_files, build_flags, lib_path)
    if board in PICO_BOARDS:
        return _write_project_pico(project_dir, board, port, sources, firmware, defines, extra_files, build_flags,
                                   lib_path)
    src = os.path.join(project_dir, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    for f in os.listdir(project_dir):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project_dir, f))
    if lib_path:
        shutil.copytree(os.path.join(lib_path, "include"), os.path.join(src, "include"))
    for name, path in list(sources.items()) + list((extra_files or {}).items()):
        shutil.copy(path, os.path.join(src, name))
    head = "".join(f"#define {k} {v}\n" for k, v in (defines or {}).items())
    with open(os.path.join(FIRMWARE_DIR, firmware)) as f:
        main_c = f.read()
    with open(os.path.join(src, "main.c"), "w") as f:
        f.write(head + main_c)
    with open(os.path.join(src, "CMakeLists.txt"), "w") as f:
        f.write('idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in sources)
                + ' INCLUDE_DIRS "." "include")\n')
    if components:
        with open(os.path.join(src, "idf_component.yml"), "w") as f:
            f.write("dependencies:\n" + "".join(f'  {c}: "*"\n' for c in components))
    with open(os.path.join(project_dir, "platformio.ini"), "w") as f:
        f.write(f"[env:{BOARDS[board]}]\nplatform = espressif32\nboard = {BOARDS[board]}\nframework = espidf\n"
                f"monitor_speed = 115200\nupload_port = {port}\n"
                + (("build_flags = " + " ".join(build_flags) + "\n") if build_flags else ""))
    with open(os.path.join(project_dir, "sdkconfig.defaults"), "w") as f:
        f.write("CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                "CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\nCONFIG_ESP32S3_DEFAULT_CPU_FREQ_240=y\n"
                "CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192\nCONFIG_LOG_DEFAULT_LEVEL_WARN=y\nCONFIG_ESP_TASK_WDT_EN=n\n"
                "CONFIG_NN_OPTIMIZED=y\n" + sdkconfig_extra)
    with open(os.path.join(project_dir, "CMakeLists.txt"), "w") as f:
        f.write("cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                f"project({os.path.basename(os.path.abspath(project_dir))})\n")


def _write_project_avr(project_dir, board, port, sources, firmware, defines, extra_files, build_flags, lib_path):
    """PlatformIO project for an AVR board with the Arduino framework: src/main.cpp (the firmware after the
    `defines`), the lib_symbex sources and include/, and the exported header(s)."""
    src = os.path.join(project_dir, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    if lib_path:
        shutil.copytree(os.path.join(lib_path, "include"), os.path.join(src, "include"))
    for name, path in list(sources.items()) + list((extra_files or {}).items()):
        shutil.copy(path, os.path.join(src, name))
    head = "".join(f"#define {k} {v}\n" for k, v in (defines or {}).items())
    with open(os.path.join(FIRMWARE_DIR, firmware)) as f:
        main = f.read()
    with open(os.path.join(src, "main.cpp"), "w") as f:
        f.write(head + main)
    flags = ["-Isrc/include", "-Os"] + list(build_flags)
    with open(os.path.join(project_dir, "platformio.ini"), "w") as f:
        f.write(f"[env:{BOARDS[board]}]\nplatform = atmelavr\nboard = {BOARDS[board]}\nframework = arduino\n"
                f"monitor_speed = 115200\nupload_port = {port}\nbuild_flags = " + " ".join(flags) + "\n")


def _write_project_pico(project_dir, board, port, sources, firmware, defines, extra_files, build_flags, lib_path):
    """PlatformIO project for the Raspberry Pi Pico 2 with the Arduino-Pico core: src/main.cpp (the firmware after
    the `defines`), the lib_symbex sources and include/, the exported header(s). pico2_arm builds for the Cortex-M33,
    pico2_riscv for the Hazard3 cores (board_build.mcu = rp2350-riscv). -O2 like the ESP-IDF projects (Arduino-Pico
    builds with -Os by default). Upload with picotool over the USB port."""
    src = os.path.join(project_dir, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    if lib_path:
        shutil.copytree(os.path.join(lib_path, "include"), os.path.join(src, "include"))
    for name, path in list(sources.items()) + list((extra_files or {}).items()):
        shutil.copy(path, os.path.join(src, name))
    head = "".join(f"#define {k} {v}\n" for k, v in (defines or {}).items())
    with open(os.path.join(FIRMWARE_DIR, firmware)) as f:
        main = f.read()
    with open(os.path.join(src, "main.cpp"), "w") as f:
        f.write(head + main)
    flags = ["-Isrc/include", "-O2"] + list(build_flags)
    with open(os.path.join(project_dir, "platformio.ini"), "w") as f:
        f.write(f"[env:{board}]\nplatform = raspberrypi\nboard = {BOARDS[board]}\nframework = arduino\n"
                f"board_build.core = earlephilhower\nupload_protocol = picotool\nupload_port = {port}\n"
                + ("board_build.mcu = rp2350-riscv\n" if board == "pico2_riscv" else "")
                + "build_flags = " + " ".join(flags) + "\nbuild_unflags = -Os\n")


def flash(project_dir, verbose=True, what=""):
    """Builds and uploads the project with PlatformIO. Raises RuntimeError if pio is missing or the build fails."""
    if shutil.which("pio") is None:
        raise RuntimeError("chip: PlatformIO (pio) not found")
    if verbose:
        print(f"    building and flashing{(' ' + what) if what else ''} (the first build downloads the framework)...")
    r = subprocess.run(["pio", "run", "-d", project_dir, "--target", "upload"], capture_output=not verbose)
    if r.returncode != 0:
        raise RuntimeError("chip: build or upload failed (run `pio run -d " + project_dir + " -t upload` to see it)")


def open_port(port, baud, reset="rts"):
    """Opens the serial port and resets the board: reset="rts" (ESP32: RTS pulse on EN), "dtr" (Arduino:
    DTR pulse on RESET, then the bootloader's ~2 s) or "usb" (ESP32-S3 native USB-Serial-JTAG: opening the port
    already resets the chip, an RTS pulse would re-enumerate the port under us; the firmware then waits for the
    host's 0x55, sent here and again by the caller while it waits for the ready lines) or "pico" (Raspberry Pi
    Pico 2: its USB port re-enumerates after flashing, so the port is opened again until it is back, then 0x55)."""
    import serial
    if reset == "pico":
        t0 = time.time()
        while True:
            try:
                s = serial.Serial(port, baud, timeout=1)
                break
            except (serial.SerialException, OSError):
                if time.time() - t0 > 30:
                    raise RuntimeError(f"chip: {port} did not come back after flashing")
                time.sleep(0.5)
        time.sleep(1.0)
        s.reset_input_buffer()
        s.write(b"\x55")
        return s
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, 1
    s.dtr = False
    s.rts = False
    s.open()
    s.reset_input_buffer()
    if reset == "usb":
        time.sleep(1.5)                                  # the chip boots again after the port opens
        s.reset_input_buffer()
        s.write(b"\x55")
    elif reset == "dtr":
        s.dtr = True
        time.sleep(0.15)
        s.dtr = False
    else:
        s.rts = True
        time.sleep(0.15)
        s.rts = False
    return s


def read_line(s):
    return s.readline().decode(errors="replace").strip()


def wait_ready(s, token, timeout=60, on_line=None):
    """Reads lines until one contains `token` (returned). on_line(line) is called for every line before it.
    Asks for a manual reset after 8 s; raises RuntimeError after `timeout` seconds."""
    t0, warned = time.time(), False
    while True:
        line = read_line(s)
        if token in line:
            return line
        if on_line and line:
            on_line(line)
        if not warned and time.time() - t0 > 8:
            print("    >>> the board did not answer the reset: press EN/RST once")
            warned = True
        if time.time() - t0 > timeout:
            raise RuntimeError(f"chip: the board did not start (no '{token}')")
