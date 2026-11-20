"""Decode CHAT1 recordings or explicitly flash prebuilt images on Eden's host."""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


class Decoder:
    """Bounded incremental ASCII framing; boot noise allowed only before BEGIN."""
    def __init__(self, reference):
        self.reference = reference
        self.buffer = bytearray()
        self.started = self.done = False
        self.ids = []
        self.reason = None

    def feed(self, data):
        for byte in data:
            if byte != 10:
                self.buffer.append(byte)
                if len(self.buffer) > 512:
                    raise ValueError("serial line exceeds 512 bytes")
                continue
            line = bytes(self.buffer).rstrip(b"\r")
            self.buffer.clear()
            if not line.startswith(b"CHAT1 "):
                if self.started:
                    raise ValueError("unexpected data inside chat stream")
                continue
            try:
                fields = line.decode("ascii").split()
                if self.done:
                    raise ValueError("records after END")
                if fields[1] == "BEGIN" and len(fields) == 3 and not self.started:
                    if int(fields[2]) != self.reference["reference_id"]:
                        raise ValueError("reference ID mismatch")
                    self.started = True
                elif fields[1] == "TOKEN" and len(fields) == 4 and self.started:
                    index, token = map(int, fields[2:])
                    if index != len(self.ids) or not 0 <= token < self.reference["vocab"]:
                        raise ValueError("invalid token index or ID")
                    if len(self.ids) >= self.reference["max_tokens"]:
                        raise ValueError("too many tokens")
                    self.ids.append(token)
                elif fields[1] == "END" and len(fields) == 4 and self.started:
                    self.reason, count = map(int, fields[2:])
                    if self.reason not in (1, 3, 4) or count != len(self.ids):
                        raise ValueError("invalid terminal record")
                    self.done = True
                else:
                    raise ValueError("invalid framing or order")
            except (UnicodeError, IndexError) as exc:
                raise ValueError("malformed CHAT1 record") from exc

    def result(self, source):
        complete = self.done and not self.buffer
        equal = (complete and self.ids == self.reference["expected_ids"]
                 and self.reason == self.reference["stop_reason"])
        return dict(protocol="CHAT1", source=source, complete=complete, passed=equal,
                    reference_id=self.reference["reference_id"], token_ids=self.ids,
                    stop_reason=self.reason, token_ids_equal=equal,
                    output_hashes_verified=False, board_verified=source == "serial" and equal)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--recording", type=Path)
    mode.add_argument("--hardware", action="store_true", help="requires separate board approval")
    ap.add_argument("--port", default="/dev/board_s3")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--flash", action="append", default=[], metavar="OFFSET:FILE",
                    help="prebuilt images only; no builds; hardware mode only")
    return ap


def flash_command(images, port):
    """Reject overlapping sector erases and anything beyond 16 MiB before flash."""
    spans, args = [], []
    for item in images:
        offset_text, filename = item.split(":", 1)
        offset, path = int(offset_text, 0), Path(filename)
        size = path.stat().st_size
        end = offset + (size + 4095) // 4096 * 4096
        if offset < 0 or offset % 4096 or size == 0 or end > 16 * 1024 * 1024:
            raise ValueError("invalid image offset or size")
        spans.append((offset, end))
        args.extend([hex(offset), str(path.resolve())])
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping flash image sectors")
    return [sys.executable, "-m", "esptool", "--chip", "esp32s3", "--port", port,
            "write_flash", "--flash_size", "16MB", *args]


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("timeout must be finite and positive")
    if args.flash and not args.hardware:
        ap.error("--flash requires --hardware")
    decoder = Decoder(json.loads(args.reference.read_text()))
    source = "recording" if args.recording else "serial"
    error = None
    try:
        if args.recording:
            with args.recording.open("rb") as stream:
                while chunk := stream.read(4096):
                    decoder.feed(chunk)
        else:
            # Lazy imports and opt-in branch ensure offline parsing has no I/O
            # to a serial device, and does not require pyserial or esptool.
            import serial
            if args.flash:
                subprocess.run(flash_command(args.flash, args.port), check=True, timeout=180)
            with serial.Serial(args.port, 115200, timeout=0.2) as stream:
                deadline = time.monotonic() + args.timeout
                while not decoder.done and time.monotonic() < deadline:
                    decoder.feed(stream.read(256))
                if not decoder.done:
                    raise ValueError("timeout waiting for CHAT1 END")
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
        error = str(exc)
    result = decoder.result(source)
    if error:
        result.update(passed=False, board_verified=False, error=error)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
