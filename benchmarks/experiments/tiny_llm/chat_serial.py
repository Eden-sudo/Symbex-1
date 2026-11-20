"""TCS1 host framing. See chat_serial.md; no serial-port dependency."""
import binascii
import struct

TOKEN, TEXT, END = 1, 2, 3
MAX_PAYLOAD = 512


def _validate(kind, payload):
    if type(payload) is not bytes or len(payload) > MAX_PAYLOAD:
        raise ValueError("payload must be bytes, at most 512 bytes")
    if type(kind) is not int or kind not in (TOKEN, TEXT, END):
        raise ValueError("unknown frame kind")
    if kind == TOKEN and (len(payload) != 4 or struct.unpack("<I", payload)[0] > 0x7fffffff):
        raise ValueError("token must be a nonnegative int32")
    if kind == TEXT and not payload:
        raise ValueError("text payload must not be empty")
    if kind == END and (len(payload) != 1 or payload[0] not in range(1, 6)):
        raise ValueError("end reason must be 1..5")


def encode(kind, payload):
    _validate(kind, payload)
    body = struct.pack("<BBH", 1, kind, len(payload)) + payload
    body += struct.pack("<H", binascii.crc_hqx(body, 0xffff))
    escaped = bytearray([0x7e])
    for byte in body:
        if byte in (0x7e, 0x7d):
            escaped.extend((0x7d, byte ^ 0x20))
        else:
            escaped.append(byte)
    return bytes(escaped) + b"\x7e"


class Decoder:
    """Incremental single-response decoder; errors poison it until replaced.

    feed returns (kind, raw payload) events. finish requires a complete END.
    TEXT is raw UTF-8 stream bytes: decode across frames, not independently.
    """
    def __init__(self):
        self.body = bytearray()
        self.started = self.escaped = self.ended = self.failed = False

    def feed(self, data):
        if self.failed:
            raise ValueError("decoder failed; start a new response")
        events = []
        try:
            if type(data) is not bytes:
                raise ValueError("feed requires bytes")
            for byte in data:
                if byte == 0x7e:
                    if self.escaped:
                        raise ValueError("truncated escape")
                    if self.body:
                        events.append(self._frame())
                        self.body.clear()
                    self.started = True
                    continue
                if not self.started or self.ended:
                    raise ValueError("data outside response framing")
                if self.escaped:
                    if byte not in (0x5e, 0x5d):
                        raise ValueError("invalid escape")
                    byte ^= 0x20
                    self.escaped = False
                elif byte == 0x7d:
                    self.escaped = True
                    continue
                self.body.append(byte)
                if len(self.body) > MAX_PAYLOAD + 6:
                    raise ValueError("frame too large")
        except ValueError:
            self.failed = True
            raise
        return events

    def _frame(self):
        body = bytes(self.body)
        if len(body) < 6:
            raise ValueError("short frame")
        version, kind, size = struct.unpack_from("<BBH", body)
        if version != 1 or size > MAX_PAYLOAD or len(body) != size + 6:
            raise ValueError("invalid version or length")
        if binascii.crc_hqx(body[:-2], 0xffff) != struct.unpack("<H", body[-2:])[0]:
            raise ValueError("CRC mismatch")
        payload = body[4:-2]
        _validate(kind, payload)
        self.ended = kind == END
        return kind, payload

    def finish(self):
        if self.failed or self.body or self.escaped or not self.ended:
            self.failed = True
            raise ValueError("incomplete or failed response")
