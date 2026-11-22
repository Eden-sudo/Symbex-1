# TCS1 board-to-host response stream (version 1)

Transport-independent binary protocol for one generated response. This is a
specification and standalone codec, **not firmware integration**. No port is
opened and no board behavior has been verified. Do not mix logs with frames.
Request framing, baud rate, flow control and request IDs are outside this version;
the caller serializes requests, drains/resets the transport, then creates a fresh
decoder for each response. There is no retransmission or multiplexing.

## Frame

`7E | escaped body | 7E`. Consecutive delimiters are allowed. In the body only,
replace `7E` with `7D 5E` and `7D` with `7D 5D`. Other escape pairs are invalid.
Unescaped body fields, in order:

| Field | Encoding |
|---|---|
| version | u8, exactly 1 |
| kind | u8: 1 TOKEN, 2 TEXT, 3 END |
| payload length | u16 little-endian, at most 512 |
| payload | exactly length bytes |
| CRC | u16 little-endian |

CRC-16/CCITT-FALSE: polynomial 0x1021, initial 0xFFFF, no reflection, xorout 0.
Cover version, kind, length and payload **before escaping**; exclude delimiters
and CRC itself. Check vector `123456789` gives 0x29B1. Maximum unescaped body is
518 bytes; maximum encoded frame is 1038 bytes (including both delimiters).

- TOKEN: exactly four bytes, little-endian ID in 0..2147483647. Vocabulary bounds
  are checked by the consumer. EOS is represented by END, not an emitted TOKEN.
- TEXT: 1..512 raw bytes of the output UTF-8 stream. NUL and delimiter bytes are
  legal. A UTF-8 code point may span frames; use incremental UTF-8 decoding.
  Empty token pieces need no TEXT frame.
- END: exactly one byte: 1 EOS, 2 string stop, 3 token limit, 4 context limit,
  5 runtime error. Exactly one END terminates a response, including zero-token
  responses. No non-delimiter bytes may follow it in that decoder.

Send TOKEN, TEXT or both as needed. When sending both, send TOKEN then its TEXT
chunks; they describe the same output, not two separate answers. Codec validation
does not enforce this optional association. Stop-byte suppression is upstream.
The C encoder is stateless; its caller owns ordering and exactly-one-END.

## Host and failure contract

`chat_serial.encode(kind, payload)` and `chat_serial.Decoder.feed(bytes)` use
raw byte payloads. `feed` accepts arbitrary chunk boundaries and returns a list
of `(kind, payload)` events only after validating whole frames. `finish()` must
succeed before treating a response as complete. EOF/timeouts without END are
incomplete, not success; timeouts are transport-owned.

Malformed escaping, version/kind/size, CRC, oversized bodies, trailing data, or
data before a delimiter raises ValueError and poisons the decoder. On error,
discard the response and create a new decoder only at the next externally
established response boundary. No automatic resynchronization or CRC bypass.
Already displayed chunks cannot be retracted by this codec; clients that require
atomic results must buffer until finish. CRC detects corruption, not attackers;
it provides no authentication and cannot detect loss of a whole valid frame.

`c/chat_serial.h` exposes a caller-buffer encoder, returns 0 on invalid arguments
or insufficient capacity without modifying the output, and allocates no heap.
Payload/output buffers must not overlap. Python/C byte equality is host-tested
with tiny synthetic frames via clang; no board or throughput claim is made.
