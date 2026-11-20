"""Bounded byte stop detector. Reasons: 0 running, 1 EOS, 2 string, 3 limit."""


class Stop:
    def __init__(self, eos, max_tokens, strings=()):
        if type(eos) is not int or not 0 <= eos <= 2147483647:
            raise ValueError("invalid eos")
        if type(max_tokens) is not int or not 0 <= max_tokens <= 0xffffffff:
            raise ValueError("invalid max_tokens")
        if len(strings) > 8 or any(type(s) is not bytes or not 1 <= len(s) <= 64 for s in strings):
            raise ValueError("at most eight nonempty byte stops, each <=64 bytes")
        self.eos, self.limit, self.strings = eos, max_tokens, tuple(strings)
        self.count, self.tail = 0, b""
        self.reason = 3 if max_tokens == 0 else 0

    def feed(self, token, piece):
        """EOS takes precedence. Stop match beats limit. Terminal state is sticky.

        Detection includes stop bytes; output suppression/holdback is caller-owned.
        EOS is not counted, ordinary tokens (including empty pieces) are counted.
        """
        if type(token) is not int or not 0 <= token <= 2147483647 or type(piece) is not bytes:
            raise ValueError("invalid token/piece")
        if self.reason:
            return self.reason
        if token == self.eos:
            self.reason = 1
            return self.reason
        self.count += 1
        for byte in piece:
            self.tail = (self.tail + bytes([byte]))[-64:]
            if any(self.tail.endswith(s) for s in self.strings):
                self.reason = 2
                return self.reason
        if self.count >= self.limit:
            self.reason = 3
        return self.reason
