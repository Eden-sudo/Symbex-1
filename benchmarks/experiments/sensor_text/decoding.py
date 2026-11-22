"""Optional integer-logit constraints for explicit numeric-copy slots."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("sensor_modules", Path(__file__).with_name("_modules.py"))
_modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modules)
prompt = _modules.load("prompt.py")


class NumberCopy:
    """A token trie over record numbers followed by one literal ASCII space.

    The caller enters a numeric slot at a token boundary, with preceding whitespace
    already emitted, and must keep this constraint active until complete(). This
    is not an automatic natural-language number detector. Pass field to prevent
    swapping values that belong to different fields. Prefixes are slot-local IDs.
    """

    def __init__(self, event, sp, field=None):
        record = prompt.canonical_record(event)
        fields = ([item[0] for item in prompt.bilingual.environment.FIELDS] + ["window"]
                  if record["source"] == "environment" else
                  [item[0] for item in prompt.FIELDS[record["source"]]])
        if field is not None:
            if field not in fields:
                raise ValueError("field is not numeric for this record")
            fields = [field]
        self.vocab = sp.vocab_size()
        self.sequences = set()
        for name in fields:
            text = str(record[name]) + " "
            ids = tuple(sp.encode(text))
            if not ids or sp.unk_id() in ids or sp.decode(ids) != text:
                raise ValueError("numeric copy needs a lossless tokenizer")
            self.sequences.add(ids)
        if any(a != b and b[:len(a)] == a for a in self.sequences for b in self.sequences):
            raise ValueError("ambiguous numeric terminator")

    def allowed(self, prefix=()):
        prefix = tuple(prefix)
        matches = [seq for seq in self.sequences if seq[:len(prefix)] == prefix]
        if not matches:
            raise ValueError("prefix is not a record number")
        return sorted({seq[len(prefix)] for seq in matches if len(seq) > len(prefix)})

    def complete(self, prefix):
        return tuple(prefix) in self.sequences

    def mask(self, logits, prefix=(), *, enabled=True):
        """Return (copied scores, legal bitmap), preserving the integer dtype.

        Forbidden scores use dtype.min. Consumers MUST honor the bitmap: dtype.min
        is also a legal input score and ordinary argmax could select a forbidden tie.
        Use choose() for safe greedy selection, including this edge case.
        """
        scores = np.asarray(logits)
        if scores.shape != (self.vocab,) or scores.dtype.kind != "i":
            raise ValueError("expected a vocabulary-sized signed integer logit vector")
        legal = np.ones(self.vocab, dtype=bool)
        if enabled:
            legal[:] = False
            legal[self.allowed(prefix)] = True
            if not legal.any():
                raise ValueError("copy slot is complete; return to ordinary decoding")
        masked = scores.copy()
        masked[~legal] = np.iinfo(scores.dtype).min
        return masked, legal

    def choose(self, logits, prefix=(), *, enabled=True):
        scores, legal = self.mask(logits, prefix, enabled=enabled)
        ids = np.flatnonzero(legal)
        return int(ids[np.argmax(scores[ids])])
