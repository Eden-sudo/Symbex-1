"""
RESERVED, NOT IMPLEMENTED.

Binarizing attention needs something other than standard.py's plain CE+STE
loop: softmax attention scores are a long-tailed probability distribution
(most values near zero, a few near one), not a sum-of-signed-products like
conv/linear. A fixed threshold (the same sign() used everywhere else in
this project) misrepresents that distribution -- confirmed against real
findings while testing an attention model today (it "failed horribly"),
and consistent with published work (BiViT, BiBERT, BinaryAttention): all
of them need a threshold that adapts to each attention vector's own
distribution, or they decouple attention's binarization from the rest of
the network's.

This file is left empty on purpose. Writing a "standard" recipe here
without our own evidence of what actually works for OUR engine would be
guessing, not engineering -- the same mistake we avoided elsewhere in this
project (see methods/: ported instead of generalized, precisely to avoid
inventing behavior for a case with no evidence yet).

Build this when there is a real methods/attention/ layer to train, not
before.
"""

raise NotImplementedError(
    "train.attention is a placeholder. There is no SYMBEX attention method yet "
    "(see methods/ -- only linear/ and conv/ exist). Do not import this module."
)
