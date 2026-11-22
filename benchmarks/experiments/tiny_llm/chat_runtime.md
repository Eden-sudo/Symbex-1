# Integer chat runtime (host prototype)

This adds runtime building blocks, not a chat-trained model or an ESP32 firmware
integration. The current TinyStories model has not been evaluated for answering
questions or following these role markers. No board performance is claimed.
`c/tinylm.c` and `chip/src/main.cpp` are unchanged.

## Contracts

- **RNG:** `sample_int.xorshift32` / `c/tinylm_rng.h`: xorshift32 (13,17,5),
  nonzero uint32 seed. Seed 1 starts 270369, 67634689, 2647435461.
- **Sampling:** `sample_int.sample` / `chat_sample`: int64 logits, stable lower-ID
  ties. Temperature is an integer number of score units per halving of weight,
  not the usual floating-point softmax temperature. Weight is
  `65536 >> floor((max-score)/temperature)`, zero beyond bucket 16.
  Temperature 0 selects argmax without advancing RNG. `k=0` keeps all IDs.
  Top-p is Q16, 1..65536; after top-k keep the shortest ranked prefix reaching
  `ceil(total_weight * top_p / 65536)`. Draw once with `next_rng % retained_mass`.
  Modulo bias and the staircase weights are intentional, inherited semantics;
  this is not a statistically exact exponential softmax sampler. C uses
  caller-owned scratch for `min(k or vocab, vocab)` IDs, O(vocab*k) ranking.
- **Repetition:** `chat_penalty.penalize` / `chat_penalize`: subtract a nonnegative
  int64 presence penalty once per distinct history ID, saturating at INT64_MIN.
  Not a multiplicative penalty. The loop includes prompt and generated IDs.
- **Stopping:** `chat_stop.Stop` / `ChatStop`: EOS, up to eight byte strings of
  1..64 bytes, and max tokens. Match across token boundaries, including NUL.
  Reasons: 0 running, 1 EOS, 2 string, 3 limit; terminal states are sticky.
  EOS is not counted/emitted; other tokens, including empty pieces, count.
  Zero max tokens stops before inference. EOS wins over strings, strings over
  the token limit. Stop detection includes matching bytes; clients wanting to
  hide them must hold back output (up to the longest stop length), not print
  then try to retract it. The callback loop does not implement that holdback.
- **Template:** `chat_format.py` / `chat_build`: BOS=1, then each turn as
  `encode(role_marker) + encode(content) + [EOS=2]`, ending with an open
  `encode("Assistant:")`. Markers are `System:`, `User:`, `Assistant:`.
  `prepare.py` has no reserved chat IDs: markers use existing pieces. Each span
  is encoded separately; encoding the concatenated text is NOT equivalent.
  Python uses the SentencePiece processor for text encode/decode; C takes
  preencoded spans and copies identical IDs. It is not a C tokenizer.
- **Memory:** `chat_memory.retain` / `chat_retain`: at most 256 IDs, including
  system/BOS and the open assistant suffix, minus generation reservation.
  Keep system and a contiguous suffix of newest whole turns; drop oldest first.
  Never cut a marker or silently drop the newest turn: reject if fixed spans or
  newest turn cannot fit. Supply paired user/assistant exchanges as spans to
  retain pairs atomically. After eviction reset and replay KV state; there is
  no sliding-position cache support. Reserve 1 by default; reserve the desired
  response length to guarantee that many positions.
- **Pieces:** `chat_pieces.py` / `chat_pieces.c`: TCP1 little-endian table,
  `magic[4], vocab:u32, offsets[vocab+1]:u32, payload`. Offsets start at zero,
  end at payload size and are monotonic. Vocab 1..65536. Control pieces empty;
  `<0xHH>` pieces become raw bytes; SentencePiece U+2581 becomes ASCII space.
  Drop one leading dummy-prefix space per independently encoded text stream.
  UTF-8 can span tokens: emit bytes and decode UTF-8 after accumulation, not
  per token. The exporter targets `prepare.py`'s tokenizer settings and default
  unknown surface; arbitrary custom normalizers/unknown surfaces are unsupported.
  The C decoder borrows the immutable blob; keep it alive. Reinitialize for a
  new independently encoded span. It does not automatically reset on EOS/BOS.

## Host driver

From `benchmarks/experiments/tiny_llm/c`:

```sh
/usr/bin/clang -std=c99 -O1 -Wall -Wextra -Werror \
  chat_host.c chat_loop.c chat_sample.c chat_penalty.c chat_stop.c tinylm.c \
  -o chat_host
./chat_host /path/to/trusted/model.bin 32
```

Input: one complete, already formatted and retained prompt per line, as decimal
token IDs separated by whitespace. Output: generated token IDs per line;
stop reason/count on stderr. Every line resets/replays the cache. Clients own
conversation history, tokenization, memory-policy calls and decoding. Defaults:
seed 1 per line, k=40, integer temperature 1, top-p=58982/65536, penalty 0, EOS 2.
The CLI uses EOS/limit only; the reusable `chat_generate` callback API also
supports piece-byte string stops, penalties and streaming token callbacks.
Context exhaustion returns reason 4; callback/validation failures return -1.
The CLI accepts trusted existing model exports up to 16 MiB and context <=256;
it is not a hardened untrusted-model loader. It links the plain host backend,
not chip allocation hooks or PIE. No model fixture existed in repository tests,
so inference was verified only with a stub forward function; real backend was
compiled, not run on a real model.

Export a table separately (not needed by the ID-only CLI):

```sh
python chat_pieces.py /path/to/tok4096.model /path/to/pieces.bin
```

## Focused verification

From repository root, run each file separately with the existing environment:

```sh
/Users/obed/repos/Symbex-1/.venv/bin/python -m pytest -q tests/test_tiny_runtime_rng.py
```

The other focused files use suffixes `sample`, `penalty`, `stop`, `format`,
`memory`, `loop`, `pieces`, and `doc` instead of `rng`. Host tests compile with
`/usr/bin/clang`, use tiny synthetic inputs, and require no model downloads,
training, firmware build or board. Sampling includes randomized logits and
int64 extremes; format/piece tests use synthetic vocabularies, not the actual
4096-piece trained tokenizer. Real tokenizer equivalence, real-model chat
quality, UI integration and ESP32 behavior remain unverified.

## Round 2: assistant-only training data (opt-in)

`chat_data.format_dialogue(sp, turns)` returns `(ids, mask)` with the **same IDs**
as `chat_format.build_chat`: BOS, complete segment-encoded turns, then the open
assistant marker. The boolean mask selects assistant content and each assistant
EOS, but not role markers, BOS, system/user tokens, or the final open marker.
An empty assistant selects its EOS. No assistant means no selected targets.
The formatter does not truncate or pad; callers must keep training examples
within the model context (256 for the production configuration).

`chat_train.ChatTrainingModel(model)` wraps an existing `train.TinyLM` without
changing train.py, its CLI, or its default recipe. It returns `(logits, loss)`:

```python
from chat_data import format_dialogue
from chat_train import ChatTrainingModel
import torch

ids, mask = format_dialogue(sp, [("user", "Hola"), ("assistant", "¡Hola!")])
batch = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
selection = torch.tensor([mask], dtype=torch.bool, device=batch.device)
trainer = ChatTrainingModel(model)
logits, loss = trainer(batch, selection)
loss.backward()
```

`model` and `sp` above are caller-supplied compatible model/tokenizer instances;
this snippet loads nothing. The mask is token-aligned: the loss shifts internally
to compare `logits[:, :-1]` with `ids[:, 1:]` selected by `mask[:, 1:]`. Do not
shift twice. Mean reduction divides by the number of selected target tokens,
not all positions or the batch size. A batch with no selected targets raises
ValueError. Omitting the mask opts into ordinary all-next-token CE. Inputs must
always be valid vocabulary IDs (use valid padding IDs and a false padding mask).
Save the wrapped `model.state_dict()` for existing export consumers, not the
wrapper's prefixed state dict. Optimizer, batching, padding and checkpointing
remain caller-owned. No production fine-tuning has been run.

`chat_synthetic.generate(seed, per_intent=1)` returns a list of records with
`language`, `intent`, and `(role, text)` `turns`. It includes each of greeting,
sensor_state, yes, no, thanks and bye in both Spanish and English exactly
`per_intent` times (12 records by default). A private seeded RNG controls values,
template choices and order; it does not change global RNG state. Sensor values
are explicitly synthetic system context, not live readings. These fixtures
exercise plumbing, not linguistic coverage, held-out accuracy or chat quality.

## Round 2: retained-prefix KV reuse (Python reference only)

`chat_cache.PrefixCache(P)` takes fixed integer parameters from
`intsim.integerize`. It supports one conversation/batch, at most 256 positions
and the exported positional-table length. Keep P/device immutable for the cache
lifetime. `update(ids, force_recompute=False)` returns final scores `[1,1,vocab]`,
final scale `[1,1]`, and the number of forwarded tokens. An identical prompt
does no work; a strict extension forwards only new tokens. A shorter/changed
prefix resets and replays from position zero. This caches per-layer quantized
keys/values and scales, not a memoized full-sequence forward result. Like intsim,
attention scaling assumes 64-wide heads; the fixture uses that head width.

Use the retention adapter so eviction cannot be hidden by repeated token IDs:

```python
from chat_cache import PrefixCache, retention_plan

cache = PrefixCache(P)
# For each turn, using caller-owned, preencoded history spans:
ids, dropped, recompute = retention_plan(
    system, turns, suffix, cache.ids, reserve=16, context=256)
scores, scale, steps = cache.update(ids, force_recompute=recompute)
```

`retention_plan` delegates to round 1's unchanged whole-turn memory policy and
reports recompute on any eviction or mismatch with cached IDs. `cache.ids`
means tokens actually forwarded: a sampled final token or EOS may still need
forwarding when constructing the next complete prompt. Append-only reuse
requires exactly the same IDs at the same absolute positions; retokenizing the
whole conversation can violate that condition. No sliding RoPE positions or
partial prefix repair is attempted. A failed update clears partial state.
The C host CLI still resets/replays every line; this is not a change to that
driver or to Eden's C/firmware files.

## Round 2: serial response framing

[`chat_serial.md`](chat_serial.md) defines TCS1: escaped 0x7E-delimited frames,
version/kind/u16 length, up to 512 payload bytes, and CRC-16/CCITT-FALSE. TOKEN
carries an int32-range ID, TEXT carries raw UTF-8 stream bytes, and END carries
one of EOS/string/token-limit/context-limit/runtime-error reasons. UTF-8 may
span frames. The protocol is standalone; no firmware or serial-port integration
is implied.

`chat_serial.encode` / `c/chat_serial.c` produce byte-identical frames.
`chat_serial.Decoder.feed` accepts arbitrary byte chunks; `finish()` requires a
valid END. Errors poison the decoder; discard the failed response and establish
a new response boundary externally. CRC is corruption detection, not security
or a guarantee against whole-frame loss. Request transport, timeouts, logging
separation, and retransmission are not implemented here.

### Round 2 checks and remaining limits

Run these files separately with the same pytest command shown above:

- `tests/test_tiny_chat_data.py`: hand-made dialogue masks and template parity.
- `tests/test_tiny_chat_train.py`: five CPU toy steps, manual selected-token CE,
  zero logit gradients at excluded positions, default loss and invalid masks.
- `tests/test_tiny_chat_synthetic.py`: seeded bilingual intent coverage and
  formatter compatibility, without downloads or global RNG side effects.
- `tests/test_tiny_chat_cache.py`: exact final scores/scales against whole-prefix
  integer recomputation for synthetic int8/int4/int2 heads, append/replay cases.
- `tests/test_tiny_chat_serial.py`: clang byte equality, escaped payload bounds,
  chunked frames, CRC failures, malformed frames and incomplete responses.
- `tests/test_tiny_runtime_doc.py`: documentation/source contract links.

All evidence is tiny host-only testing. There is no trained chat checkpoint,
real-tokenizer quality evaluation, production-model cache validation, measured
cache speedup, firmware integration, board run or board throughput result.
