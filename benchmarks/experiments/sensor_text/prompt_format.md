# Sensor prompt v1

UTF-8 (ASCII subset), no BOM, newline, or implicit whitespace:

```
<sensor_v1>|source=ecg_mitbih|class=N|heart_rate_before=72 bpm|heart_rate_after=90 bpm|<text>
```

Field order is fixed: version, source, class, then the source's numeric fields,
then `<text>`. A training example appends the sentence and `</text>`.
The pipe separates fields; `=` separates keys and values; one ASCII space separates
a number and its unit. No padding, signs, leading zeros, decimal points or rounding.
Readings are already integer classifier outputs; floats (including `1.0`), booleans,
out-of-range readings, missing/extra fields and unknown sources are rejected.

| Source | Ordered numeric fields | Unit | Inclusive range |
| --- | --- | --- | --- |
| har_ultracompact | duration_seconds | s | 1..120 |
| ecg_mitbih | heart_rate_before, heart_rate_after | bpm | 40..180 each |
| fomo_voc_person | count | objects | 0..5 |

Class is a case-sensitive ASCII identifier `[A-Za-z][A-Za-z0-9_]{0,63}`.
Dataset generation supplies the repository's classifier labels; serialization checks
syntax, not membership in those evolving label lists. `template_id` is optional
dataset metadata, ignored by serialization and absent from parsed records: it must
not leak the held-out target wording into the prompt. No other metadata is accepted.
`parse(serialize(event))` returns the canonical record without `template_id`.
The parser accepts only the canonical spelling, ordering and units.
