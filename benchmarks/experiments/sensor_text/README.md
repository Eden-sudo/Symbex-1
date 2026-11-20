# Synthetic sensor-to-text dataset

This prepares synthetic classifier outputs and short English sentences; it is not a trained language model, hardware result, or accuracy measurement. It needs Python 3 and NumPy, and does not download data or use an accelerator.

```
python benchmarks/experiments/sensor_text/make_dataset.py --out output/sensor-text --n 200 50 50 --seed 0
```

`--n` gives train, validation, and test sizes. Train must contain at least one row per source/class pair (currently 41); validation and test can be zero. The labels are loaded from `har_data.CLASSES`, `ecg_data.CLASSES`, and `fomo_classes.VOC_CLASSES` in the HAR, ECG, and FOMO benchmarks. All VOC object classes are included; background is not an object class.

Each JSONL line contains exactly `event` and `text`. For example:

```
{"event": {"source": "har_ultracompact", "class": "Walking", "template_id": "har_ultracompact.a", "duration_seconds": 10}, "text": "Activity Walking lasted 10 seconds."}
```

Each event contains its source, canonical class, and template ID. HAR duration is 1–120 seconds; ECG before and after heart rates are independently 40–180 beats per minute; FOMO detection count is 0–5. These values are synthetic, not measurements or diagnoses.

Each source has four sentence forms: IDs `.a` and `.b` are used only in train, `.c` only in validation, and `.d` only in test. Rows cycle through sources and classes in canonical order without shuffling. A local seeded random generator selects numeric values and permitted templates, making outputs reproducible for the same inputs. Duplicates within a split are possible.

The destination must not exist. The generator creates its parents and writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and finally `meta.json`, containing the seed, sizes, ordered classes, template inventories, and counts per class including zeros. Empty splits produce empty files. Existing destinations are never overwritten. A write failure can leave an incomplete directory without metadata; retry with a fresh destination or deliberately remove the incomplete output first.
