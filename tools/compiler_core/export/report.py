"""
Before/after report shared by ALL exporters.

Every exporter builds one row per exported layer and calls print_report(). A row is a dict:
  required: name, shape (free text), n_weights, bits (per weight), weights_bytes (packed, with padding),
            params_bytes (everything else the layer needs)
  optional: parts       {array label: bytes}                    -> "ROM by array" (detail)
            map_bytes   bytes of the layer's output activation  -> RAM
            xnor_words  32-bit XNOR+popcount words per image    -> compute share (detail)
The INPUT side is read straight from the PyTorch model (exact bytes), so it does not depend on
the method or the architecture.
"""
import os


def describe_input(model):
    """Weight tensors (conv / linear / gate / core / planes) and everything else, as PyTorch stores them."""
    weights, other = [], 0
    for name, p in model.named_parameters():
        if p.dim() >= 2 and name.endswith("weight"):
            weights.append((name, tuple(p.shape), p.numel(), p.numel() * p.element_size()))
        else:
            other += p.numel() * p.element_size()
    for _, b in model.named_buffers():
        other += b.numel() * b.element_size()
    return weights, other


def _kb(b):
    return b / 1024


def print_report(model, rows, out_path, detail=False, skipped=()):
    """skipped: names of weight tensors that exist in the model but are intentionally not exported."""
    weights, other = describe_input(model)
    n_in = sum(w[2] for w in weights)
    n_expected = sum(w[2] for w in weights if w[0] not in skipped)
    in_bytes = sum(w[3] for w in weights) + other

    print("\nINPUT MODEL  (already binarized, as PyTorch stores it)")
    print(f"{'WEIGHT TENSOR':<36}{'SHAPE':<16}{'WEIGHTS':>9}{'FP32 KB':>10}")
    print("-" * 71)
    for name, shape, n, b in weights:
        label = name + ("*" if name in skipped else "")
        print(f"{label:<36}{'x'.join(map(str, shape)):<16}{n:>9,}{_kb(b):>10.2f}")
    print(f"{'other (BatchNorm, RPReLU, thresholds)':<61}{_kb(other):>10.2f}")
    print("-" * 71)
    print(f"{'TOTAL':<52}{n_in:>9,}{_kb(in_bytes):>10.2f}")
    if any(w[0] in skipped for w in weights):
        print("  * not exported (unused at inference)")

    print("\nOUTPUT  (header for the engine)")
    print(f"{'LAYER':<24}{'SHAPE':<18}{'PADDING':>8}{'WEIGHTS KB':>12}{'PARAMS KB':>11}{'TOTAL KB':>10}")
    print("-" * 83)
    for r in rows:
        pad = 100 * (1 - r["n_weights"] * r["bits"] / 8 / r["weights_bytes"])
        print(f"{r['name']:<24}{r['shape']:<18}{pad:>7.0f}%{_kb(r['weights_bytes']):>12.2f}"
              f"{_kb(r['params_bytes']):>11.2f}{_kb(r['weights_bytes'] + r['params_bytes']):>10.2f}")

    n_out = sum(r["n_weights"] for r in rows)
    w_bytes = sum(r["weights_bytes"] for r in rows)
    p_bytes = sum(r["params_bytes"] for r in rows)
    minimum = sum(r["n_weights"] * r["bits"] / 8 for r in rows)
    rom = w_bytes + p_bytes
    print("-" * 83)
    print(f"{'TOTAL':<42}{'':>8}{_kb(w_bytes):>12.2f}{_kb(p_bytes):>11.2f}{_kb(rom):>10.2f}")

    print("\nMEMORY")
    print(f"  ROM (flash) : {_kb(rom):7.1f} KB  = weights {_kb(minimum):.1f} + padding {_kb(w_bytes - minimum):.1f}"
          f" + params {_kb(p_bytes):.1f}    ({in_bytes / rom:.0f}x smaller than the FP32 input, {_kb(in_bytes):.1f} KB)")
    mapped = [r for r in rows if "map_bytes" in r]
    if mapped:
        big = max(mapped, key=lambda r: r["map_bytes"])
        print(f"  RAM         : {_kb(big['map_bytes']):7.1f} KB  per activation buffer (largest: {big['name']});"
              f" the engine keeps several of them")
    print(f"  header file : {_kb(os.path.getsize(out_path)):7.1f} KB  on disk (text, not flash)")
    if n_expected != n_out:
        print(f"  [!] the model has {n_expected:,} weights to export but {n_out:,} were exported: some layer was skipped")

    if not detail:
        return

    # ---- ROM by array
    parts = {}
    for r in rows:
        for k, v in r.get("parts", {}).items():
            parts[k] = parts.get(k, 0) + v
    print("\nROM BY ARRAY")
    print(f"{'ARRAY':<44}{'KB':>9}{'%':>8}")
    print("-" * 61)
    items = [("weights, packed at their bit width", minimum), ("alignment padding", w_bytes - minimum)]
    items += list(parts.items())
    for label, b in items:
        print(f"{label:<44}{_kb(b):>9.2f}{100 * b / rom:>7.1f}%")
    print("-" * 61)
    print(f"{'TOTAL':<44}{_kb(rom):>9.2f}{100:>7.1f}%")

    # ---- by stage (stage = prefix of the layer name: STEM, LAYER1, LAYER2, ..., CLASIFICADOR)
    stages = {}
    for r in rows:
        d = stages.setdefault(r["name"].split("_")[0], dict(layers=0, rom=0, map=0, xnor=0))
        d["layers"] += 1
        d["rom"] += r["weights_bytes"] + r["params_bytes"]
        d["map"] = max(d["map"], r.get("map_bytes", 0))
        d["xnor"] += r.get("xnor_words", 0)
    total_x = sum(d["xnor"] for d in stages.values())
    print("\nBY STAGE")
    print(f"{'STAGE':<16}{'LAYERS':>7}{'ROM KB':>9}{'MAP KB':>9}{'XNOR WORDS':>13}{'COMPUTE':>9}")
    print("-" * 63)
    for name, d in stages.items():
        share = f"{100 * d['xnor'] / total_x:>8.1f}%" if total_x else f"{'-':>9}"
        print(f"{name:<16}{d['layers']:>7}{_kb(d['rom']):>9.2f}{_kb(d['map']):>9.1f}{d['xnor']:>13,}{share}")
    print("-" * 63)
    print(f"{'TOTAL':<16}{sum(d['layers'] for d in stages.values()):>7}{_kb(rom):>9.2f}"
          f"{'':>9}{total_x:>13,}")
