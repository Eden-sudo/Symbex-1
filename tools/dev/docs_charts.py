"""
docs_charts.py -- the charts of docs/, drawn from the result files of the benchmarks.

    python docs_charts.py            # writes docs/assets/*.svg (light and dark) and prints the numbers it used

No plotting library: the SVG is written directly, so the charts can be re-created with plain Python anywhere.
Every number comes from a JSON file under benchmarks/*/results/ (the paths are in SOURCES and printed with each row);
nothing is typed in by hand. When a benchmark is re-measured, run this again and the charts follow.

Charts:
  speedup.svg   how many times faster the fully binary network runs than the official INT8 one (TensorFlow Lite
                Micro), same network, same chip, weights in flash for both (plus the MLP with weights in RAM)
  flash.svg     flash used by each model, SYMBEX vs. INT8 (the whole model: weights, thresholds, scales)
Each one also as <name>_dark.svg, for GitHub's dark theme (the pages pick one with <picture>).
"""
import json
import os
import statistics
from xml.sax.saxutils import escape

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BENCH = os.path.join(ROOT, "benchmarks")
OUT_DIR = os.path.join(ROOT, "docs", "assets")

RESNET = "conv2d/cnn_cifar_resnet8/results"
FOMO = "conv2d/fomo_voc_person/results"
FASHION = "mlp/mlp_fashion/results"
SOURCES = {
    "resnet_s3": f"{RESNET}/symbex_seeds_s3_swar_w1.json",
    "resnet_s3_int8": f"{RESNET}/tflite_s3.json",
    "resnet_esp32": f"{RESNET}/symbex_seeds_esp32_lut16_w1.json",
    "resnet_esp32_int8": f"{RESNET}/tflite_esp32.json",
    "fomo_s3": f"{FOMO}/symbex_seeds_s3_swar_c32-32-128_b1-2-2_w40_e100.json",
    "fomo_s3_int8": f"{FOMO}/tflite_s3_c32-32-128_b1-2-2_w20_e60.json",
    "fomo_esp32": f"{FOMO}/symbex_seeds_esp32_lut16_c32-32-128_b1-2-2_w40_e100.json",
    "mlp_sizing": f"{FASHION}/sizing/fashion_sizing.json",
    "mlp_s3_int8": f"{FASHION}/tflite/tflite_s3_h128_512.json",
    "mlp_s3_int8_ram": f"{FASHION}/tflite/tflite_s3_h128_512_ram.json",
    "mlp_esp32": f"{FASHION}/symbex/symbex_h128_512_s42_chip_esp32_swar_flash.json",
    "mlp_esp32_int8": f"{FASHION}/tflite/tflite_esp32_h128_512.json",
}
MLP_SHAPE = "784-128-512-10"

# Colors: the first two slots of a color-blind-checked categorical palette, stepped for each theme. Text never takes a
# series color.
THEMES = {
    "light": dict(text="#0b0b0b", muted="#52514e", grid="#e4e3df", base="#8f8e88",
                  series=["#2a78d6", "#eb6834"]),
    "dark": dict(text="#f0efec", muted="#c3c2b7", grid="#3a3a38", base="#8f8e88",
                 series=["#3987e5", "#d95926"]),
}
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def load(key):
    with open(os.path.join(BENCH, SOURCES[key])) as f:
        return json.load(f)


def seeds_mean(d, field):
    return statistics.mean(s[field] for s in d["seeds"])


def mlp_row():
    d = load("mlp_sizing")
    return next(r for r in d["rows"] if r["shape"] == MLP_SHAPE)


def speedup_rows():
    """(label, SYMBEX ms, INT8 ms or None when INT8 does not fit, source keys)."""
    mlp = mlp_row()
    return [
        ("MLP · ESP32-S3", mlp["ms_measured"]["s3 flash"], load("mlp_s3_int8")["median_ms"],
         ["mlp_sizing", "mlp_s3_int8"]),
        ("MLP · ESP32-S3, weights in RAM", mlp["ms_measured"]["s3 ram"], load("mlp_s3_int8_ram")["median_ms"],
         ["mlp_sizing", "mlp_s3_int8_ram"]),
        ("MLP · ESP32", load("mlp_esp32")["median_ms"], load("mlp_esp32_int8")["median_ms"],
         ["mlp_esp32", "mlp_esp32_int8"]),
        ("ResNet-8 · ESP32-S3", seeds_mean(load("resnet_s3"), "latency_ms"), load("resnet_s3_int8")["latency_ms"],
         ["resnet_s3", "resnet_s3_int8"]),
        ("ResNet-8 · ESP32", seeds_mean(load("resnet_esp32"), "latency_ms"),
         load("resnet_esp32_int8")["latency_ms"], ["resnet_esp32", "resnet_esp32_int8"]),
        ("FOMO person · ESP32-S3", seeds_mean(load("fomo_s3"), "latency_ms"), load("fomo_s3_int8")["latency_ms"],
         ["fomo_s3", "fomo_s3_int8"]),
        ("FOMO person · ESP32", seeds_mean(load("fomo_esp32"), "latency_ms"), None, ["fomo_esp32"]),
    ]


def flash_rows():
    """(label, SYMBEX KB, INT8 KB, source keys)."""
    return [
        (f"MLP {MLP_SHAPE}", mlp_row()["flash_kb"], load("mlp_s3_int8")["flash_kb"], ["mlp_sizing", "mlp_s3_int8"]),
        ("ResNet-8 (CIFAR-10)", seeds_mean(load("resnet_s3"), "flash_kb"), load("resnet_s3_int8")["model_kb"],
         ["resnet_s3", "resnet_s3_int8"]),
        ("FOMO person detection", seeds_mean(load("fomo_s3"), "flash_kb"), load("fomo_s3_int8")["model_kb"],
         ["fomo_s3", "fomo_s3_int8"]),
    ]


def ms(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def text(x, y, s, color, size=13, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(s)}</text>')


def bar(x, y, w, h, color, r=4):
    """A bar anchored at x, its data end (the right one) rounded."""
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{color}"/>')


def svg(width, height, title, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT}" role="img">\n'
            f'<title>{escape(title)}</title>\n' + "\n".join(body) + "\n</svg>\n")


def axis_ticks(vmax, step):
    return [step * i for i in range(int(vmax // step) + 1)]


def speedup_svg(rows, th):
    width, left, right, top, row_h = 820, 230, 200, 80, 34
    height = top + row_h * len(rows) + 40
    vmax = max(i / s for _, s, i, _ in rows if i is not None)
    vmax = 5 * (int(vmax // 5) + 1)
    plot_w = width - left - right
    xs = lambda v: left + plot_w * v / vmax
    body = [text(16, 26, "Latency on the chip: how many times faster than INT8", th["text"], 16, weight="bold"),
            text(16, 46, "Same network, same chip, TensorFlow Lite Micro INT8; weights in flash unless noted",
                 th["muted"], 12)]
    bottom = top + row_h * len(rows)
    for t in axis_ticks(vmax, 5):
        body.append(f'<line x1="{xs(t):.1f}" y1="{top - 6}" x2="{xs(t):.1f}" y2="{bottom}" stroke="{th["grid"]}"/>')
        body.append(text(xs(t), bottom + 18, f"{t}×", th["muted"], 11, "middle"))
    for k, (label, s, i, _) in enumerate(rows):
        y = top + row_h * k
        body.append(text(left - 12, y + row_h / 2 + 4, label, th["text"], 13, "end"))
        if i is None:
            body.append(text(left + 6, y + row_h / 2 + 4, f"SYMBEX {ms(s)} ms · INT8 does not fit in the chip",
                             th["muted"], 12))
            continue
        ratio = i / s
        factor = f"{ratio:.1f}×" if ratio >= 10 else f"{ratio:.2f}×"
        body.append(bar(left, y + 8, xs(ratio) - left, row_h - 16, th["series"][0]))
        body.append(text(xs(ratio) + 8, y + row_h / 2 + 4, factor, th["text"], 13, weight="bold"))
        body.append(text(xs(ratio) + 16 + 8.5 * len(factor), y + row_h / 2 + 4, f"{ms(s)} vs {ms(i)} ms",
                         th["muted"], 11))       # 8.5 px: about one bold 13 px character
    body.append(f'<line x1="{xs(1):.1f}" y1="{top - 6}" x2="{xs(1):.1f}" y2="{bottom}" stroke="{th["base"]}" '
                f'stroke-dasharray="3 3"/>')
    body.append(text(xs(1) + 4, top - 10, "1× = as fast as INT8", th["muted"], 10))
    return svg(width, height, "SYMBEX speed-up over TensorFlow Lite Micro INT8", body)


def flash_svg(rows, th):
    width, left, right, top, group_h, bh = 820, 230, 90, 84, 62, 18
    height = top + group_h * len(rows) + 36
    vmax = max(i for _, _, i, _ in rows)
    vmax = 50 * (int(vmax // 50) + 1)
    plot_w = width - left - right
    xs = lambda v: left + plot_w * v / vmax
    body = [text(16, 26, "Flash used by the model", th["text"], 16, weight="bold"),
            text(16, 46, "KB, the whole model (weights, thresholds, scales); INT8 = the .tflite file", th["muted"], 12)]
    for k, name in enumerate(["SYMBEX (1 bit)", "INT8 (TFLite Micro)"]):
        lx = left + 170 * k
        body.append(f'<rect x="{lx}" y="58" width="12" height="12" rx="2" fill="{th["series"][k]}"/>')
        body.append(text(lx + 18, 69, name, th["text"], 12))
    bottom = top + group_h * len(rows)
    for t in axis_ticks(vmax, 50):
        body.append(f'<line x1="{xs(t):.1f}" y1="{top - 4}" x2="{xs(t):.1f}" y2="{bottom}" stroke="{th["grid"]}"/>')
        body.append(text(xs(t), bottom + 18, f"{t}", th["muted"], 11, "middle"))
    for k, (label, s, i, _) in enumerate(rows):
        y = top + group_h * k + 10
        body.append(text(left - 12, y + bh + 5, label, th["text"], 13, "end"))
        for j, v in enumerate((s, i)):
            yy = y + j * (bh + 2)                   # 2 px gap between the two bars of a group
            body.append(bar(left, yy, xs(v) - left, bh, th["series"][j]))
            note = f"{v:.1f}" + (f"  ({i / s:.1f}× less)" if j == 0 else "")
            body.append(text(xs(v) + 6, yy + bh - 4, note, th["text"] if j == 0 else th["muted"], 12,
                             weight="bold" if j == 0 else "normal"))
    return svg(width, height, "Flash used by the model: SYMBEX vs. INT8", body)


def write(name, content):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"    -> {os.path.relpath(path, ROOT)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    speed, flash = speedup_rows(), flash_rows()
    print("\n    speed-up (SYMBEX ms / INT8 ms)")
    for label, s, i, keys in speed:
        print(f"    {label:<34}{ms(s):>8}{ms(i) if i else 'no fit':>9}  "
              f"{(f'{i / s:.2f}x') if i else '-':>7}  {', '.join(SOURCES[k] for k in keys)}")
    print("\n    flash KB (SYMBEX / INT8)")
    for label, s, i, keys in flash:
        print(f"    {label:<34}{s:>8.1f}{i:>9.1f}  {i / s:>6.1f}x  {', '.join(SOURCES[k] for k in keys)}")
    print()
    for mode, th in THEMES.items():
        suffix = "" if mode == "light" else "_dark"
        write(f"speedup{suffix}.svg", speedup_svg(speed, th))
        write(f"flash{suffix}.svg", flash_svg(flash, th))


if __name__ == "__main__":
    main()
