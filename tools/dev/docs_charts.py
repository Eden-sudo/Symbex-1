"""
docs_charts.py -- the charts of docs/, drawn from the result files of the benchmarks.

    python docs_charts.py            # writes docs/assets/*.svg (light and dark) and prints the numbers it used

No plotting library: the SVG is written directly, so the charts can be re-created with plain Python anywhere.
Every number comes from a JSON file under benchmarks/*/results/ (the paths are in SOURCES and printed with each row);
nothing is typed in by hand. When a benchmark is re-measured, run this again and the charts follow.

Charts:
  speedup.svg   how many times faster the fully binary network runs than the official INT8 one (TensorFlow Lite
                Micro), same network, same chip, weights in flash for both (plus the MLP with weights in RAM); the
                ECG row is its large 1D network (the benchmark's other two are in its README)
  flash.svg     flash used by each model, SYMBEX vs. INT8 (the whole model: weights, thresholds, scales)
  pareto_mlp.svg  accuracy against latency, MLP on Fashion-MNIST, ESP32-S3, weights in flash and in RAM: INT8 /
                INT4 / INT2 (tools/lowbit, QAT) and SYMBEX networks sized for binary, the same recipe (validation,
                3 seeds, distillation), with the Pareto front (docs/en/methodology.md)
  pareto_resnet8.svg  accuracy against latency, ResNet-8 on CIFAR-10, ESP32-S3 and ESP32, weights in flash: the
                official MLPerf Tiny INT8 model (TFLite Micro) and the fully binary ResNet-8s (test accuracy, 3 seeds)
  cost.svg      the accuracy binary costs (INT8 minus SYMBEX, points of each task's metric on its test set) against the
                speed-up of speedup.svg, one line per benchmark from the ESP32 to the ESP32-S3 (activity recognition
                is left out: its SYMBEX chip times are kept in runs/, not in results/)
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
    "resnet_s3": f"{RESNET}/symbex/symbex_seeds_s3_swar_w1.json",
    "resnet_s3_int8": f"{RESNET}/tflite/tflite_s3.json",
    "resnet_esp32": f"{RESNET}/symbex/symbex_seeds_esp32_lut16_w1.json",
    "resnet_esp32_int8": f"{RESNET}/tflite/tflite_esp32.json",
    "fomo_s3": f"{FOMO}/symbex/symbex_seeds_s3_swar_c32-32-128_b1-2-2_w40_e100.json",
    "fomo_s3_int8": f"{FOMO}/tflite/tflite_s3_c32-32-128_b1-2-2_w20_e60.json",
    "fomo_esp32": f"{FOMO}/symbex/symbex_seeds_esp32_lut16_c32-32-128_b1-2-2_w40_e100.json",
    "mlp_sizing": f"{FASHION}/sizing/fashion_sizing.json",
    "mlp_s3_int8": f"{FASHION}/tflite/tflite_s3_h128_512.json",
    "mlp_s3_int8_ram": f"{FASHION}/tflite/tflite_s3_h128_512_ram.json",
    "mlp_esp32": f"{FASHION}/symbex/symbex_h128_512_s42_chip_esp32_swar_flash.json",
    "mlp_esp32_int8": f"{FASHION}/tflite/tflite_esp32_h128_512.json",
    "mlp_lowbit": f"{FASHION}/lowbit/lowbit.json",
    "mlp_lowbit_s3_flash": f"{FASHION}/lowbit/lowbit_esp_nn_s3_flash.json",
    "mlp_lowbit_s3_ram": f"{FASHION}/lowbit/lowbit_esp_nn_s3_ram.json",
    "resnet_pc_int8": f"{RESNET}/tflite/tflite_pc.json",
    "resnet_s3_vertical": f"{RESNET}/symbex/symbex_seeds_s3_swar_w1_vertical.json",
    "mlp_sizing_labels": f"{FASHION}/sizing/fashion_sizing_nodistill.json",
    "mlp_int8_qat_labels": f"{FASHION}/lowbit/int8_e15_h128_512_nodistill_pc.json",
}
# the ResNet-8s of the Pareto chart: (label, file name part); the boards' files are symbex_seeds_<board>_<name>.json
RESNET_PARETO = {
    "s3": [("binary", "swar_w1"), ("binary, vertical", "swar_w1_vertical"),
           ("16-32-128", "swar_c16-32-128_b1-1-2"), ("int8 stem", "swar_w1_int8stem_pool2"),
           ("int8 stem, vertical", "swar_w1_int8stem_pool2_vertical")],
    "esp32": [("binary", "lut16_w1"), ("16-32-128", "lut16_c16-32-128_b1-1-2"),
              ("int8 stem", "lut16_w1_int8stem_pool2")],
}
for _board, _rows in RESNET_PARETO.items():
    for _label, _name in _rows:
        SOURCES[f"resnet_{_board}_{_name}"] = f"{RESNET}/symbex/symbex_seeds_{_board}_{_name}.json"
MLP_SHAPE = "784-128-512-10"
# the ECG benchmark writes one file per seed and board (3 seeds, RR recipe): these keys hold a list of files
ECG = "conv1d/ecg_mitbih/results"
ECG_SEEDS = (42, 43, 44)
SOURCES.update({
    "ecg_s3": [f"{ECG}/symbex/l_binary_e100_s{s}_rr/chip_s3_swar_flash_mask.json" for s in ECG_SEEDS],
    "ecg_s3_int8": [f"{ECG}/tflite/tflite_s3_l_s{s}_rr.json" for s in ECG_SEEDS],
    "ecg_esp32": [f"{ECG}/symbex/l_binary_e100_s{s}_rr/chip_esp32_swar_flash_mask.json" for s in ECG_SEEDS],
    "ecg_esp32_int8": [f"{ECG}/tflite/tflite_esp32_l_s{s}_rr.json" for s in ECG_SEEDS],
    "ecg_train": [f"{ECG}/symbex/l_binary_e100_s{s}_rr/train.json" for s in ECG_SEEDS],
    "ecg_pc_int8": [f"{ECG}/tflite/tflite_pc_l_s{s}_rr.json" for s in ECG_SEEDS],
    "fomo_pc_int8": [f"{FOMO}/tflite/tflite_pc_c32-32-128_b1-2-2_w20_e60{x}.json" for x in ("", "_s43", "_s44")],
})

# Colors: the first four slots of a color-blind-checked categorical palette, stepped for each theme (SYMBEX, INT8,
# INT4, INT2: the first two as in every chart). Text never takes a series color; every point of the Pareto chart is
# labelled, because INT4's and INT2's colors are below 3:1 on the light background.
THEMES = {
    "light": dict(text="#0b0b0b", muted="#52514e", grid="#e4e3df", base="#8f8e88",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], surface="#ffffff"),
    "dark": dict(text="#f0efec", muted="#c3c2b7", grid="#3a3a38", base="#8f8e88",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"], surface="#0d1117"),
}
THEME_SLOT = {"SYMBEX": 0, "INT8": 1, "INT4": 2, "INT2": 3}    # a method keeps its color in every chart
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def load(key):
    with open(os.path.join(BENCH, SOURCES[key])) as f:
        return json.load(f)


def files_mean(key, field):
    """Mean of one field over a key that lists one file per seed; a field "a.b" reads d["a"]["b"]."""
    values = []
    for path in SOURCES[key]:
        with open(os.path.join(BENCH, path)) as f:
            v = json.load(f)
        for part in field.split("."):
            v = v[part]
        values.append(v)
    return statistics.mean(values)


def source(key):
    """The file behind a key, for the printed table; a per-seed list is shown as one pattern."""
    v = SOURCES[key]
    if isinstance(v, str):
        return v
    one = v[0].replace(f"s{ECG_SEEDS[0]}", "s{" + ",".join(map(str, ECG_SEEDS)) + "}")
    return one if one != v[0] else f"{v[0]} (+ {len(v) - 1} more seeds)"


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
        ("ResNet-8 · ESP32-S3, vertical", seeds_mean(load("resnet_s3_vertical"), "latency_ms"),
         load("resnet_s3_int8")["latency_ms"], ["resnet_s3_vertical", "resnet_s3_int8"]),
        ("ResNet-8 · ESP32", seeds_mean(load("resnet_esp32"), "latency_ms"),
         load("resnet_esp32_int8")["latency_ms"], ["resnet_esp32", "resnet_esp32_int8"]),
        ("FOMO person · ESP32-S3", seeds_mean(load("fomo_s3"), "latency_ms"), load("fomo_s3_int8")["latency_ms"],
         ["fomo_s3", "fomo_s3_int8"]),
        ("FOMO person · ESP32", seeds_mean(load("fomo_esp32"), "latency_ms"), None, ["fomo_esp32"]),
        ("ECG beats, large 1D net · ESP32-S3", files_mean("ecg_s3", "median_ms"), files_mean("ecg_s3_int8", "median_ms"),
         ["ecg_s3", "ecg_s3_int8"]),
        ("ECG beats, large 1D net · ESP32", files_mean("ecg_esp32", "median_ms"),
         files_mean("ecg_esp32_int8", "median_ms"), ["ecg_esp32", "ecg_esp32_int8"]),
    ]


def flash_rows():
    """(label, SYMBEX KB, INT8 KB, source keys)."""
    return [
        (f"MLP {MLP_SHAPE}", mlp_row()["flash_kb"], load("mlp_s3_int8")["flash_kb"], ["mlp_sizing", "mlp_s3_int8"]),
        ("ResNet-8 (CIFAR-10)", seeds_mean(load("resnet_s3"), "flash_kb"), load("resnet_s3_int8")["model_kb"],
         ["resnet_s3", "resnet_s3_int8"]),
        ("FOMO person detection", seeds_mean(load("fomo_s3"), "flash_kb"), load("fomo_s3_int8")["model_kb"],
         ["fomo_s3", "fomo_s3_int8"]),
        ("ECG beats, large 1D network", files_mean("ecg_s3", "model_flash") / 1024, files_mean("ecg_s3_int8", "model_kb"),
         ["ecg_s3", "ecg_s3_int8"]),
    ]


def cost_rows(speed):
    """(benchmark, metric, INT8 score, SYMBEX score, {chip: speed-up}, source keys). Scores: the test set, mean of 3
    seeds, the same network and the same recipe for both methods; the speed-ups are the rows of speedup.svg (weights
    in flash), so a chip where INT8 does not fit has no point."""
    up = {label: i / s for label, s, i, _ in speed if i is not None}
    chips = lambda name: {c: up[f"{name} · {c}"] for c in ("ESP32", "ESP32-S3") if f"{name} · {c}" in up}
    mlp = next(r for r in load("mlp_sizing_labels")["rows"] if r["shape"] == MLP_SHAPE)
    fomo = load("fomo_s3")
    return [
        ("MLP, Fashion-MNIST", "accuracy", load("mlp_int8_qat_labels")["metodos"]["qat"]["precision_10k"], mlp["test"],
         chips("MLP"), ["mlp_int8_qat_labels", "mlp_sizing_labels"]),
        ("ResNet-8, CIFAR-10", "accuracy", load("resnet_pc_int8")["int8_accuracy"],
         seeds_mean(load("resnet_s3"), "test_accuracy"), chips("ResNet-8"), ["resnet_pc_int8", "resnet_s3"]),
        ("FOMO, person detection", "F1", files_mean("fomo_pc_int8", "f1_calibrated.f1"), seeds_mean(fomo, "f1_test"),
         chips("FOMO person"), ["fomo_pc_int8", "fomo_s3"]),
        ("ECG beats, large 1D net", "balanced accuracy", files_mean("ecg_pc_int8", "test.balanced"),
         files_mean("ecg_train", "test.balanced"), chips("ECG beats, large 1D net"), ["ecg_pc_int8", "ecg_train"]),
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


def cost_svg(rows, th):
    """One horizontal line per benchmark at its accuracy cost, from its ESP32 speed-up (filled dot) to its ESP32-S3
    one (ring); x on a log scale. Lower is more accurate, further right is faster."""
    import math
    width, left, right, top, ph = 820, 80, 40, 118, 330
    height = top + ph + 58
    pw = width - left - right
    costs = [i - b for _, _, i, b, _, _ in rows]
    ylo, yhi = 5 * math.floor((min(costs) - 4) / 5), 5 * math.ceil(max(costs) / 5 + 0.2)   # room for labels below
    ys = lambda v: top + ph * (yhi - v) / (yhi - ylo)
    lo, hi = 1, 50
    xs = lambda v: left + pw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    body = [text(16, 26, "What binary costs and what it buys", th["text"], 16, weight="bold"),
            text(16, 46, "Points lost against INT8 (each task's metric, test set, mean of 3 seeds) and speed-up on "
                 "the chip (weights in flash)", th["muted"], 12),
            text(16, 63, "Same network and same recipe for both methods; lower is more accurate, further right is faster",
                 th["muted"], 12)]
    c = th["series"][THEME_SLOT["SYMBEX"]]
    body.append(f'<circle cx="{left + 6}" cy="84" r="5" fill="{c}"/>')
    body.append(text(left + 16, 88, "ESP32 (no SIMD)", th["text"], 12))
    body.append(f'<circle cx="{left + 146}" cy="84" r="4" fill="{th["surface"]}" stroke="{c}" stroke-width="2"/>')
    body.append(text(left + 156, 88, "ESP32-S3 (SIMD)", th["text"], 12))
    for t in range(ylo, yhi + 1, 5):
        body.append(f'<line x1="{left}" y1="{ys(t):.1f}" x2="{left + pw}" y2="{ys(t):.1f}" stroke="{th["grid"]}"/>')
        body.append(text(left - 8, ys(t) + 4, f"{t:+d}" if t else "0", th["muted"], 11, "end"))
    for v in (1, 2, 5, 10, 20, 50):
        body.append(f'<line x1="{xs(v):.1f}" y1="{top}" x2="{xs(v):.1f}" y2="{top + ph}" stroke="{th["grid"]}"/>')
        body.append(text(xs(v), top + ph + 16, f"{v}×", th["muted"], 11, "middle"))
    body.append(text(left + pw / 2, top + ph + 36, "times faster than TFLite Micro INT8 (log scale)", th["muted"], 11,
                     "middle"))
    body.append(text(16, top - 10, "points lost (INT8 − SYMBEX)", th["muted"], 11))
    body.append(f'<line x1="{left}" y1="{ys(0):.1f}" x2="{left + pw}" y2="{ys(0):.1f}" stroke="{th["base"]}" '
                f'stroke-dasharray="3 3"/>')
    body.append(text(left + 4, ys(0) - 6, "as accurate as INT8", th["muted"], 10))
    body.append(f'<line x1="{xs(1):.1f}" y1="{top}" x2="{xs(1):.1f}" y2="{top + ph}" stroke="{th["base"]}" '
                f'stroke-dasharray="3 3"/>')
    placed = []
    for name, metric, i, b, up, _ in sorted(rows, key=lambda r: -(r[2] - r[3])):
        y = ys(i - b)
        pts = sorted(up.items(), key=lambda kv: kv[1])
        x1, x2 = xs(pts[0][1]), xs(pts[-1][1])
        if len(pts) > 1:
            body.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2" '
                        f'stroke-opacity="0.5"/>')
        for chip, v in pts:
            x = xs(v)
            tip = f"{name} · {chip}: {v:.2f}× faster, INT8 {i:.1f} vs SYMBEX {b:.1f} ({metric})"
            fill, r = (c, 5) if chip == "ESP32" else (th["surface"], 4)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{c if chip != "ESP32" else th["surface"]}" '
                        f'stroke-width="2"><title>{escape(tip)}</title></circle>')
            body.append(text(x, y + 18, f"{v:.1f}×" if v >= 10 else f"{v:.2f}×", th["muted"], 11, "middle"))
        s = f"{name} · {metric}: SYMBEX {b:.1f} %, INT8 {i:.1f} %"
        w = 6.6 * len(s)
        if len(pts) == 1:                               # one dot: the label on its right, clear of the zero line
            box = (x1 + 10, y - 6, x1 + 10 + w, y + 6)
        else:
            bx = max(left + 4, min((x1 + x2) / 2 - w / 2, left + pw - w))
            box = (bx, y - 20, bx + w, y - 8)
            if any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3] for q in placed):
                box = (bx, y + 22, bx + w, y + 34)      # the label above collides: put it under the numbers
        placed.append(box)
        placed.append((x1 - 20, y + 6, x2 + 20, y + 22))  # the speed-up numbers under the dots
        body.append(text(box[0], box[3] - 1, s, th["text"], 12))
    return svg(width, height, "Accuracy cost against speed-up per benchmark, ESP32 and ESP32-S3", body)


def pareto_points(placement):
    """(method, label, ms, validation mean, sd, source keys) for the MLP on the ESP32-S3, weights in flash or ram.
    INT8 / INT4 / INT2: tools/lowbit QAT of 784-256-128-10, accuracy over 3 seeds, latency of seed 42 (dense
    multiply-accumulate: the time does not depend on the weight values). SYMBEX: the networks sized for binary, same
    recipe (15 epochs, distillation, 3 seeds, validation), latency measured."""
    low, chip = load("mlp_lowbit"), load(f"mlp_lowbit_s3_{placement}")
    shape = low["shape"].rsplit("-", 1)[0]
    pts = []
    for bits in ("8", "4", "2"):
        v = [s["validation"] for s in low["bits"][bits]["qat"]["seeds"]]
        t = next(r["latency_ms"] for r in chip if r["bits"] == int(bits))
        pts.append((f"INT{bits}", f"INT{bits}", t, statistics.mean(v), statistics.stdev(v),
                    ["mlp_lowbit", f"mlp_lowbit_s3_{placement}"]))
    for r in load("mlp_sizing")["rows"]:
        if f"s3 {placement}" in (r.get("ms_measured") or {}):
            pts.append(("SYMBEX", r["shape"].rsplit("-", 1)[0].split("-", 1)[1], r["ms_measured"][f"s3 {placement}"],
                        r["val"], r["val_sd"], ["mlp_sizing"]))
    return pts


def pareto_points_resnet(board):
    """(method, label, ms, test mean, sd, source keys) for ResNet-8 on a board, weights in flash. INT8: the official
    MLPerf Tiny model, one model (sd 0), test accuracy of its .tflite on the PC (on the chip TFLite Micro gives the
    PC interpreter's class on 98.1 % of 1,000 images). SYMBEX: 3 seeds, test accuracy of the trained model on the PC
    (train.json; the exported C gives the integer simulation's class on 100 % of the images). Latency: on the chip."""
    pts = [("INT8", "INT8 · MLPerf", load(f"resnet_{board}_int8")["latency_ms"], load("resnet_pc_int8")["int8_accuracy"],
            0.0, [f"resnet_{board}_int8", "resnet_pc_int8"])]
    for label, name in RESNET_PARETO[board]:
        d = load(f"resnet_{board}_{name}")
        acc = [s["test_accuracy"] for s in d["seeds"]]
        pts.append(("SYMBEX", label, seeds_mean(d, "latency_ms"), statistics.mean(acc), statistics.stdev(acc),
                    [f"resnet_{board}_{name}"]))
    return pts


def pareto_front(pts):
    """The points no other point beats on both axes (faster and at least as accurate), fastest first."""
    front, best = [], -1.0
    for p in sorted(pts, key=lambda p: (p[2], -p[3])):
        if p[3] > best:
            front.append(p)
            best = p[3]
    return front


MLP_TITLE = ("Accuracy against latency: where one bit sits",
             "MLP, Fashion-MNIST, ESP32-S3 · validation accuracy, mean ± sd of 3 seeds, the same recipe",
             "INT8 / INT4 / INT2: 784-256-128, QAT, ESP-NN kernels · SYMBEX: networks sized for binary",
             "Accuracy against latency, MLP on the ESP32-S3: SYMBEX vs. INT8 / INT4 / INT2")
RESNET_TITLE = ("ResNet-8 on CIFAR-10: speed bought with accuracy",
                "Accuracy on the PC (10,000 test images; SYMBEX mean ± sd of 3 seeds), latency on the chip, weights in flash",
                "INT8: official MLPerf Tiny model (TFLite Micro) · SYMBEX: the same network in 1 bit; 16-32-128 sized for "
                "binary; int8 stem: 8-bit first layer + pooled int8 classifier",
                "Accuracy against latency, ResNet-8 on the ESP32-S3 and the ESP32: SYMBEX vs. INT8")


def pareto_svg(panels, ceiling, th, titles=MLP_TITLE, methods=("SYMBEX", "INT8", "INT4", "INT2"), ystep=0.5,
               label_all=False):
    """ceiling: the FP32 accuracy line, or None; label_all: label every SYMBEX point, not only the front's."""
    import math
    methods = list(methods)
    width, top, pw, ph, gap, left = 900, 124, 370, 300, 60, 70
    height = top + ph + 64
    allp = [p for _, pts in panels for p in pts]
    ylo = math.floor(min(p[3] - p[4] for p in allp) / ystep) * ystep - ystep
    yhi = math.ceil(max(ceiling or 0, max(p[3] + p[4] for p in allp)) / ystep) * ystep
    ys = lambda v: top + ph * (yhi - v) / (yhi - ylo)
    yticks = [ylo + ystep * i for i in range(int(round((yhi - ylo) / ystep)) + 1)]
    body = [text(16, 26, titles[0], th["text"], 16, weight="bold"),
            text(16, 46, titles[1], th["muted"], 12),
            text(16, 63, titles[2], th["muted"], 12)]
    for k, name in enumerate(methods):
        lx = left + 120 * k
        body.append(f'<circle cx="{lx + 6}" cy="84" r="5" fill="{th["series"][THEME_SLOT[name]]}"/>')
        body.append(text(lx + 16, 88, name + (" (1 bit)" if name == "SYMBEX" else ""), th["text"], 12))
    fx = left + max(490, 120 * len(methods) + 10)
    body.append(f'<line x1="{fx}" y1="84" x2="{fx + 24}" y2="84" stroke="{th["muted"]}" stroke-width="2"/>')
    body.append(text(fx + 30, 88, "Pareto front: nothing is both faster and more accurate", th["text"], 12))
    for t in yticks:
        body.append(text(left - 8, ys(t) + 4, f"{t:.1f} %" if ystep < 1 else f"{t:.0f} %", th["muted"], 11, "end"))
    for n, (title, pts) in enumerate(panels):
        x0 = left + n * (pw + gap)
        tmin, tmax = min(p[2] for p in pts), max(p[2] for p in pts)
        steps = [d * m for d in (0.01, 0.1, 1, 10, 100, 1000) for m in (1, 2, 5)]
        lo = max(v for v in steps if v <= tmin * 0.4)                    # and on the left
        hi = min(v for v in steps if v >= tmax * 3.5)                    # room for the labels on the right
        xs = lambda v, x0=x0, lo=lo, hi=hi: x0 + pw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        body.append(text(x0, top - 12, title, th["text"], 13, weight="bold"))
        for t in yticks:
            body.append(f'<line x1="{x0}" y1="{ys(t):.1f}" x2="{x0 + pw}" y2="{ys(t):.1f}" stroke="{th["grid"]}"/>')
        for v in (v for v in steps if lo <= v <= hi):
            body.append(f'<line x1="{xs(v):.1f}" y1="{top}" x2="{xs(v):.1f}" y2="{top + ph}" stroke="{th["grid"]}"/>')
            body.append(text(xs(v), top + ph + 16, f"{v:g}", th["muted"], 11, "middle"))
        body.append(text(x0 + pw / 2, top + ph + 34, "ms per inference (log scale)", th["muted"], 11, "middle"))
        if ceiling is not None:
            body.append(f'<line x1="{x0}" y1="{ys(ceiling):.1f}" x2="{x0 + pw}" y2="{ys(ceiling):.1f}" '
                        f'stroke="{th["base"]}" stroke-dasharray="3 3"/>')
            body.append(text(x0 + 4, ys(ceiling) - 5, f"FP32 {ceiling:.1f} %", th["muted"], 10))
        front = pareto_front(pts)
        path = [f"M{xs(front[0][2]):.1f},{ys(front[0][3]):.1f}"]
        for p in front[1:]:                      # a staircase: each step holds its accuracy until the next point
            path.append(f"H{xs(p[2]):.1f} V{ys(p[3]):.1f}")
        path.append(f"H{x0 + pw:.1f}")
        body.append(f'<path d="{" ".join(path)}" fill="none" stroke="{th["muted"]}" stroke-width="2" '
                    f'stroke-opacity="0.6"/>')
        # labels keep off the dots, the error bars, the front and the FP32 label
        boxes = [(xs(p[2]) - 7, ys(p[3]) - 7, xs(p[2]) + 7, ys(p[3]) + 7) for p in pts]
        boxes += [(xs(p[2]) - 2, ys(p[3] + p[4]), xs(p[2]) + 2, ys(p[3] - p[4])) for p in pts]
        if ceiling is not None:
            boxes.append((x0, ys(ceiling) - 14, x0 + 70, ys(ceiling)))
        for a, b in zip(front, front[1:]):
            boxes.append((xs(a[2]), ys(a[3]) - 2, xs(b[2]), ys(a[3]) + 2))
            boxes.append((xs(b[2]) - 2, ys(b[3]), xs(b[2]) + 2, ys(a[3])))
        boxes.append((xs(front[-1][2]), ys(front[-1][3]) - 2, x0 + pw, ys(front[-1][3]) + 2))
        marks = []
        for p in sorted(pts, key=lambda p: -p[3]):
            c = th["series"][THEME_SLOT[p[0]]]
            x, y = xs(p[2]), ys(p[3])
            body.append(f'<line x1="{x:.1f}" y1="{ys(p[3] + p[4]):.1f}" x2="{x:.1f}" y2="{ys(p[3] - p[4]):.1f}" '
                        f'stroke="{c}" stroke-width="2"/>')
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="{th["surface"]}" '
                        f'stroke-width="2"><title>{escape(p[0] + " " + p[1] if p[0] == "SYMBEX" else p[1])}: {p[3]:.2f} ± {p[4]:.2f} %, {ms(p[2])} ms</title>'
                        f'</circle>')
            marks.append((p, x, y))
        for p, x, y in marks:                     # the label: the first free spot right, left, below or above the dot
            if p[0] == "SYMBEX" and p not in front and not label_all:
                continue                          # selective labels: the front and every INT point (its color alone
                                                  # is below 3:1); the others are in the table of docs/en/methodology.md
            s = f"{p[1]} · {ms(p[2])} ms"
            w = 6.1 * len(s)                      # about one 11 px character
            spots = [(0, "start"), (0, "end"), (18, "middle"), (-14, "middle")]
            spots += [(dy, a) for dy in sorted(range(-60, 61, 12), key=abs) if dy for a in ("start", "end")]
            for dy, anchor in spots:
                bx = {"start": x + 9, "end": x - 9 - w, "middle": x - w / 2}[anchor]
                box = (bx, y + dy - 8, bx + w, y + dy + 4)
                inside = x0 <= box[0] and box[2] <= x0 + pw
                if inside and not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                                      for b in boxes):
                    break
            boxes.append(box)
            if abs(dy) > 12 and anchor != "middle":   # a label away from its dot gets a thin leader line
                ex = box[0] if anchor == "start" else box[2]
                body.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{ex:.1f}" y2="{y + dy:.1f}" stroke="{th["base"]}" '
                            f'stroke-width="1"/>')
            body.append(text(box[0], y + dy + 4, s, th["text"], 11))
    return svg(width, height, titles[3], body)


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
              f"{(f'{i / s:.2f}x') if i else '-':>7}  {', '.join(source(k) for k in keys)}")
    print("\n    flash KB (SYMBEX / INT8)")
    for label, s, i, keys in flash:
        print(f"    {label:<34}{s:>8.1f}{i:>9.1f}  {i / s:>6.1f}x  {', '.join(source(k) for k in keys)}")
    panels = [("ESP32-S3 · weights in flash", pareto_points("flash")), ("ESP32-S3 · weights in RAM", pareto_points("ram"))]
    ceiling = load("mlp_lowbit")["fp32"]["validation"]
    for title, pts in panels:
        print(f"\n    Pareto, MLP, {title} (validation %, ms){'':>4}front: {', '.join(p[0] + ' ' + p[1] if p[0] == 'SYMBEX' else p[1] for p in pareto_front(pts))}")
        for method, label, t, v, sd, keys in pts:
            print(f"    {method + ' ' + label if method == 'SYMBEX' else label:<34}{v:>6.2f} +- {sd:.2f}{ms(t):>8}  {', '.join(SOURCES[k] for k in keys)}")
    resnet = [("ESP32-S3 · weights in flash", pareto_points_resnet("s3")),
              ("ESP32 · weights in flash", pareto_points_resnet("esp32"))]
    for title, pts in resnet:
        print(f"\n    Pareto, ResNet-8, {title} (test %, ms){'':>4}front: {', '.join(p[1] for p in pareto_front(pts))}")
        for method, label, t, v, sd, keys in pts:
            print(f"    {label:<34}{v:>6.2f} +- {sd:.2f}{ms(t):>8}  {', '.join(SOURCES[k] for k in keys)}")
    cost = cost_rows(speed)
    print("\n    accuracy cost (INT8 - SYMBEX, test) and speed-up")
    for name, metric, i, b, up, keys in cost:
        print(f"    {name:<34}{metric:<18}{i:>6.2f} - {b:6.2f} = {i - b:+6.2f}  "
              f"{', '.join(f'{c} {v:.2f}x' for c, v in up.items())}  {', '.join(source(k) for k in keys)}")
    print()
    for mode, th in THEMES.items():
        suffix = "" if mode == "light" else "_dark"
        write(f"speedup{suffix}.svg", speedup_svg(speed, th))
        write(f"flash{suffix}.svg", flash_svg(flash, th))
        write(f"cost{suffix}.svg", cost_svg(cost, th))
        write(f"pareto_mlp{suffix}.svg", pareto_svg(panels, ceiling, th))
        write(f"pareto_resnet8{suffix}.svg", pareto_svg(resnet, None, th, RESNET_TITLE, ("SYMBEX", "INT8"), 2,
                                                         label_all=True))


if __name__ == "__main__":
    main()
