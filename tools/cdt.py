"""
cdt.py -- SYMBEX-1 CDT: command-line entry point of the binarization pipeline.

The CDT itself does no math: it reads the configuration, loads the model and the data, and calls the
library (compiler_core/) stage by stage, printing what each stage does.

Usage (run from tools/):
    python cdt.py configs/example.toml                   # everything from the file
    python cdt.py configs/example.toml --epochs 10       # a flag overrides one value of the file
    python cdt.py configs/example.toml --show-config     # print the final configuration and exit
    python cdt.py --help                                 # every flag, with its default

Priority: built-in defaults < TOML file < command-line flags (a flag always wins).
Flag names are the field names of the file (--epochs, --budget, --method, --file ...). A name that
appears in more than one section carries the section in front: --data-directory, --output-directory.

Stages
    1. inspect    the model: layers, parameters, precision, kind (conv / mlp)
    2. estimate   sensitivity of every layer to every bit option, memory of each option, and the
                  decision (start at 1 bit, raise only when needed, budget as a ceiling)
    3. train      decides the WIDTH and the bits of every layer BY ACCURACY, then trains the final model:
                    a. quick trial ([target] search_epochs) with every layer at 1 bit ("bin"), width x1
                    b. if the accuracy drop vs. the original is <= [target] max_accuracy_drop: done
                    c. WIDTH first: try the next width in [target] widths, all layers still at 1 bit (the hidden
                       layers get wider, copying the original neurons / channels: methods.widen; plain chain
                       networks only, MLPs or CNNs without residual connections)
                    d. then BITS: raise ONE layer to the next option in [target] options, the one with the
                       lowest cosine in the estimate (the most sensitive); repeat the quick trial
                    every step is skipped if it would go over the hardware budget
                    e. full training ([training] epochs) of the chosen width and plan
                  "bin" layers use the plain method (1 bit), k-bit layers the kbit method, both starting from
                  the original weights, with distillation from the original when [training] distill = true.
                  Quick trials are short, so they UNDERestimate the final accuracy: the search may raise a
                  layer the full training would not need.
    4. export     (MLP, every layer at 1 bit) folds the trained model into integers and writes
                  model_symbex.h for lib_symbex (symbex_mlp_infer(const uint8_t* x))
    6. chip       ([chip] board = "esp32" / "s3") builds and flashes an ESP-IDF project with the header and
                  lib_symbex, streams validation images and reports: equal to the integer simulation (must be
                  100%), accuracy and latency measured on the chip. Needs PlatformIO + a connected board.
    5. validate   (runs together with export) FIDELITY: PyTorch vs. the integer simulation (what folding
                  costs); EXACTNESS: the compiled C on this PC vs. the integer simulation (must be 100%)
Choosing the stages ([run] from / to in the TOML, or --from / --to):
    python cdt.py my.toml                          every stage: inspect -> estimate -> train -> export -> chip
    python cdt.py my.toml --to export              create, train, export and validate (no board)
    python cdt.py my.toml --from export            reuse the trained model in the output folder: export, validate, board
    python cdt.py my.toml --from chip --board s3   only the board, with the existing export
Starting after a stage reuses what that stage left in the output directory (train.json + model_binarized.pth,
export.json + model_symbex.h + validation_*.u8). --stop-after is the old name of --to.

Outputs (in the output directory, "<model name>_binarized/" by default)
    config_used.toml   the exact configuration of this run (to reproduce it)
    estimate.json      the result of the estimate stage
    chip.json          accuracy, equality with the simulation and latency measured on the board (chip stage)
    model_binarized.pth  the trained binarized model (weights; checkpoint policy from [training] checkpoint)
    train.json         plan per layer, accuracy of the original and of the binarized model, history
    modelo_symbex.h    the C header for the chip (export stage)
    export.json        fidelity, exactness and memory of the exported model

Requires PyTorch for every stage; --show-config works without it.
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import time

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # pragma: no cover
    tomllib = None

# ---------------------------------------------------------------------------
# Configuration: defaults (same structure and names as configs/example.toml)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "model": {"file": "", "definition": "", "method": "plain"},
    "data": {"directory": "", "format": "npy", "validation_split": 0.1, "calibration_samples": 32,
             "mean": [], "std": [], "input_bits": 8},
    "hardware": {"ram_kb": 320, "flash_kb": 4096, "psram_kb": 0, "cpu_bits": 32},
    "target": {"budget": 0.25, "tolerance": 0.99, "options": ["bin", 2, 4, 8], "max_accuracy_drop": 2.0,
               "search_epochs": 5, "widths": [1.0, 1.5, 2.0]},
    "training": {"epochs": 150, "distill": True, "temperature": 4.0, "alpha": 0.85, "learning_rate": 1e-3,
                 "batch_size": 256, "checkpoint": "last", "seed": 42, "data_on_gpu": False},
    "output": {"directory": "", "verbose": True, "lib_path": "../Symbex1_PIO/lib/lib_symbex",
               "validation_samples": 2000, "arithmetic": "auto", "int8_reference": False},
    "chip": {"board": "", "port": "/dev/ttyUSB0", "baud": 921600, "samples": 1000, "upload": True,
             "popcount": "lut16", "weights": "flash", "compare_arithmetic": False},
    "run": {"from": "inspect", "to": "chip"},
}
# Stages in order. "validate" runs together with "export" (accepted as a name for it).
STAGES = ["inspect", "estimate", "train", "export", "chip"]
STAGE_NAMES = STAGES + ["validate"]
INTEGRATED = list(STAGES)
METHODS = ["plain", "gated", "boolean", "ternary"]
BIN_WEIGHT_BITS = {"ternary": 2}       # weight bits of a "bin" layer per method (default 1: one sign bit)
FORMATS_LOCAL = ["npy", "imagefolder"]


def _flag_names():
    """(section, key) -> flag name. A key that appears in several sections gets the section as prefix."""
    count = {}
    for sec, keys in DEFAULTS.items():
        for k in keys:
            count[k] = count.get(k, 0) + 1
    return {(sec, k): (k if count[k] == 1 else f"{sec}_{k}").replace("_", "-")
            for sec, keys in DEFAULTS.items() for k in keys}


def _parse_list(text):
    """'bin,2,4,8' -> ['bin', 2, 4, 8]; '0.5,0.5,0.5' -> [0.5, 0.5, 0.5]."""
    out = []
    for p in (x.strip() for x in text.split(",") if x.strip()):
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="SYMBEX-1 CDT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("config", nargs="?", default="", help="TOML configuration file (optional)")
    ap.add_argument("--show-config", action="store_true", help="print the final configuration and exit")
    ap.add_argument("--stop-after", choices=STAGE_NAMES, default=None,
                    help="last stage to run (same as --to; kept for the old scripts)")
    for (sec, k), flag in _flag_names().items():
        v = DEFAULTS[sec][k]
        dest = f"{sec}__{k}"
        help_text = f"[{sec}] {k}"
        if isinstance(v, bool):
            ap.add_argument(f"--{flag}", dest=dest, action=argparse.BooleanOptionalAction, default=None, help=help_text)
        elif isinstance(v, list):
            ap.add_argument(f"--{flag}", dest=dest, type=_parse_list, default=None,
                            help=help_text + " (comma separated)")
        else:
            ap.add_argument(f"--{flag}", dest=dest, type=type(v), default=None, help=help_text)
    return ap


def load_config(args):
    """defaults < TOML file < flags. Returns (config, list of 'where each value came from' notes)."""
    cfg = copy.deepcopy(DEFAULTS)
    source = {}
    if args.config:
        if tomllib is None:
            sys.exit("[X] Reading TOML needs Python 3.11 or newer (tomllib).")
        if not os.path.exists(args.config):
            sys.exit(f"[X] Configuration file not found: {args.config}")
        with open(args.config, "rb") as f:
            file_data = tomllib.load(f)
        for sec, keys in file_data.items():
            if sec not in cfg:
                sys.exit(f"[X] {args.config}: unknown section [{sec}] (valid: {', '.join(cfg)})")
            for k, v in keys.items():
                if k not in cfg[sec]:
                    sys.exit(f"[X] {args.config}: unknown field '{k}' in [{sec}] (valid: {', '.join(cfg[sec])})")
                cfg[sec][k] = v
                source[(sec, k)] = "file"
    for sec, keys in DEFAULTS.items():
        for k in keys:
            v = getattr(args, f"{sec}__{k}")
            if v is not None:
                cfg[sec][k] = v
                source[(sec, k)] = "flag"
    return cfg, source


def _stage_index(name):
    return STAGES.index("export" if name == "validate" else name)


def validate_config(cfg):
    """Checks types and ranges before anything heavy runs. Exits with a clear message on the first problem."""
    errors = []
    m, d, h, t, tr = cfg["model"], cfg["data"], cfg["hardware"], cfg["target"], cfg["training"]
    if not m["file"]:
        errors.append("[model] file is empty: the path of the original model is required")
    r = cfg["run"]
    for key in ("from", "to"):
        if r[key] not in STAGE_NAMES:
            errors.append(f"[run] {key} must be one of {STAGES}, got {r[key]!r}")
    if r["from"] in STAGE_NAMES and r["to"] in STAGE_NAMES and \
            _stage_index(r["from"]) > _stage_index(r["to"]):
        errors.append(f"[run] from ({r['from']}) comes after to ({r['to']})")
    if m["method"] not in METHODS:
        errors.append(f"[model] method must be one of {METHODS}, got {m['method']!r}")
    if not d["directory"]:
        errors.append("[data] directory is empty")
    if not (0 < d["validation_split"] < 1):
        errors.append("[data] validation_split must be between 0 and 1")
    if d["calibration_samples"] < 1:
        errors.append("[data] calibration_samples must be >= 1")
    if len(d["mean"]) != len(d["std"]):
        errors.append("[data] mean and std must have the same length (or both be empty)")
    for k in ("ram_kb", "flash_kb"):
        if h[k] <= 0:
            errors.append(f"[hardware] {k} must be > 0")
    if h["psram_kb"] < 0:
        errors.append("[hardware] psram_kb must be >= 0")
    if h["cpu_bits"] not in (8, 16, 32, 64):
        errors.append("[hardware] cpu_bits must be 8, 16, 32 or 64")
    if not (0 < t["budget"] <= 1):
        errors.append("[target] budget must be in (0, 1]")
    if not (0 < t["tolerance"] <= 1):
        errors.append("[target] tolerance must be in (0, 1]")
    for op in t["options"]:
        if op != "bin" and not (isinstance(op, int) and 1 <= op <= 16):
            errors.append(f"[target] options: unknown option {op!r} (use 'bin' or 1-16 bits)")
    if t["max_accuracy_drop"] < 0:
        errors.append("[target] max_accuracy_drop must be >= 0 (accuracy points)")
    if not isinstance(cfg["data"]["input_bits"], int) or not 1 <= cfg["data"]["input_bits"] <= 8:
        errors.append("[data] input_bits must be an integer from 1 to 8")
    if t["search_epochs"] < 1:
        errors.append("[target] search_epochs must be >= 1")
    if not t["widths"] or any((not isinstance(a, (int, float))) or a < 1 for a in t["widths"]):
        errors.append("[target] widths must be a list of numbers >= 1 (e.g. [1.0, 1.5, 2.0])")
    if cfg["output"]["arithmetic"] not in ("auto", "and", "xnor"):
        errors.append("[output] arithmetic must be \"auto\" (default), \"and\" or \"xnor\"")
    if cfg["chip"]["weights"] not in ("flash", "ram"):
        errors.append("[chip] weights must be \"flash\" or \"ram\"")
    if cfg["chip"]["popcount"] not in ("lut16", "swar"):
        errors.append("[chip] popcount must be \"lut16\" (64 KB table, fastest) or \"swar\" (no table, minimum RAM)")
    if cfg["chip"]["board"] not in ("", "esp32", "s3", "uno"):
        errors.append("[chip] board must be \"\" (no chip), \"esp32\", \"s3\" or \"uno\"")
    if tr["checkpoint"] not in ("last", "best"):
        errors.append("[training] checkpoint must be 'last' or 'best'")
    if tr["epochs"] < 1 or tr["batch_size"] < 1:
        errors.append("[training] epochs and batch_size must be >= 1")
    if not (0 <= tr["alpha"] <= 1):
        errors.append("[training] alpha must be in [0, 1]")
    if errors:
        sys.exit("[X] Configuration errors:\n    " + "\n    ".join(errors))


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return repr(v)


def to_toml(cfg):
    """Writes the configuration back as TOML (our structure only: sections of plain values and lists)."""
    lines = []
    for sec, keys in cfg.items():
        lines.append(f"[{sec}]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in keys.items()]
        lines.append("")
    return "\n".join(lines)


def print_config(cfg, source):
    print("\nCONFIGURATION  (value  <- where it came from: default / file / flag)")
    for sec, keys in cfg.items():
        print(f"  [{sec}]")
        for k, v in keys.items():
            print(f"    {k:<20} = {_toml_value(v):<28} <- {source.get((sec, k), 'default')}")


def output_directory(cfg):
    """The configured directory, or '<model name>_binarized/' next to the model."""
    d = cfg["output"]["directory"]
    if not d:
        base = os.path.splitext(cfg["model"]["file"])[0]
        d = base + "_binarized"
    return d


# ---------------------------------------------------------------------------
# Loading (PyTorch is imported only here, so --show-config works without it)
# ---------------------------------------------------------------------------
def _is_torchscript(path):
    """A TorchScript archive is a zip with a constants.pkl / code/ entry; a torch.save file is not."""
    import zipfile
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as z:
        return any(n.endswith("constants.pkl") or "/code/" in n for n in z.namelist())


def load_model(cfg):
    """Returns the model (nn.Module, eval mode). Three cases:
       TorchScript file | whole model saved with torch.save(model) | weights only + [model] definition."""
    import torch
    import torch.nn as nn
    path, definition = cfg["model"]["file"], cfg["model"]["definition"]
    if not os.path.exists(path):
        sys.exit(f"[X] Model file not found: {path}")
    if _is_torchscript(path):
        model = torch.jit.load(path, map_location="cpu")
        return model.eval(), "TorchScript"
    # weights_only=False is needed for a whole saved model: only load files you trust
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval(), "whole model (torch.save(model))"
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not definition:
            sys.exit(f"[X] {path} contains only the weights (a state_dict): set [model] definition = "
                     f"\"python_file.py:ClassName\" so the CDT can rebuild the architecture.")
        if ":" not in definition:
            sys.exit(f"[X] [model] definition must look like \"file.py:ClassName\", got {definition!r}")
        file_data, class_name = definition.rsplit(":", 1)
        if not os.path.exists(file_data):
            sys.exit(f"[X] [model] definition: file not found: {file_data}")
        spec = importlib.util.spec_from_file_location("cdt_user_model", file_data)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(os.path.abspath(file_data)))
        spec.loader.exec_module(mod)
        if not hasattr(mod, class_name):
            sys.exit(f"[X] [model] definition: {file_data} has no class {class_name}")
        try:
            model = getattr(mod, class_name)()
        except TypeError as e:
            sys.exit(f"[X] [model] definition: {class_name}() must be buildable with no arguments ({e})")
        missing, extra = model.load_state_dict(state, strict=False)
        if missing or extra:
            sys.exit(f"[X] The weights do not match {class_name}: missing {list(missing)[:5]}, unexpected {list(extra)[:5]}")
        return model.eval(), f"weights + {definition}"
    sys.exit(f"[X] {path}: unrecognized content ({type(obj).__name__})")


def load_data(cfg):
    """Returns (train, validation, test) datasets. Validation drives every selection (search, checkpoint);
    test is touched once, for the final report. Formats without a separate test set (npy, imagefolder
    without a test/ folder) return test=None and the final report falls back to validation."""
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, random_split
    d = cfg["data"]
    fmt, folder = d["format"], d["directory"]
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])

    from compiler_core.train.standard import InputBitQuantizer
    s_bits = 8 - d["input_bits"]
    quantize = InputBitQuantizer(d["input_bits"])          # picklable: DataLoader workers need it (Python 3.14)

    def normalize(x):
        x = quantize(x)
        if d["mean"]:
            shape = (1, -1) + (1,) * (x.dim() - 2)
            x = (x - torch.tensor(d["mean"]).view(shape)) / torch.tensor(d["std"]).view(shape)
        return x

    def split(everything):
        n_val = max(1, int(len(everything) * d["validation_split"]))
        return tuple(random_split(everything, [len(everything) - n_val, n_val], generator=gen))

    if fmt == "npy":
        rx, ry = os.path.join(folder, "x.npy"), os.path.join(folder, "y.npy")
        for r in (rx, ry):
            if not os.path.exists(r):
                sys.exit(f"[X] [data] format npy: {r} not found")
        x = np.load(rx)
        y = np.load(ry).astype(np.int64)
        if len(x) != len(y):
            sys.exit(f"[X] x.npy has {len(x)} samples and y.npy {len(y)}")
        x = torch.from_numpy(x)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.float()
        if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()           # NHWC -> NCHW
        x = normalize(x)
        y = torch.from_numpy(y)
        rv = os.path.join(folder, "val.npy")
        if os.path.exists(rv):
            val = torch.from_numpy(np.load(rv).astype(bool))
            return TensorDataset(x[~val], y[~val]), TensorDataset(x[val], y[val]), None
        return split(TensorDataset(x, y)) + (None,)

    import torchvision
    import torchvision.transforms as T
    steps = [T.ToTensor()] + ([quantize] if s_bits else []) + \
            ([T.Normalize(d["mean"], d["std"])] if d["mean"] else [])
    tf = T.Compose(steps)
    if fmt == "imagefolder":
        tr, va = os.path.join(folder, "train"), os.path.join(folder, "val")
        te = os.path.join(folder, "test")
        test = torchvision.datasets.ImageFolder(te, tf) if os.path.isdir(te) else None
        if os.path.isdir(tr) and os.path.isdir(va):
            return torchvision.datasets.ImageFolder(tr, tf), torchvision.datasets.ImageFolder(va, tf), test
        return split(torchvision.datasets.ImageFolder(tr if os.path.isdir(tr) else folder, tf)) + (test,)
    name = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "mnist": "MNIST",
              "fashionmnist": "FashionMNIST"}.get(fmt.lower().replace("-", "").replace("_", ""))
    if name is None:
        sys.exit(f"[X] [data] format {fmt!r} not supported (npy, imagefolder, cifar10, cifar100, mnist, fashionmnist)")
    class_name = getattr(torchvision.datasets, name)
    return split(class_name(folder, train=True, download=True, transform=tf)) + (
        class_name(folder, train=False, download=True, transform=tf),)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_inspect(model, model_source, cfg):
    import torch
    import torch.nn as nn
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    convs = sum(isinstance(m, nn.Conv2d) for _, m in layers)
    dense_layers = len(layers) - convs
    types = {p.dtype for p in model.parameters()}
    quantized = any("quantized" in type(m).__module__ for m in model.modules())
    print(f"\n[1] INSPECT  ({model_source})")
    print(f"    layers      : {len(layers)}  ({convs} Conv2d, {dense_layers} Linear)")
    print(f"    parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"    precision   : {', '.join(str(t).replace('torch.', '') for t in types) or 'unknown'}")
    if quantized or (types and not types <= {torch.float32, torch.float64}):
        print("    [!] the model is not FP32: part of the information is already lost, the estimate is less reliable")
    kind = "conv" if convs else "mlp"
    print(f"    kind        : {kind}  -> method '{cfg['model']['method']}'")
    if cfg["model"]["method"] in ("gated", "ternary") and convs:
        sys.exit(f"[X] method '{cfg['model']['method']}' only supports MLPs (linear layers), "
                 "and this model has Conv2d layers")
    if cfg["model"]["method"] == "boolean":
        print("    method      : boolean -- a fully binary network, trained as defined (lib_symbex layers/conv/symbex_bool)")
    if cfg["output"]["verbose"]:
        for n, m in layers:
            print(f"      {n:<32} {type(m).__name__:<8} {'x'.join(str(s) for s in m.weight.shape)}")
    return kind


def stage_estimate(model, train, cfg, out_dir):
    if cfg["model"]["method"] == "boolean":
        print("\n[2] ESTIMATE  skipped: method 'boolean' (the network is already binary; its memory is in the export)")
        return dict(rows=[])
    import torch
    from compiler_core.sensitivity import Sensitivity
    n = min(cfg["data"]["calibration_samples"], len(train))
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    idx = torch.randperm(len(train), generator=gen)[:n].tolist()
    calib = torch.stack([train[i][0] for i in idx])
    print(f"\n[2] ESTIMATE  ({n} calibration samples, shape {tuple(calib.shape)})")
    h, t = cfg["hardware"], cfg["target"]
    res = Sensitivity(model).estimate(calib, ram_kb=h["ram_kb"], flash_kb=h["flash_kb"], budget=t["budget"],
                                      tolerance=t["tolerance"], options=t["options"],
                                      detail=cfg["output"]["verbose"], word_bits=h["cpu_bits"])
    old_keys = {"filas", "saltadas", "ram_pico", "limite_flash", "limite_ram", "presupuesto", "tolerancia",
                "opciones", "palabra_bits", "cabe"}
    clean = {k: v for k, v in res.items() if k != "rows" and k not in old_keys}
    clean["layers"] = [dict(name=f["name"], type=f["type"], shape=f["shape"], choice=f["chosen"],
                             meets=f["meets"], options={str(o): v for o, v in f["options"].items()})
                        for f in res["rows"]]
    with open(os.path.join(out_dir, "estimate.json"), "w") as fh:
        json.dump(clean, fh, indent=1, default=str)
    print(f"    -> {os.path.join(out_dir, 'estimate.json')}")
    print("    (the CHOICE column is a cosine-based suggestion; the train stage decides the bits by accuracy)")
    return res


def _bits(op):
    return 1 if op == "bin" else int(op)


def _plan_memory(rows, plan, width=1.0, word=32, bin_bits=1):
    """(flash bytes, peak RAM bytes) of a plan at a width. Exact arithmetic from the layer shapes:
    hidden layers get `width` times more outputs (and the next layer `width` times more inputs); each
    output row is padded to a whole CPU word; per-channel parameters as in Sensitivity.layer_memory.
    bin_bits: weight bits of a "bin" layer (1 plain, 2 ternary: two bit planes); its activations stay 1 bit."""
    flash, rams = 0, []
    for i, f in enumerate(rows):
        bits = _bits(plan[f["name"]])
        weight_bits = bin_bits if plan[f["name"]] == "bin" else bits
        mi = 1.0 if i == 0 else width
        mo = 1.0 if i == len(rows) - 1 else width
        channels = int(round(f["channels"] * mo))
        row = -(-int(round(f["fan_in"] * mi)) * weight_bits // word) * word
        flash += (channels * row + 7) // 8 + channels * (4 if plan[f["name"]] == "bin" else 8)
        rams.append((int(round(f["n_inputs"] * mi)) * bits + 7) // 8)
    return flash, max(a + b for a, b in zip(rams, rams[1:] + [0]))


def _fits(flash, ram, estimate_res, cfg):
    """(fits?, note). RAM may also use PSRAM (slower) when [hardware] psram_kb > 0."""
    if flash > estimate_res["flash_limit"]:
        return False, "flash over budget"
    if ram <= estimate_res["ram_limit"]:
        return True, ""
    psram = cfg["target"]["budget"] * cfg["hardware"]["psram_kb"] * 1024
    if ram <= estimate_res["ram_limit"] + psram:
        return True, "needs PSRAM (slower)"
    return False, "RAM over budget"


def _plan_text(rows, plan, width=1.0):
    return f"x{width:g} " + " ".join("b" if plan[f["name"]] == "bin" else str(plan[f["name"]]) for f in rows)


def _train_plan(model, teacher, plan, first, train, trl, val_l, cfg, epochs, dev, width=1.0, path=None,
                   verbose=False):
    """Builds the model for `plan` at width `width` from the original weights and trains it.
    Returns the result dict of train/ (the trained model is in res["model"])."""
    import torch
    from compiler_core.methods import apply_plan, init_thresholds, widen
    from compiler_core.train.distillation import distill_student
    from compiler_core.train.standard import train_model
    tr = cfg["training"]
    torch.manual_seed(tr["seed"])
    n = min(cfg["data"]["calibration_samples"], len(train))
    idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(tr["seed"]))[:n].tolist()
    calib = torch.stack([train[i][0] for i in idx])
    base = copy.deepcopy(model).cpu()
    if width > 1:
        widen(base, width, calib[:2], seed=tr["seed"])
    method = cfg["model"]["method"]
    student = apply_plan(base, plan, first=first, bin_method=method if method == "ternary" else "plain")
    init_thresholds(student, calib)
    common =dict(epochs=epochs, lr=tr["learning_rate"], device=dev, verbose=verbose, checkpoint_path=path,
                   save=tr["checkpoint"],
                   average_last=min(5, epochs), bn_momentum=0.3)
    if tr["distill"]:
        res = distill_student(student, teacher, trl, val_l, temperature=tr["temperature"], alpha=tr["alpha"],
                                  **common)
    else:
        res = train_model(student, trl, val_l, **common)
    res["model"] = student
    return res


class DeviceLoader:
    """A whole dataset as two tensors on the device, batched there: no worker processes and no copies per batch.
    For datasets that fit in VRAM ([training] data_on_gpu = true). The dataset's transforms run once, at load
    time, so they must be deterministic (the CDT's are: normalization and the input_bits quantization)."""

    def __init__(self, dataset, batch_size, device, shuffle=False, generator=None):
        import torch
        from torch.utils.data import DataLoader
        xs, ys = [], []
        for x, y in DataLoader(dataset, batch_size=1024, num_workers=2):
            xs.append(x)
            ys.append(y)
        self.x, self.y = torch.cat(xs).to(device), torch.cat(ys).to(device)
        self.batch_size, self.shuffle, self.generator = batch_size, shuffle, generator
        self.nbytes = self.x.element_size() * self.x.nelement() + self.y.element_size() * self.y.nelement()

    def __len__(self):
        return -(-len(self.y) // self.batch_size)

    def __iter__(self):
        import torch
        n = len(self.y)
        order = torch.randperm(n, generator=self.generator).to(self.x.device) if self.shuffle else None
        for i in range(0, n, self.batch_size):
            sel = order[i:i + self.batch_size] if order is not None else slice(i, i + self.batch_size)
            yield self.x[sel], self.y[sel]


def _references(model, train, val_l, rows, cfg, dev):
    """INT8 reference: the original model with every layer (input included) quantized to 8 bits, weights
    per output channel, activation ranges calibrated on training samples, NO retraining (post-training,
    like TFLite). Flash: 1 byte per weight + scale and zero point per channel (2 x int32)."""
    import torch
    import torch.nn as nn
    from compiler_core.methods import apply_plan, calibrate_ranges
    from compiler_core.train.standard import evaluate
    layers = [m for _, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    flash_fp32 = sum(p.numel() * 4 for p in model.parameters())
    flash_int8 = sum(m.weight.numel() + m.weight.shape[0] * 8 for m in layers)
    int8 = apply_plan(copy.deepcopy(model).cpu(), {f["name"]: 8 for f in rows}, first=rows[0]["name"],
                      keep_first_input=False).to(dev)
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    idx = torch.randperm(len(train), generator=gen)[:min(1024, len(train))].tolist()
    batches = [torch.stack([train[i][0] for i in idx[j:j + 256]]).to(dev) for j in range(0, len(idx), 256)]
    calibrate_ranges(int8, batches)
    acc = evaluate(int8.eval(), val_l, dev)
    del int8                                           # free the GPU memory of the reference
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return dict(acc_int8=acc, flash_int8=flash_int8, flash_fp32=flash_fp32)


def _predictions(model, val_l, dev):
    import torch
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in val_l:
            out.append(model(x.to(dev)).argmax(1).cpu())
    model.train(was_training)
    return torch.cat(out)


def stage_train(model, train, val, estimate_res, cfg, out_dir, kind="mlp", test=None):
    if cfg["model"]["method"] == "boolean":
        from compiler_core.export import boolean
        return boolean.train(model, train, val, cfg, out_dir)
    import torch
    from torch.utils.data import DataLoader
    from compiler_core.train.standard import evaluate
    if isinstance(model, torch.jit.ScriptModule):
        sys.exit("[X] train: a TorchScript model cannot be rebuilt layer by layer; give the weights + definition")
    tr, t = cfg["training"], cfg["target"]
    rows = estimate_res["rows"]
    first = rows[0]["name"]
    word = cfg["hardware"]["cpu_bits"]
    bin_bits = BIN_WEIGHT_BITS.get(cfg["model"]["method"], 1)
    options = sorted(t["options"], key=_bits)
    widths = sorted(set(float(a) for a in t["widths"]))
    if len(widths) > 1:                              # widening only works on plain chains: check it once
        from compiler_core.methods import widen
        try:
            widen(copy.deepcopy(model).cpu().eval(), widths[-1], torch.stack([train[i][0] for i in range(2)]))
        except ValueError as e:
            print(f"    [!] this model cannot be widened ({e}); searched at width x1 only")
            widths = [1.0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(tr["seed"])
    if tr["data_on_gpu"] and dev.type == "cuda":
        print("    data on the GPU: loading the training and validation sets into VRAM once ...")
        trl = DeviceLoader(train, tr["batch_size"], dev, shuffle=True, generator=gen)
        val_l = DeviceLoader(val, max(tr["batch_size"], 512), dev)
        print(f"    data on the GPU: {(trl.nbytes + val_l.nbytes) / 2**20:.0f} MB of VRAM")
    else:
        # persistent workers: created once, not again at every epoch (each restart left the GPU idle for seconds)
        trl = DataLoader(train, batch_size=tr["batch_size"], shuffle=True, generator=gen, num_workers=2,
                         persistent_workers=True)
        val_l = DataLoader(val, batch_size=max(tr["batch_size"], 512), num_workers=2, persistent_workers=True)
    teacher = model.to(dev).eval()
    acc_original = evaluate(teacher, val_l, dev)
    with_int8 = cfg["output"]["int8_reference"]
    if with_int8:
        ref = _references(model, train, val_l, rows, cfg, dev)
    else:
        ref = dict(acc_int8=None, flash_int8=None, flash_fp32=sum(p.numel() * 4 for p in model.parameters()))
    ref["acc_fp32"] = acc_original
    print(f"\n[3] REFERENCES  (the binarized model is compared against these)")
    print(f"    {'model':<44}{'accuracy':>9}{'flash KB':>11}")
    print(f"    {'FP32 original':<44}{acc_original:>8.2f}%{ref['flash_fp32'] / 1024:>11.1f}")
    if with_int8:
        print(f"    {'INT8 post-training (per-channel, like TFLite)':<44}{ref['acc_int8']:>8.2f}%"
              f"{ref['flash_int8'] / 1024:>11.1f}")
    else:
        print("    (INT8 reference off: [output] int8_reference = true to compare against INT8 too)")

    # ---------------- search: 1 bit at width x1 first; then wider; then more bits on the most sensitive layer
    print(f"\n    SEARCH  (quick trials of {t['search_epochs']} epochs, accept a drop <= {t['max_accuracy_drop']} points;"
          f" widths {widths}, options {options} | {dev})")
    print(f"    plan = width, then option per layer: {' '.join(f['name'] for f in rows)}   (b = 1 bit)")
    print(f"    {'step':>4} | {'plan':<26} | {'flash KB':>9} | {'accuracy':>8} | {'drop':>6} | decision")
    print("    " + "-" * 84)
    plan, width = {f["name"]: options[0] for f in rows}, widths[0]
    flash, ram = _plan_memory(rows, plan, width, word, bin_bits)
    ok, note = _fits(flash, ram, estimate_res, cfg)
    if not ok:
        sys.exit(f"[X] even the smallest plan does not fit the budget ({note}: flash {flash / 1024:.1f} KB, "
                 f"RAM {ram / 1024:.1f} KB)")
    history, step = [], 0
    while True:
        res = _train_plan(model, teacher, plan, first, train, trl, val_l, cfg, t["search_epochs"], dev, width)
        acc = res["final_average"]
        drop = acc_original - acc
        flash, ram = _plan_memory(rows, plan, width, word, bin_bits)
        row = dict(step=step, width=width, plan={k: str(v) for k, v in plan.items()}, flash=flash, ram=ram,
                    accuracy=acc, drop=drop)
        history.append(row)
        next_step = None
        if drop <= t["max_accuracy_drop"]:
            decision = "OK: within the accepted drop"
        else:
            # 1) wider, all-1-bit layers first
            wider = [a for a in widths if a > width]
            if all(v == options[0] for v in plan.values()):
                for a in wider:
                    fl, rm = _plan_memory(rows, plan, a, word, bin_bits)
                    if _fits(fl, rm, estimate_res, cfg)[0]:
                        next_step = (f"wider -> x{a:g}", plan, a)
                        break
            # 2) then more bits on the most sensitive layer (lowest cosine at its current option)
            if next_step is None:
                candidates = sorted((f for f in rows if options.index(plan[f["name"]]) + 1 < len(options)),
                                    key=lambda f: f["options"][plan[f["name"]]]["fid"])
                for f in candidates:
                    trial = dict(plan)
                    trial[f["name"]] = options[options.index(plan[f["name"]]) + 1]
                    fl, rm = _plan_memory(rows, trial, width, word, bin_bits)
                    if _fits(fl, rm, estimate_res, cfg)[0]:
                        next_step = (f"raise {f['name']} -> {trial[f['name']]}", trial, width)
                        break
            decision = next_step[0] if next_step else "STOP: nothing else fits the budget or can go up"
        row["decision"] = decision
        print(f"    {step:>4} | {_plan_text(rows, plan, width):<26} | {flash / 1024:>9.2f} | {acc:>7.2f}% | "
              f"{drop:>+6.2f} | {decision}")
        if next_step is None:
            break
        _, plan, width = next_step
        step += 1
    if not decision.startswith("OK"):
        print(f"    [!] the accepted drop ({t['max_accuracy_drop']} points) was not reached; the last plan is used")

    # ---------------- final training of the chosen width and plan
    flash, ram = _plan_memory(rows, plan, width, word, bin_bits)
    note = _fits(flash, ram, estimate_res, cfg)[1]
    print(f"\n    FINAL TRAINING  ({tr['epochs']} epochs, {'distillation' if tr['distill'] else 'labels only'},"
          f" width x{width:g})")
    print(f"    {'layer':<32}{'option':>8}")
    for f in rows:
        op = plan[f["name"]]
        print(f"    {f['name']:<32}{'bin' if op == 'bin' else str(op) + ' bits':>8}"
              + ("   (input kept)" if f["name"] == first else ""))
    path = os.path.join(out_dir, "model_binarized.pth")
    res = _train_plan(model, teacher, plan, first, train, trl, val_l, cfg, tr["epochs"], dev, width, path=path,
                         verbose=cfg["output"]["verbose"])
    last = res["history"][-1][3]
    pred_b, pred_o = _predictions(res["model"], val_l, dev), _predictions(teacher, val_l, dev)
    labels = torch.cat([y.cpu() for _, y in val_l])
    identity = 100 * float((pred_b == pred_o).float().mean())
    orig_correct = pred_o == labels
    retained = 100 * float((pred_b[orig_correct] == labels[orig_correct]).float().mean()) if orig_correct.any() else 0.0
    gained = int(((pred_b == labels) & ~orig_correct).sum())
    import numpy as np
    np.savez(os.path.join(out_dir, "predictions.npz"), labels=labels.numpy(), original=pred_o.numpy(),
             binarized=pred_b.numpy())                      # per-sample predictions, for per-class analysis
    print(f"\n    binarized validation accuracy: last epoch {last:.2f}% | mean of last 5 {res['final_average']:.2f}%"
          f" | (best epoch {res['best_test']:.2f}%, optimistic)")
    base_name, base_flash = ("INT8", ref["flash_int8"]) if with_int8 else ("FP32", ref["flash_fp32"])
    print(f"\n    {'model':<34}{'accuracy':>9}{'vs FP32':>9}" + (f"{'vs INT8':>9}" if with_int8 else "")
          + f"{'flash KB':>10}{'flash vs ' + base_name:>15}")
    models = [("FP32 original", acc_original, ref["flash_fp32"])]
    if with_int8:
        models.append(("INT8 post-training", ref["acc_int8"], ref["flash_int8"]))
    models.append(("SYMBEX (" + _plan_text(rows, plan, width) + ")", last, flash))
    for name, acc_m, fl in models:
        print(f"    {name:<34}{acc_m:>8.2f}%{acc_m - acc_original:>+9.2f}"
              + (f"{acc_m - ref['acc_int8']:>+9.2f}" if with_int8 else "")
              + f"{fl / 1024:>10.1f}{base_flash / fl:>14.1f}x")
    print(f"    (flash vs {base_name} = how many times less flash than the {base_name} model; "
          f"RAM peak SYMBEX {ram / 1024:.2f} KB" + (f", {note}" if note else "") + ")")
    print(f"    IDENTITY : the binarized model predicts the same class as the original on {identity:.2f}% of the "
          f"validation samples (how much of the original's behavior it keeps)")
    print(f"    RETAINED : of the {int(orig_correct.sum())} samples the original gets right, the binarized model still gets "
          f"{retained:.2f}% right (knowledge kept); it also gets right {gained} that the original got wrong")
    held_out = {}
    if test is not None:                              # held-out test: evaluated once, after every selection
        test_l = DataLoader(test, batch_size=max(tr["batch_size"], 512), num_workers=2)
        held_out = dict(acc_fp32=evaluate(teacher, test_l, dev), acc_symbex=evaluate(res["model"], test_l, dev),
                        acc_int8=_references(model, train, test_l, rows, cfg, dev)["acc_int8"] if with_int8 else None,
                        n=len(test))
        int8_text = f" | INT8 {held_out['acc_int8']:.2f}%" if with_int8 else ""
        print(f"\n    HELD-OUT TEST ({held_out['n']} samples, not used for any selection): "
              f"FP32 {held_out['acc_fp32']:.2f}%{int8_text} | SYMBEX {held_out['acc_symbex']:.2f}%")
    else:
        print("\n    [!] no separate test set for this data format: the accuracies above are on the validation"
              " split that also drove the search and checkpoint selection (optimistic)")
    with open(os.path.join(out_dir, "train.json"), "w") as fh:
        json.dump(dict(test=held_out, width=width, plan={k: str(v) for k, v in plan.items()}, flash_total=flash, ram_peak=ram,
                       references=ref, acc_original=acc_original, acc_last=last,
                       acc_mean_last5=res["final_average"], acc_best_epoch=res["best_test"], identity=identity, retained=retained, gained=gained,
                       search=history, history=res["history"]), fh, indent=1)
    print(f"    -> {path}\n    -> {os.path.join(out_dir, 'train.json')}")
    res["plan"] = plan
    res["width"] = width
    return res


def _dense_shapes(model):
    """[(n_in, n_out, first)] of the binarized dense layers, in order (for the per-model arithmetic choice)."""
    import torch.nn as nn
    from compiler_core.methods.dense.plain import DensePlain
    shapes = [(m.weight.shape[1], m.weight.shape[0]) for m in model.modules() if isinstance(m, (DensePlain, nn.Linear))]
    return [(i, o, k == 0) for k, (i, o) in enumerate(shapes)]


def choose_arithmetic(cfg, out_dir, shapes=None):
    """[output] arithmetic: "and" / "xnor" as given; "auto" = the fastest one on the board for THIS model, predicted
    from the chip profile (tools/chip_profiles/<board>_<popcount>.json, measured now with the kernel benchmark if
    there is none yet, or if it is damaged) for the real shape of every layer and the [data] input_bits.
    Without a board: "and" (the same results; it is the faster one on the chips measured so far)."""
    arithmetic = cfg["output"]["arithmetic"]
    if arithmetic != "auto":
        return arithmetic
    from compiler_core.chip.kernels import load_profile, run_kernels, save_profile
    ch = cfg["chip"]
    if not ch["board"]:
        print("    arithmetic  : auto -> AND (no [chip] board to measure; both give the same results)")
        return "and"
    if ch["board"] == "uno":
        print("    arithmetic  : auto -> XNOR (AVR: no SIMD; its self-contained kernels use XNOR and the smaller rows)")
        return "xnor"
    profile = load_profile(ch["board"], ch["popcount"])
    if profile is None:
        print(f"    arithmetic  : auto -> no profile for {ch['board']} (popcount {ch['popcount']}): measuring the kernels")
        try:
            profile = run_kernels(ch["board"], ch["port"], cfg["output"]["lib_path"],
                                  os.path.join(out_dir, f"kernels_{ch['board']}"), ch["popcount"], upload=ch["upload"])
        except (RuntimeError, ValueError) as e:
            print(f"    [!] kernel benchmark failed ({e}): using AND")
            return "and"
        if profile.get("checksums_ok"):
            print(f"    -> chip profile saved: {save_profile(profile)}")
    if not profile.get("checksums_ok"):
        print("    [!] the chip profile is not trusted (checksums differ): using AND")
        return "and"
    from compiler_core.chip.kernels import model_cost
    if shapes:
        cost = {a: model_cost(profile, shapes, a, input_bits=cfg["data"]["input_bits"]) for a in ("and", "xnor")}
        best = min(cost, key=cost.get)
        print(f"    arithmetic  : auto -> {best.upper()} on {ch['board']}, for this model "
              f"(expected: AND {cost['and'] / 1000:.0f}k cycles, XNOR {cost['xnor'] / 1000:.0f}k cycles, from the chip profile)")
        return best
    print(f"    arithmetic  : auto -> {profile['best_arithmetic'].upper()} (fastest on {ch['board']}, from its chip profile)")
    return profile["best_arithmetic"]


def stage_export(trained, val, cfg, out_dir, kind, original_model=None, train=None):
    if cfg["model"]["method"] == "boolean":
        from compiler_core.export import boolean
        print("\n[4-5] EXPORT + VALIDATE  (method 'boolean')")
        try:
            res = boolean.export_and_validate(trained["model"], val, out_dir, cfg["output"]["lib_path"],
                                              cfg["output"]["validation_samples"])
        except ValueError as e:
            print(f"    [X] {e}")
            return None
        with open(os.path.join(out_dir, "export.json"), "w") as fh:
            json.dump({k: v for k, v in res.items() if not k.startswith("_")}, fh, indent=1)
        print(f"    -> {os.path.join(out_dir, 'export.json')}")
        return res
    import torch
    from compiler_core.export.dense import export_and_validate
    from compiler_core.export.int8_reference import export_int8
    print("\n[4-5] EXPORT + VALIDATE")
    plan = trained["plan"]
    if kind != "mlp" or cfg["model"]["method"] not in ("plain", "ternary"):
        print("    [!] export is only integrated for MLPs with the plain or ternary method; skipped")
        return None
    ternary = cfg["model"]["method"] == "ternary"
    if ternary:
        from compiler_core.export.ternary import export_and_validate
    if any(op != "bin" for op in plan.values()):
        print("    [!] the plan has k-bit layers (" + ", ".join(f"{k}: {v}" for k, v in plan.items() if v != "bin")
              + "): the C dense layer with k > 1 is not verified yet, so only all-1-bit plans are exported."
              "\n        Run with --options bin to get an exportable model.")
        return None
    try:
        res = export_and_validate(trained["model"], val, out_dir, cfg["output"]["lib_path"], cfg["data"]["mean"],
                                 cfg["data"]["std"], cfg["output"]["validation_samples"],
                                 arithmetic="and" if ternary
                                 else choose_arithmetic(cfg, out_dir, _dense_shapes(trained["model"])),
                                 input_bits=cfg["data"]["input_bits"])
    except ValueError as e:
        print(f"    [X] {e}")
        return None
    if cfg["output"]["int8_reference"] and original_model is not None and train is not None:
        try:
            gen = torch.Generator().manual_seed(cfg["training"]["seed"])
            idx = torch.randperm(len(train), generator=gen)[:min(1024, len(train))].tolist()
            calib = torch.stack([train[i][0] for i in idx])
            res["int8"] = export_int8(original_model, calib, res["_xs"], res["_x_u8"], res["_y"], out_dir,
                                        cfg["output"]["lib_path"], cfg["data"]["mean"], cfg["data"]["std"])
        except ValueError as e:
            print(f"    [!] INT8 reference not exported: {e}")
    with open(os.path.join(out_dir, "export.json"), "w") as fh:
        json.dump({k: v for k, v in res.items() if not k.startswith("_")}, fh, indent=1)
    print(f"    -> {os.path.join(out_dir, 'export.json')}")
    return res


def _int8_of(res_export, out_dir):
    """The INT8 reference exported with the model, if any, in the form measure_on_chip expects."""
    import numpy as np
    r8 = res_export.get("int8")
    path = os.path.join(out_dir, "validation_int8.u8")
    if not r8 or not os.path.exists(path):
        return None
    return dict(header=r8["header"], expected=np.fromfile(path, dtype=np.uint8).astype(np.int64), flash=r8["flash_bytes"])


def _measure(res_export, header, cfg, out_dir, tag, int8=None):
    from compiler_core.chip.measure import measure_on_chip
    ch = cfg["chip"]
    return measure_on_chip(header, res_export["_x_u8"], res_export["_y"], res_export["_pred_sim"],
                           os.path.join(out_dir, f"chip_{ch['board']}{tag}"), cfg["output"]["lib_path"], ch["board"],
                           ch["port"], ch["baud"], ch["samples"], ch["upload"], ch["popcount"],
                           res_export["flash_bytes"], weights=ch["weights"], int8=int8)


def stage_chip(res_export, cfg, out_dir):
    ch = cfg["chip"]
    print(f"\n[6] CHIP  ({ch['board']} on {ch['port']}, {ch['samples']} validation images, weights in {ch['weights']})")
    try:
        res = _measure(res_export, res_export["header"], cfg, out_dir, "", _int8_of(res_export, out_dir))
    except (RuntimeError, ValueError) as e:
        print(f"    [X] {e}")
        return None
    res["arithmetic"] = res_export.get("arithmetic")
    with open(os.path.join(out_dir, "chip.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    -> {os.path.join(out_dir, 'chip.json')}")
    if ch["compare_arithmetic"] and cfg["model"]["method"] == "boolean":
        print("    (--compare-arithmetic: not for method 'boolean'; its layers use the HAL's AND count on every chip)")
    elif ch["compare_arithmetic"]:
        compare_arithmetic(res_export, res, cfg, out_dir)
    return res


def compare_arithmetic(res_export, res_main, cfg, out_dir):
    """The WHOLE model with the other arithmetic too, on the same board and the same images: the final word on
    which one is faster for this model (the kernel benchmark only predicts it). Both give the same classes."""
    from compiler_core.export.dense import flash_bytes, validate_c, write_header
    main = res_export.get("arithmetic") or "and"
    other = "xnor" if main == "and" else "and"
    layers = res_export.get("_layers")
    if layers is None:
        print("    [!] compare_arithmetic needs the export of this run (run from the export stage, not from chip)")
        return None
    header = os.path.join(out_dir, f"model_symbex_{other}.h")
    write_header(layers, header, arithmetic=other, input_bits=cfg["data"]["input_bits"])
    eq, tot, msg = validate_c(header, cfg["output"]["lib_path"], res_export["_x_u8"], res_export["_pred_sim"])
    if eq is not None and eq != tot:
        print(f"    [X] the {other.upper()} header is not exact on the PC ({eq}/{tot}): not measured")
        return None
    print(f"\n    COMPARISON: the same model with {other.upper()} arithmetic (exact on the PC: {eq}/{tot})")
    try:
        res_other = _measure(dict(res_export, flash_bytes=flash_bytes(layers, other)), header, cfg, out_dir, f"_{other}")
    except (RuntimeError, ValueError) as e:
        print(f"    [X] {e}")
        return None
    res_other["arithmetic"] = other
    with open(os.path.join(out_dir, f"chip_{other}.json"), "w") as fh:
        json.dump(res_other, fh, indent=1)
    rows = sorted([(main, res_main), (other, res_other)], key=lambda r: r[1]["median_ms"])
    fastest = rows[0][1]["median_ms"]
    print(f"\n    WHOLE MODEL, BOTH ARITHMETICS  ({cfg['chip']['board']}, weights in {cfg['chip']['weights']}, "
          f"{res_main['n']} images)")
    print(f"    {'arithmetic':<12}{'latency ms':>12}{'vs fastest':>12}{'equal to sim.':>15}   per layer (ms)")
    print("    " + "-" * 86)
    for name, r in rows:
        print(f"    {name.upper():<12}{r['median_ms']:>12.3f}{r['median_ms'] / fastest:>11.2f}x{r['equal_simulation']:>14.2f}%   "
              + "  ".join(f"{c['median_ms']:.3f}" for c in r["layers"]))
    print(f"    -> on this board, for this model: {rows[0][0].upper()}"
          + ("  (the export used it)" if rows[0][0] == main else f"  (the export used {main.upper()}: "
             f"set [output] arithmetic = \"{rows[0][0]}\" to export with it)"))
    return res_other


def _quiet_estimate(model, train, cfg):
    if cfg["model"]["method"] == "boolean":
        return dict(rows=[])
    """The estimate rows (layer shapes and the first layer) without printing: needed to train or to rebuild the
    trained model when the run starts after the estimate stage."""
    import torch
    from compiler_core.sensitivity import Sensitivity
    n = min(cfg["data"]["calibration_samples"], len(train))
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    idx = torch.randperm(len(train), generator=gen)[:n].tolist()
    calib = torch.stack([train[i][0] for i in idx])
    h, tg = cfg["hardware"], cfg["target"]
    return Sensitivity(model).estimate(calib, ram_kb=h["ram_kb"], flash_kb=h["flash_kb"], budget=tg["budget"],
                                       tolerance=tg["tolerance"], options=tg["options"], verbose=False,
                                       detail=False, word_bits=h["cpu_bits"])


def _option(v):
    """'bin' stays 'bin'; '2' (as saved in JSON) -> 2."""
    return v if v == "bin" else int(v)


def load_trained(model, train, estimate_res, cfg, out_dir):
    if cfg["model"]["method"] == "boolean":
        from compiler_core.export import boolean
        return boolean.load_trained(model, out_dir)
    """Rebuilds the trained binarized model from out_dir (train.json + model_binarized.pth), without training:
    the original model, widened as the search chose, binarized with the chosen plan, with the saved weights."""
    import torch
    from compiler_core.methods import apply_plan, widen
    tj, wp = os.path.join(out_dir, "train.json"), os.path.join(out_dir, "model_binarized.pth")
    for p in (tj, wp):
        if not os.path.exists(p):
            sys.exit(f"[X] {p} not found: there is no trained model to resume from (run from the train stage)")
    info = json.load(open(tj))
    plan = {k: _option(v) for k, v in info["plan"].items()}
    width = float(info.get("width", 1.0))
    base = copy.deepcopy(model).cpu()
    if width > 1:
        widen(base, width, torch.stack([train[i][0] for i in range(2)]), seed=cfg["training"]["seed"])
    method = cfg["model"]["method"]
    student = apply_plan(base, plan, first=estimate_res["rows"][0]["name"],
                         bin_method=method if method == "ternary" else "plain")
    student.load_state_dict(torch.load(wp, map_location="cpu"))
    student.eval()
    print(f"\n[3] TRAIN  skipped: resumed the trained model from {out_dir} (width x{width:g}, plan "
          + " ".join("b" if v == "bin" else str(v) for v in plan.values()) + ")")
    return dict(model=student, plan=plan, width=width)


def load_export(cfg, out_dir):
    """What the chip stage needs from an existing export in out_dir (model_symbex.h, validation_*.u8, export.json)."""
    import re
    from compiler_core.export.dense import load_validation
    header, ej = os.path.join(out_dir, "model_symbex.h"), os.path.join(out_dir, "export.json")
    for p in (header, ej, os.path.join(out_dir, "validation_x.u8")):
        if not os.path.exists(p):
            sys.exit(f"[X] {p} not found: there is no export to resume from (run from the export stage)")
    m = re.search(r"#define SYMBEX_MLP_N_INPUTS (\d+)", open(header).read())
    if not m:
        sys.exit(f"[X] {header} was exported before the English translation: run from the export stage")
    res = json.load(open(ej))
    x, y, sim = load_validation(out_dir, int(m.group(1)))
    res.update(header=header, _x_u8=x, _y=y, _pred_sim=sim)
    print(f"\n[4-5] EXPORT  skipped: resumed the export in {out_dir}")
    return res


def main():
    args = build_parser().parse_args()
    cfg, source = load_config(args)
    if args.stop_after is not None:                    # old flag, same as --to
        cfg["run"]["to"] = args.stop_after
        source[("run", "to")] = "flag"
    if args.show_config:
        print_config(cfg, source)
        return
    validate_config(cfg)
    out_dir = output_directory(cfg)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.toml"), "w") as fh:
        fh.write(f"# written by cdt.py on {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + to_toml(cfg))
    if cfg["output"]["verbose"]:
        print_config(cfg, source)
    start, end = _stage_index(cfg["run"]["from"]), _stage_index(cfg["run"]["to"])
    runs = lambda stage: start <= STAGES.index(stage) <= end
    print(f"\n[*] output directory: {out_dir}   (config_used.toml written)")
    print(f"[*] stages: {' -> '.join(s.upper() if runs(s) else s for s in STAGES)}   (UPPER = runs)")

    import torch
    torch.manual_seed(cfg["training"]["seed"])
    model, model_source = load_model(cfg)
    kind = stage_inspect(model, model_source, cfg)       # always: cheap, and every later stage needs `kind`
    if end == STAGES.index("inspect"):
        return
    train = val = test = None
    if start <= STAGES.index("export"):                  # estimate / train / export need the data
        train, val, test = load_data(cfg)
        print(f"\n[*] data: {len(train)} training / {len(val)} validation / "
              f"{len(test) if test is not None else 'no'} test samples ({cfg['data']['format']})")
    estimate_res = trained = res_export = None
    if runs("estimate"):
        estimate_res = stage_estimate(model, train, cfg, out_dir)
    elif start <= STAGES.index("export"):
        estimate_res = _quiet_estimate(model, train, cfg)
    if end == STAGES.index("estimate"):
        return
    if runs("train"):
        trained = stage_train(model, train, val, estimate_res, cfg, out_dir, kind, test=test)
    elif start == STAGES.index("export"):
        trained = load_trained(model, train, estimate_res, cfg, out_dir)
    if end == STAGES.index("train"):
        return
    if runs("export"):
        res_export = stage_export(trained, test if test is not None else val, cfg, out_dir, kind, model, train)
        if res_export is None:
            return
    else:
        res_export = load_export(cfg, out_dir)
    if end == STAGES.index("export"):
        return
    if cfg["chip"]["board"] and cfg["model"]["method"] == "ternary":
        print("\n[6] CHIP  skipped: the chip harness (chip/measure.py) does not build symbex_ternary.c yet")
    elif cfg["chip"]["board"]:
        stage_chip(res_export, cfg, out_dir)
    else:
        print("\n[6] CHIP  skipped: set [chip] board = \"esp32\" or \"s3\" (or --board esp32) to measure on a board")


if __name__ == "__main__":
    main()
