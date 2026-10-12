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
                  modelo_symbex.h for lib_symbex (symbex_mlp_inferir(const uint8_t* x))
    6. chip       ([chip] board = "esp32" / "s3") builds and flashes an ESP-IDF project with the header and
                  lib_symbex, streams validation images and reports: equal to the integer simulation (must be
                  100%), accuracy and latency measured on the chip. Needs PlatformIO + a connected board.
    5. validate   (runs together with export) FIDELITY: PyTorch vs. the integer simulation (what folding
                  costs); EXACTNESS: the compiled C on this PC vs. the integer simulation (must be 100%)
--stop-after chooses the last stage to run (default: the last one that is integrated).

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
             "mean": [], "std": []},
    "hardware": {"ram_kb": 320, "flash_kb": 4096, "psram_kb": 0, "cpu_bits": 32},
    "target": {"budget": 0.25, "tolerance": 0.99, "options": ["bin", 2, 4, 8], "max_accuracy_drop": 2.0,
               "search_epochs": 5, "widths": [1.0, 1.5, 2.0]},
    "training": {"epochs": 150, "distill": True, "temperature": 4.0, "alpha": 0.85, "learning_rate": 1e-3,
                 "batch_size": 256, "checkpoint": "last", "seed": 42},
    "output": {"directory": "", "verbose": True, "lib_path": "../Symbex1_PIO/lib/lib_symbex",
               "validation_samples": 2000},
    "chip": {"board": "", "port": "/dev/ttyUSB0", "baud": 921600, "samples": 1000, "upload": True,
             "popcount": "lut16"},
}
STAGES = ["inspect", "estimate", "train", "export", "validate", "chip"]
INTEGRATED = ["inspect", "estimate", "train", "export", "validate", "chip"]
METHODS = ["plain", "gated", "boolean"]
FORMATS_LOCAL = ["npy", "imagefolder"]


def _flag_names():
    """(section, key) -> flag name. A key that appears in several sections gets the section as prefix."""
    count = {}
    for sec, keys in DEFAULTS.items():
        for k in keys:
            count[k] = count.get(k, 0) + 1
    return {(sec, k): (k if count[k] == 1 else f"{sec}_{k}").replace("_", "-")
            for sec, keys in DEFAULTS.items() for k in keys}


def _parse_list(texto):
    """'bin,2,4,8' -> ['bin', 2, 4, 8]; '0.5,0.5,0.5' -> [0.5, 0.5, 0.5]."""
    out = []
    for p in (x.strip() for x in texto.split(",") if x.strip()):
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
    ap.add_argument("--stop-after", choices=STAGES, default=INTEGRATED[-1], help="last stage to run")
    for (sec, k), flag in _flag_names().items():
        v = DEFAULTS[sec][k]
        dest = f"{sec}__{k}"
        ayuda = f"[{sec}] {k}"
        if isinstance(v, bool):
            ap.add_argument(f"--{flag}", dest=dest, action=argparse.BooleanOptionalAction, default=None, help=ayuda)
        elif isinstance(v, list):
            ap.add_argument(f"--{flag}", dest=dest, type=_parse_list, default=None,
                            help=ayuda + " (comma separated)")
        else:
            ap.add_argument(f"--{flag}", dest=dest, type=type(v), default=None, help=ayuda)
    return ap


def load_config(args):
    """defaults < TOML file < flags. Returns (config, list of 'where each value came from' notes)."""
    cfg = copy.deepcopy(DEFAULTS)
    origen = {}
    if args.config:
        if tomllib is None:
            sys.exit("[X] Reading TOML needs Python 3.11 or newer (tomllib).")
        if not os.path.exists(args.config):
            sys.exit(f"[X] Configuration file not found: {args.config}")
        with open(args.config, "rb") as f:
            archivo = tomllib.load(f)
        for sec, keys in archivo.items():
            if sec not in cfg:
                sys.exit(f"[X] {args.config}: unknown section [{sec}] (valid: {', '.join(cfg)})")
            for k, v in keys.items():
                if k not in cfg[sec]:
                    sys.exit(f"[X] {args.config}: unknown field '{k}' in [{sec}] (valid: {', '.join(cfg[sec])})")
                cfg[sec][k] = v
                origen[(sec, k)] = "file"
    for sec, keys in DEFAULTS.items():
        for k in keys:
            v = getattr(args, f"{sec}__{k}")
            if v is not None:
                cfg[sec][k] = v
                origen[(sec, k)] = "flag"
    return cfg, origen


def validate_config(cfg):
    """Checks types and ranges before anything heavy runs. Exits with a clear message on the first problem."""
    errores = []
    m, d, h, t, tr = cfg["model"], cfg["data"], cfg["hardware"], cfg["target"], cfg["training"]
    if not m["file"]:
        errores.append("[model] file is empty: the path of the original model is required")
    if m["method"] not in METHODS:
        errores.append(f"[model] method must be one of {METHODS}, got {m['method']!r}")
    if not d["directory"]:
        errores.append("[data] directory is empty")
    if not (0 < d["validation_split"] < 1):
        errores.append("[data] validation_split must be between 0 and 1")
    if d["calibration_samples"] < 1:
        errores.append("[data] calibration_samples must be >= 1")
    if len(d["mean"]) != len(d["std"]):
        errores.append("[data] mean and std must have the same length (or both be empty)")
    for k in ("ram_kb", "flash_kb"):
        if h[k] <= 0:
            errores.append(f"[hardware] {k} must be > 0")
    if h["psram_kb"] < 0:
        errores.append("[hardware] psram_kb must be >= 0")
    if h["cpu_bits"] not in (8, 16, 32, 64):
        errores.append("[hardware] cpu_bits must be 8, 16, 32 or 64")
    if not (0 < t["budget"] <= 1):
        errores.append("[target] budget must be in (0, 1]")
    if not (0 < t["tolerance"] <= 1):
        errores.append("[target] tolerance must be in (0, 1]")
    for op in t["options"]:
        if op != "bin" and not (isinstance(op, int) and 1 <= op <= 16):
            errores.append(f"[target] options: unknown option {op!r} (use 'bin' or 1-16 bits)")
    if t["max_accuracy_drop"] < 0:
        errores.append("[target] max_accuracy_drop must be >= 0 (accuracy points)")
    if t["search_epochs"] < 1:
        errores.append("[target] search_epochs must be >= 1")
    if not t["widths"] or any((not isinstance(a, (int, float))) or a < 1 for a in t["widths"]):
        errores.append("[target] widths must be a list of numbers >= 1 (e.g. [1.0, 1.5, 2.0])")
    if cfg["chip"]["popcount"] not in ("lut16", "swar"):
        errores.append("[chip] popcount must be \"lut16\" (64 KB table, fastest) or \"swar\" (no table, minimum RAM)")
    if cfg["chip"]["board"] not in ("", "esp32", "s3"):
        errores.append("[chip] board must be \"\" (no chip), \"esp32\" or \"s3\"")
    if tr["checkpoint"] not in ("last", "best"):
        errores.append("[training] checkpoint must be 'last' or 'best'")
    if tr["epochs"] < 1 or tr["batch_size"] < 1:
        errores.append("[training] epochs and batch_size must be >= 1")
    if not (0 <= tr["alpha"] <= 1):
        errores.append("[training] alpha must be in [0, 1]")
    if errores:
        sys.exit("[X] Configuration errors:\n    " + "\n    ".join(errores))


def _toml_valor(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_valor(x) for x in v) + "]"
    return repr(v)


def to_toml(cfg):
    """Writes the configuration back as TOML (our structure only: sections of plain values and lists)."""
    lineas = []
    for sec, keys in cfg.items():
        lineas.append(f"[{sec}]")
        lineas += [f"{k} = {_toml_valor(v)}" for k, v in keys.items()]
        lineas.append("")
    return "\n".join(lineas)


def print_config(cfg, origen):
    print("\nCONFIGURATION  (value  <- where it came from: default / file / flag)")
    for sec, keys in cfg.items():
        print(f"  [{sec}]")
        for k, v in keys.items():
            print(f"    {k:<20} = {_toml_valor(v):<28} <- {origen.get((sec, k), 'default')}")


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
def _es_torchscript(ruta):
    """A TorchScript archive is a zip with a constants.pkl / code/ entry; a torch.save file is not."""
    import zipfile
    if not zipfile.is_zipfile(ruta):
        return False
    with zipfile.ZipFile(ruta) as z:
        return any(n.endswith("constants.pkl") or "/code/" in n for n in z.namelist())


def load_model(cfg):
    """Returns the model (nn.Module, eval mode). Three cases:
       TorchScript file | whole model saved with torch.save(model) | weights only + [model] definition."""
    import torch
    import torch.nn as nn
    ruta, definicion = cfg["model"]["file"], cfg["model"]["definition"]
    if not os.path.exists(ruta):
        sys.exit(f"[X] Model file not found: {ruta}")
    if _es_torchscript(ruta):
        modelo = torch.jit.load(ruta, map_location="cpu")
        return modelo.eval(), "TorchScript"
    # weights_only=False is needed for a whole saved model: only load files you trust
    obj = torch.load(ruta, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval(), "whole model (torch.save(model))"
    if isinstance(obj, dict):
        estado = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not definicion:
            sys.exit(f"[X] {ruta} contains only the weights (a state_dict): set [model] definition = "
                     f"\"python_file.py:ClassName\" so the CDT can rebuild the architecture.")
        if ":" not in definicion:
            sys.exit(f"[X] [model] definition must look like \"file.py:ClassName\", got {definicion!r}")
        archivo, clase = definicion.rsplit(":", 1)
        if not os.path.exists(archivo):
            sys.exit(f"[X] [model] definition: file not found: {archivo}")
        spec = importlib.util.spec_from_file_location("cdt_user_model", archivo)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(os.path.abspath(archivo)))
        spec.loader.exec_module(mod)
        if not hasattr(mod, clase):
            sys.exit(f"[X] [model] definition: {archivo} has no class {clase}")
        try:
            modelo = getattr(mod, clase)()
        except TypeError as e:
            sys.exit(f"[X] [model] definition: {clase}() must be buildable with no arguments ({e})")
        faltan, sobran = modelo.load_state_dict(estado, strict=False)
        if faltan or sobran:
            sys.exit(f"[X] The weights do not match {clase}: missing {list(faltan)[:5]}, unexpected {list(sobran)[:5]}")
        return modelo.eval(), f"weights + {definicion}"
    sys.exit(f"[X] {ruta}: unrecognized content ({type(obj).__name__})")


def load_data(cfg):
    """Returns (train dataset, validation dataset) as TensorDatasets or torchvision datasets."""
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, random_split
    d = cfg["data"]
    fmt, carpeta = d["format"], d["directory"]
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])

    def normalizar(x):
        if d["mean"]:
            forma = (1, -1) + (1,) * (x.dim() - 2)
            x = (x - torch.tensor(d["mean"]).view(forma)) / torch.tensor(d["std"]).view(forma)
        return x

    if fmt == "npy":
        rx, ry = os.path.join(carpeta, "x.npy"), os.path.join(carpeta, "y.npy")
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
        x = normalizar(x)
        y = torch.from_numpy(y)
        rv = os.path.join(carpeta, "val.npy")
        if os.path.exists(rv):
            val = torch.from_numpy(np.load(rv).astype(bool))
            return TensorDataset(x[~val], y[~val]), TensorDataset(x[val], y[val])
        todo = TensorDataset(x, y)
        n_val = max(1, int(len(todo) * d["validation_split"]))
        return tuple(random_split(todo, [len(todo) - n_val, n_val], generator=gen))

    import torchvision
    import torchvision.transforms as T
    pasos = [T.ToTensor()] + ([T.Normalize(d["mean"], d["std"])] if d["mean"] else [])
    tf = T.Compose(pasos)
    if fmt == "imagefolder":
        tr, va = os.path.join(carpeta, "train"), os.path.join(carpeta, "val")
        if os.path.isdir(tr) and os.path.isdir(va):
            return torchvision.datasets.ImageFolder(tr, tf), torchvision.datasets.ImageFolder(va, tf)
        todo = torchvision.datasets.ImageFolder(carpeta, tf)
        n_val = max(1, int(len(todo) * d["validation_split"]))
        return tuple(random_split(todo, [len(todo) - n_val, n_val], generator=gen))
    nombre = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "mnist": "MNIST",
              "fashionmnist": "FashionMNIST"}.get(fmt.lower().replace("-", "").replace("_", ""))
    if nombre is None:
        sys.exit(f"[X] [data] format {fmt!r} not supported (npy, imagefolder, cifar10, cifar100, mnist, fashionmnist)")
    clase = getattr(torchvision.datasets, nombre)
    return clase(carpeta, train=True, download=True, transform=tf), clase(carpeta, train=False, download=True, transform=tf)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_inspect(modelo, origen_modelo, cfg):
    import torch
    import torch.nn as nn
    capas = [(n, m) for n, m in modelo.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    convs = sum(isinstance(m, nn.Conv2d) for _, m in capas)
    lineales = len(capas) - convs
    tipos = {p.dtype for p in modelo.parameters()}
    cuantizado = any("quantized" in type(m).__module__ for m in modelo.modules())
    print(f"\n[1] INSPECT  ({origen_modelo})")
    print(f"    layers      : {len(capas)}  ({convs} Conv2d, {lineales} Linear)")
    print(f"    parameters  : {sum(p.numel() for p in modelo.parameters()):,}")
    print(f"    precision   : {', '.join(str(t).replace('torch.', '') for t in tipos) or 'unknown'}")
    if cuantizado or (tipos and not tipos <= {torch.float32, torch.float64}):
        print("    [!] the model is not FP32: part of the information is already lost, the estimate is less reliable")
    kind = "conv" if convs else "mlp"
    print(f"    kind        : {kind}  -> method '{cfg['model']['method']}'")
    if cfg["model"]["method"] == "gated" and convs:
        sys.exit("[X] method 'gated' only supports MLPs (linear layers), and this model has Conv2d layers")
    if cfg["model"]["method"] == "boolean":
        sys.exit("[X] method 'boolean' (SYMBEX v3) is not integrated into lib_symbex yet: use 'plain'")
    if cfg["output"]["verbose"]:
        for n, m in capas:
            print(f"      {n:<32} {type(m).__name__:<8} {'x'.join(str(s) for s in m.weight.shape)}")
    return kind


def stage_estimate(modelo, train, cfg, salida):
    import torch
    from compiler_core.sensitivity import Sensitivity
    n = min(cfg["data"]["calibration_samples"], len(train))
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    idx = torch.randperm(len(train), generator=gen)[:n].tolist()
    calib = torch.stack([train[i][0] for i in idx])
    print(f"\n[2] ESTIMATE  ({n} calibration samples, shape {tuple(calib.shape)})")
    h, t = cfg["hardware"], cfg["target"]
    res = Sensitivity(modelo).estimar(calib, ram_kb=h["ram_kb"], flash_kb=h["flash_kb"], presupuesto=t["budget"],
                                      tolerancia=t["tolerance"], opciones=t["options"],
                                      detalle=cfg["output"]["verbose"], palabra_bits=h["cpu_bits"])
    limpio = {k: v for k, v in res.items() if k != "filas"}
    limpio["layers"] = [dict(name=f["name"], type=f["type"], shape=f["shape"], choice=f["elegida"],
                             meets=f["cumple"], options={str(o): v for o, v in f["opciones"].items()})
                        for f in res["filas"]]
    with open(os.path.join(salida, "estimate.json"), "w") as fh:
        json.dump(limpio, fh, indent=1, default=str)
    print(f"    -> {os.path.join(salida, 'estimate.json')}")
    print("    (the CHOICE column is a cosine-based suggestion; the train stage decides the bits by accuracy)")
    return res


def _bits(op):
    return 1 if op == "bin" else int(op)


def _memoria_plan(filas, plan, ancho=1.0, palabra=32):
    """(flash bytes, peak RAM bytes) of a plan at a width. Exact arithmetic from the layer shapes:
    hidden layers get `ancho` times more outputs (and the next layer `ancho` times more inputs); each
    output row is padded to a whole CPU word; per-channel parameters as in Sensitivity.memoria_capa."""
    flash, rams = 0, []
    for i, f in enumerate(filas):
        bits = _bits(plan[f["name"]])
        mi = 1.0 if i == 0 else ancho
        mo = 1.0 if i == len(filas) - 1 else ancho
        canales = int(round(f["canales"] * mo))
        fila = -(-int(round(f["fan_in"] * mi)) * bits // palabra) * palabra
        flash += (canales * fila + 7) // 8 + canales * (4 if plan[f["name"]] == "bin" else 8)
        rams.append((int(round(f["n_entradas"] * mi)) * bits + 7) // 8)
    return flash, max(a + b for a, b in zip(rams, rams[1:] + [0]))


def _cabe(flash, ram, estimacion, cfg):
    """(fits?, note). RAM may also use PSRAM (slower) when [hardware] psram_kb > 0."""
    if flash > estimacion["limite_flash"]:
        return False, "flash over budget"
    if ram <= estimacion["limite_ram"]:
        return True, ""
    psram = cfg["target"]["budget"] * cfg["hardware"]["psram_kb"] * 1024
    if ram <= estimacion["limite_ram"] + psram:
        return True, "needs PSRAM (slower)"
    return False, "RAM over budget"


def _texto_plan(filas, plan, ancho=1.0):
    return f"x{ancho:g} " + " ".join("b" if plan[f["name"]] == "bin" else str(plan[f["name"]]) for f in filas)


def _entrenar_plan(modelo, maestro, plan, primera, train, trl, val_l, cfg, epochs, dev, ancho=1.0, ruta=None,
                   verbose=False):
    """Builds the model for `plan` at width `ancho` from the original weights and trains it.
    Returns the result dict of train/ (the trained model is in res["modelo"])."""
    import torch
    from compiler_core.methods import apply_plan, init_thresholds, widen
    from compiler_core.train import destilar_estudiante, entrenar
    tr = cfg["training"]
    torch.manual_seed(tr["seed"])
    n = min(cfg["data"]["calibration_samples"], len(train))
    idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(tr["seed"]))[:n].tolist()
    calib = torch.stack([train[i][0] for i in idx])
    base = copy.deepcopy(modelo).cpu()
    if ancho > 1:
        widen(base, ancho, calib[:2], seed=tr["seed"])
    alumno = apply_plan(base, plan, primera=primera)
    init_thresholds(alumno, calib)
    comunes = dict(epochs=epochs, lr=tr["learning_rate"], device=dev, verbose=verbose, ruta_checkpoint=ruta,
                   guardar={"last": "ultima", "best": "mejor"}[tr["checkpoint"]],
                   promedio_ultimas=min(5, epochs), bn_momentum=0.3)
    if tr["distill"]:
        res = destilar_estudiante(alumno, maestro, trl, val_l, temperatura=tr["temperature"], alpha=tr["alpha"],
                                  **comunes)
    else:
        res = entrenar(alumno, trl, val_l, **comunes)
    res["modelo"] = alumno
    return res


def _referencias(modelo, train, val_l, filas, cfg, dev):
    """INT8 reference: the original model with every layer (input included) quantized to 8 bits, weights
    per output channel, activation ranges calibrated on training samples, NO retraining (post-training,
    like TFLite). Flash: 1 byte per weight + scale and zero point per channel (2 x int32)."""
    import torch
    import torch.nn as nn
    from compiler_core.methods import apply_plan, calibrate_ranges
    from compiler_core.train import evaluar
    capas = [m for _, m in modelo.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    flash_fp32 = sum(p.numel() * 4 for p in modelo.parameters())
    flash_int8 = sum(m.weight.numel() + m.weight.shape[0] * 8 for m in capas)
    int8 = apply_plan(copy.deepcopy(modelo).cpu(), {f["name"]: 8 for f in filas}, primera=filas[0]["name"],
                      mantener_entrada_primera=False).to(dev)
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    idx = torch.randperm(len(train), generator=gen)[:min(1024, len(train))].tolist()
    lotes = [torch.stack([train[i][0] for i in idx[j:j + 256]]).to(dev) for j in range(0, len(idx), 256)]
    calibrate_ranges(int8, lotes)
    acc = evaluar(int8.eval(), val_l, dev)
    del int8                                           # free the GPU memory of the reference
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return dict(acc_int8=acc, flash_int8=flash_int8, flash_fp32=flash_fp32)


def _predicciones(modelo, val_l, dev):
    import torch
    estaba = modelo.training
    modelo.eval()
    out = []
    with torch.no_grad():
        for x, _ in val_l:
            out.append(modelo(x.to(dev)).argmax(1).cpu())
    modelo.train(estaba)
    return torch.cat(out)


def stage_train(modelo, train, val, estimacion, cfg, salida, kind="mlp"):
    import torch
    from torch.utils.data import DataLoader
    from compiler_core.train import evaluar
    if isinstance(modelo, torch.jit.ScriptModule):
        sys.exit("[X] train: a TorchScript model cannot be rebuilt layer by layer; give the weights + definition")
    tr, t = cfg["training"], cfg["target"]
    filas = estimacion["filas"]
    primera = filas[0]["name"]
    palabra = cfg["hardware"]["cpu_bits"]
    opciones = sorted(t["options"], key=_bits)
    anchos = sorted(set(float(a) for a in t["widths"]))
    if len(anchos) > 1:                              # widening only works on plain chains: check it once
        from compiler_core.methods import widen
        try:
            widen(copy.deepcopy(modelo).cpu().eval(), anchos[-1], torch.stack([train[i][0] for i in range(2)]))
        except ValueError as e:
            print(f"    [!] this model cannot be widened ({e}); searched at width x1 only")
            anchos = [1.0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(tr["seed"])
    trl = DataLoader(train, batch_size=tr["batch_size"], shuffle=True, generator=gen, num_workers=2)
    val_l = DataLoader(val, batch_size=max(tr["batch_size"], 512), num_workers=2)
    maestro = modelo.to(dev).eval()
    acc_original = evaluar(maestro, val_l, dev)
    ref = _referencias(modelo, train, val_l, filas, cfg, dev)
    ref["acc_fp32"] = acc_original
    print(f"\n[3] REFERENCES  (the binarized model is compared against these)")
    print(f"    {'model':<44}{'accuracy':>9}{'flash KB':>11}")
    print(f"    {'FP32 original':<44}{acc_original:>8.2f}%{ref['flash_fp32'] / 1024:>11.1f}")
    print(f"    {'INT8 post-training (per-channel, like TFLite)':<44}{ref['acc_int8']:>8.2f}%"
          f"{ref['flash_int8'] / 1024:>11.1f}")

    # ---------------- search: 1 bit at width x1 first; then wider; then more bits on the most sensitive layer
    print(f"\n    SEARCH  (quick trials of {t['search_epochs']} epochs, accept a drop <= {t['max_accuracy_drop']} points;"
          f" widths {anchos}, options {opciones} | {dev})")
    print(f"    plan = width, then option per layer: {' '.join(f['name'] for f in filas)}   (b = 1 bit)")
    print(f"    {'step':>4} | {'plan':<26} | {'flash KB':>9} | {'accuracy':>8} | {'drop':>6} | decision")
    print("    " + "-" * 84)
    plan, ancho = {f["name"]: opciones[0] for f in filas}, anchos[0]
    flash, ram = _memoria_plan(filas, plan, ancho, palabra)
    ok, nota = _cabe(flash, ram, estimacion, cfg)
    if not ok:
        sys.exit(f"[X] even the smallest plan does not fit the budget ({nota}: flash {flash / 1024:.1f} KB, "
                 f"RAM {ram / 1024:.1f} KB)")
    historial, paso = [], 0
    while True:
        res = _entrenar_plan(modelo, maestro, plan, primera, train, trl, val_l, cfg, t["search_epochs"], dev, ancho)
        acc = res["promedio_final"]
        caida = acc_original - acc
        flash, ram = _memoria_plan(filas, plan, ancho, palabra)
        fila = dict(step=paso, width=ancho, plan={k: str(v) for k, v in plan.items()}, flash=flash, ram=ram,
                    accuracy=acc, drop=caida)
        historial.append(fila)
        siguiente = None
        if caida <= t["max_accuracy_drop"]:
            decision = "OK: within the accepted drop"
        else:
            # 1) wider, all-1-bit layers first
            mas_anchos = [a for a in anchos if a > ancho]
            if all(v == opciones[0] for v in plan.values()):
                for a in mas_anchos:
                    fl, rm = _memoria_plan(filas, plan, a, palabra)
                    if _cabe(fl, rm, estimacion, cfg)[0]:
                        siguiente = (f"wider -> x{a:g}", plan, a)
                        break
            # 2) then more bits on the most sensitive layer (lowest cosine at its current option)
            if siguiente is None:
                candidatas = sorted((f for f in filas if opciones.index(plan[f["name"]]) + 1 < len(opciones)),
                                    key=lambda f: f["opciones"][plan[f["name"]]]["fid"])
                for f in candidatas:
                    prueba = dict(plan)
                    prueba[f["name"]] = opciones[opciones.index(plan[f["name"]]) + 1]
                    fl, rm = _memoria_plan(filas, prueba, ancho, palabra)
                    if _cabe(fl, rm, estimacion, cfg)[0]:
                        siguiente = (f"raise {f['name']} -> {prueba[f['name']]}", prueba, ancho)
                        break
            decision = siguiente[0] if siguiente else "STOP: nothing else fits the budget or can go up"
        fila["decision"] = decision
        print(f"    {paso:>4} | {_texto_plan(filas, plan, ancho):<26} | {flash / 1024:>9.2f} | {acc:>7.2f}% | "
              f"{caida:>+6.2f} | {decision}")
        if siguiente is None:
            break
        _, plan, ancho = siguiente
        paso += 1
    if not decision.startswith("OK"):
        print(f"    [!] the accepted drop ({t['max_accuracy_drop']} points) was not reached; the last plan is used")

    # ---------------- final training of the chosen width and plan
    flash, ram = _memoria_plan(filas, plan, ancho, palabra)
    nota = _cabe(flash, ram, estimacion, cfg)[1]
    print(f"\n    FINAL TRAINING  ({tr['epochs']} epochs, {'distillation' if tr['distill'] else 'labels only'},"
          f" width x{ancho:g})")
    print(f"    {'layer':<32}{'option':>8}")
    for f in filas:
        op = plan[f["name"]]
        print(f"    {f['name']:<32}{'bin' if op == 'bin' else str(op) + ' bits':>8}"
              + ("   (input kept)" if f["name"] == primera else ""))
    ruta = os.path.join(salida, "model_binarized.pth")
    res = _entrenar_plan(modelo, maestro, plan, primera, train, trl, val_l, cfg, tr["epochs"], dev, ancho, ruta=ruta,
                         verbose=cfg["output"]["verbose"])
    ultima = res["historial"][-1][3]
    pred_b, pred_o = _predicciones(res["modelo"], val_l, dev), _predicciones(maestro, val_l, dev)
    etiquetas = torch.cat([y for _, y in val_l])
    identidad = 100 * float((pred_b == pred_o).float().mean())
    bien_o = pred_o == etiquetas
    retenido = 100 * float((pred_b[bien_o] == etiquetas[bien_o]).float().mean()) if bien_o.any() else 0.0
    ganado = int(((pred_b == etiquetas) & ~bien_o).sum())
    print(f"\n    binarized validation accuracy: last epoch {ultima:.2f}% | mean of last 5 {res['promedio_final']:.2f}%"
          f" | (best epoch {res['mejor_test']:.2f}%, optimistic)")
    print(f"\n    {'model':<34}{'accuracy':>9}{'vs FP32':>9}{'vs INT8':>9}{'flash KB':>10}{'flash vs INT8':>15}")
    for nombre, acc_m, fl in (("FP32 original", acc_original, ref["flash_fp32"]),
                              ("INT8 post-training", ref["acc_int8"], ref["flash_int8"]),
                              ("SYMBEX (" + _texto_plan(filas, plan, ancho) + ")", ultima, flash)):
        print(f"    {nombre:<34}{acc_m:>8.2f}%{acc_m - acc_original:>+9.2f}{acc_m - ref['acc_int8']:>+9.2f}"
              f"{fl / 1024:>10.1f}{ref['flash_int8'] / fl:>14.1f}x")
    print(f"    (flash vs INT8 = how many times less flash than the INT8 model; RAM peak SYMBEX {ram / 1024:.2f} KB"
          + (f", {nota}" if nota else "") + ")")
    print(f"    IDENTITY : the binarized model predicts the same class as the original on {identidad:.2f}% of the "
          f"validation samples (how much of the original's behavior it keeps)")
    print(f"    RETAINED : of the {int(bien_o.sum())} samples the original gets right, the binarized model still gets "
          f"{retenido:.2f}% right (knowledge kept); it also gets right {ganado} that the original got wrong")
    with open(os.path.join(salida, "train.json"), "w") as fh:
        json.dump(dict(width=ancho, plan={k: str(v) for k, v in plan.items()}, flash_total=flash, ram_peak=ram,
                       references=ref, acc_original=acc_original, acc_last=ultima,
                       acc_mean_last5=res["promedio_final"], acc_best_epoch=res["mejor_test"], identity=identidad, retained=retenido, gained=ganado,
                       search=historial, history=res["historial"]), fh, indent=1)
    print(f"    -> {ruta}\n    -> {os.path.join(salida, 'train.json')}")
    res["plan"] = plan
    res["ancho"] = ancho
    return res


def stage_export(entrenado, val, cfg, salida, kind, modelo_original=None, train=None):
    import torch
    from compiler_core.export.dense import exportar_y_validar, exportar_int8
    print("\n[4-5] EXPORT + VALIDATE")
    plan = entrenado["plan"]
    if kind != "mlp" or cfg["model"]["method"] != "plain":
        print("    [!] export is only integrated for MLPs with the plain method; skipped")
        return None
    if any(op != "bin" for op in plan.values()):
        print("    [!] the plan has k-bit layers (" + ", ".join(f"{k}: {v}" for k, v in plan.items() if v != "bin")
              + "): the C dense layer with k > 1 is not verified yet, so only all-1-bit plans are exported."
              "\n        Run with --options bin to get an exportable model.")
        return None
    try:
        res = exportar_y_validar(entrenado["modelo"], val, salida, cfg["output"]["lib_path"], cfg["data"]["mean"],
                                 cfg["data"]["std"], cfg["output"]["validation_samples"])
    except ValueError as e:
        print(f"    [X] {e}")
        return None
    if modelo_original is not None and train is not None:
        try:
            gen = torch.Generator().manual_seed(cfg["training"]["seed"])
            idx = torch.randperm(len(train), generator=gen)[:min(1024, len(train))].tolist()
            calib = torch.stack([train[i][0] for i in idx])
            res["int8"] = exportar_int8(modelo_original, calib, res["_xs"], res["_x_u8"], res["_y"], salida,
                                        cfg["output"]["lib_path"], cfg["data"]["mean"], cfg["data"]["std"])
        except ValueError as e:
            print(f"    [!] INT8 reference not exported: {e}")
    with open(os.path.join(salida, "export.json"), "w") as fh:
        json.dump({k: v for k, v in res.items() if not k.startswith("_")}, fh, indent=1)
    print(f"    -> {os.path.join(salida, 'export.json')}")
    return res


def _int8_de(res_export, salida):
    """The INT8 reference exported with the model, if any, in the form medir_en_chip expects."""
    import numpy as np
    r8 = res_export.get("int8")
    ruta = os.path.join(salida, "validacion_int8.u8")
    if not r8 or not os.path.exists(ruta):
        return None
    return dict(header=r8["header"], esperado=np.fromfile(ruta, dtype=np.uint8).astype(np.int64), flash=r8["flash_bytes"])


def stage_chip(res_export, cfg, salida):
    import numpy as np
    from compiler_core.export.dense import medir_en_chip
    ch = cfg["chip"]
    print(f"\n[6] CHIP  ({ch['board']} on {ch['port']}, {ch['samples']} validation images)")
    try:
        res = medir_en_chip(res_export["header"], res_export["_x_u8"], res_export["_y"], res_export["_pred_sim"],
                            os.path.join(salida, f"chip_{ch['board']}"), cfg["output"]["lib_path"], ch["board"],
                            ch["port"], ch["baud"], ch["samples"], ch["upload"], ch["popcount"],
                            res_export["flash_bytes"], int8=_int8_de(res_export, salida))
    except (RuntimeError, ValueError) as e:
        print(f"    [X] {e}")
        return None
    with open(os.path.join(salida, "chip.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    -> {os.path.join(salida, 'chip.json')}")
    return res


def main():
    args = build_parser().parse_args()
    cfg, origen = load_config(args)
    if args.show_config:
        print_config(cfg, origen)
        return
    validate_config(cfg)
    salida = output_directory(cfg)
    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, "config_used.toml"), "w") as fh:
        fh.write(f"# written by cdt.py on {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + to_toml(cfg))
    if cfg["output"]["verbose"]:
        print_config(cfg, origen)
    print(f"\n[*] output directory: {salida}   (config_used.toml written)")

    import torch
    torch.manual_seed(cfg["training"]["seed"])
    modelo, origen_modelo = load_model(cfg)
    kind = stage_inspect(modelo, origen_modelo, cfg)
    if args.stop_after == "inspect":
        return
    train, val = load_data(cfg)
    print(f"\n[*] data: {len(train)} training / {len(val)} validation samples ({cfg['data']['format']})")
    estimacion = stage_estimate(modelo, train, cfg, salida)
    if args.stop_after == "estimate":
        return
    entrenado = stage_train(modelo, train, val, estimacion, cfg, salida, kind)
    if args.stop_after == "train":
        return
    res_export = stage_export(entrenado, val, cfg, salida, kind, modelo, train)
    if args.stop_after in ("export", "validate") or res_export is None:
        return
    if cfg["chip"]["board"]:
        stage_chip(res_export, cfg, salida)
    else:
        print("\n[6] CHIP  skipped: set [chip] board = \"esp32\" or \"s3\" (or --board esp32) to measure on a board")


if __name__ == "__main__":
    main()
