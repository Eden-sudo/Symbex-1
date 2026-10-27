"""
train_bench.py -- how fast does the CDT train, and what does it use? A development tool, for the training speed.

    python train_bench.py                              # every configuration, one training at a time
    python train_bench.py --parallel 1 2 3 4           # the best data path with 1, 2, 3, 4 trainings at once
    python train_bench.py --configs cpu gpu --steps 300
    python train_bench.py --try-compile                # also torch.compile (slow to start: compiles first)

Run from tools/dev/ (environment env_bench). It trains the binary ResNet-8 of benchmarks/conv2d/cnn_cifar_resnet8 (model.py,
BoolResNet8x1) on CIFAR-10 with the training step of the CDT's boolean method (compiler_core/export/boolean.py: crop +
flip on the GPU, cross-entropy, Adam) for --steps batches after --warmup batches, and measures:
    speed      images per second and ms per step (warm-up excluded)
    GPU        mean utilization (nvidia-smi: the whole GPU, other programs included) and peak memory of this training
    CPU        mean utilization of the whole process tree (data loader workers included; 100% = one core),
               threads and processes
    RAM        resident memory of the process tree
Configurations:
    cpu        the CDT today: a DataLoader decodes and converts every image on the CPU (4 workers), every epoch
    gpu        the whole training set on the GPU once (uint8), batches taken from it there
    gpu+cl     ... with channels_last memory format
    gpu+bf16   ... with bf16 mixed precision (autocast)
    gpu+all    ... both
    gpu+b512   ... batches of 512 (4x fewer steps per epoch; the training itself changes: lr, accuracy to check)
    gpu+graphs ... CUDA graphs (torch.compile mode "reduce-overhead": the step is recorded once and replayed)
    gpu+noaug  ... no crop + flip: only to measure what the augmentation costs (not a way to train)
    gpu+compile  ... torch.compile of the model (with --try-compile)
Speed only: before a configuration goes into the CDT, its final accuracy must be checked against the current one.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # tools/: compiler_core
from compiler_core.export.boolean import _augment                                             # noqa: E402

CONFIGS = {"cpu": dict(data="cpu"), "gpu": dict(data="gpu"), "gpu+cl": dict(data="gpu", cl=True),
           "gpu+bf16": dict(data="gpu", bf16=True), "gpu+all": dict(data="gpu", cl=True, bf16=True),
           "gpu+b512": dict(data="gpu", batch=512), "gpu+graphs": dict(data="gpu", compile="reduce-overhead"),
           "gpu+noaug": dict(data="gpu", noaug=True),
           "gpu+compile": dict(data="gpu", cl=True, bf16=True, compile=True)}


def describe_device():
    """the accelerator PyTorch sees, detected (never assumed): CUDA (NVIDIA, or AMD through ROCm), XPU (Intel), MPS
    (Apple), else the CPU; with its name, memory and whether it is integrated (shares the system memory) or dedicated"""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        kind = "integrated" if getattr(p, "is_integrated", False) else "dedicated"
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        return (f"{p.name} | {backend}, {kind}, {p.total_memory / 2 ** 30:.1f} GB, {p.multi_processor_count} SMs"
                + (f" | {torch.cuda.device_count()} GPUs, using the first" if torch.cuda.device_count() > 1 else ""))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return f"{torch.xpu.get_device_name(0)} | Intel XPU"
    if torch.backends.mps.is_available():
        return "Apple GPU | MPS (integrated)"
    return "no accelerator: CPU"


def load_model(folder, name):
    spec = importlib.util.spec_from_file_location("bench_model", os.path.join(folder, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, name)()


class Monitor(threading.Thread):
    """samples the process tree (CPU, threads, processes, RAM) and the GPU utilization every 0.5 s"""
    def __init__(self, pid):
        super().__init__(daemon=True)
        import psutil
        self.proc, self.stop, self.rows = psutil.Process(pid), threading.Event(), []
        self.known = {}                                  # pid -> Process: cpu_percent needs the SAME object twice

    def tree(self):
        import psutil
        out = []
        for p in [self.proc] + self.proc.children(recursive=True):
            if p.pid not in self.known:
                self.known[p.pid] = p
                try:
                    p.cpu_percent(None)                  # first call: starts its measurement (returns 0)
                except psutil.Error:
                    pass
            out.append(self.known[p.pid])
        return out

    def run(self):
        import psutil
        self.tree()
        while not self.stop.wait(0.5):
            cpu = threads = ram = 0
            procs = self.tree()
            for p in procs:
                try:
                    cpu += p.cpu_percent(None)
                    threads += p.num_threads()
                    ram += p.memory_info().rss
                except psutil.Error:
                    pass
            try:
                gpu = float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=2).stdout.split()[0])
            except Exception:
                gpu = float("nan")
            self.rows.append((cpu, threads, len(procs), ram, gpu))

    def summary(self):
        if not self.rows:
            return {}
        n = len(self.rows)
        return dict(cpu_percent=sum(r[0] for r in self.rows) / n, threads=max(r[1] for r in self.rows),
                    processes=max(r[2] for r in self.rows), ram_mb=max(r[3] for r in self.rows) / 2 ** 20,
                    gpu_util=sum(r[4] for r in self.rows) / n)


def train_run(cfg, args):
    """one training of --warmup + --steps batches in this process; returns the measurements"""
    import torchvision
    dev = torch.device("cuda")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    model = load_model(args.model_dir, args.model_class).to(dev)
    if cfg.get("cl"):
        model = model.to(memory_format=torch.channels_last)
    mode = cfg.get("compile")
    step_model = (torch.compile(model, mode=mode) if isinstance(mode, str) else torch.compile(model)) if mode else model
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    bs = cfg.get("batch", args.batch_size)
    if cfg["data"] == "cpu":
        from torch.utils.data import DataLoader
        ds = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True,
                            pin_memory=True)

        def batches():
            while True:
                for x, y in loader:
                    yield x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    else:
        ds = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True)
        xs = torch.as_tensor(ds.data, device=dev).permute(0, 3, 1, 2).contiguous()          # uint8, N x 3 x 32 x 32
        ys = torch.as_tensor(ds.targets, device=dev)

        def batches():
            while True:
                perm = torch.randperm(len(xs), device=dev)
                for i in range(0, len(perm) - bs + 1, bs):
                    idx = perm[i:i + bs]
                    yield xs[idx].float().div_(255), ys[idx]
    it = batches()
    torch.cuda.reset_peak_memory_stats()

    def step():
        x, y = next(it)
        if not cfg.get("noaug"):
            x = _augment(x)
        if cfg.get("cl"):
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg.get("bf16"))):
            loss = F.cross_entropy(step_model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t_start = time.time()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t_ready = time.time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t_ready
    return dict(images_per_s=args.steps * bs / dt, ms_per_step=1000 * dt / args.steps, start_s=t_ready - t_start,
                gpu_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20)


def measure(name, args, n_parallel=1):
    """n_parallel trainings of configuration `name` at once (child processes), measured from here"""
    cmd = [sys.executable, os.path.abspath(__file__), "--child", name] + sum(
        ([f"--{k.replace('_', '-')}", str(v)] for k, v in (("steps", args.steps), ("warmup", args.warmup),
                                                           ("batch_size", args.batch_size), ("data_dir", args.data_dir),
                                                           ("model_dir", args.model_dir),
                                                           ("model_class", args.model_class))), [])
    mon = Monitor(os.getpid())
    mon.start()
    t0 = time.time()
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) for _ in range(n_parallel)]
    outs = [p.communicate()[0] for p in procs]
    wall = time.time() - t0
    mon.stop.set()
    mon.join()
    res = [json.loads(o.strip().splitlines()[-1]) for o in outs if o.strip()]
    if len(res) != n_parallel:
        sys.exit(f"[X] {name}: a training failed (run it alone with --parallel 1 to see why)")
    total = sum(r["images_per_s"] for r in res)
    return dict(config=name, parallel=n_parallel, images_per_s=total, ms_per_step=sum(r["ms_per_step"] for r in res) / len(res),
                start_s=max(r["start_s"] for r in res), gpu_mem_mb=sum(r["gpu_mem_mb"] for r in res), total_s=wall,
                per_training_s=wall / n_parallel, **mon.summary())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["cpu", "gpu", "gpu+b512", "gpu+graphs", "gpu+noaug"],
                    help="any of: " + ", ".join(CONFIGS))
    ap.add_argument("--try-compile", action="store_true", help="also gpu+compile (torch.compile, default mode)")
    ap.add_argument("--parallel", nargs="+", type=int, default=[], help="trainings at once, for the --parallel-config")
    ap.add_argument("--parallel-config", default="gpu", help="the configuration of the parallel measurements")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128, help="the benchmark's batch size (symbex.toml): 128")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "..", "benchmarks", "data", "cifar10"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "..", "benchmarks", "conv2d", "cnn_cifar_resnet8"))
    ap.add_argument("--model-class", default="BoolResNet8x1")
    ap.add_argument("--child", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(train_run(CONFIGS[args.child], args)))
        return
    if not torch.cuda.is_available():
        sys.exit("[X] no CUDA GPU visible to PyTorch in this environment")
    names = list(args.configs) + (["gpu+compile"] if args.try_compile else [])
    rows = [measure(n, args) for n in names] if not args.parallel else []
    rows += [measure(args.parallel_config, args, n) for n in args.parallel]
    print(f"\n    TRAINING BENCH  ({args.model_class}, CIFAR-10, batch {args.batch_size}, {args.steps} steps after "
          f"{args.warmup} of warm-up)\n    device: {describe_device()}\n")
    base = rows[0]["images_per_s"]
    print(f"    {'config':<13}{'trainings':>10}{'images/s':>10}{'vs first':>9}{'ms/step':>9}{'GPU %':>7}{'GPU MB':>8}"
          f"{'CPU %':>8}{'threads':>8}{'procs':>6}{'RAM MB':>8}{'start s':>8}{'total s':>8}{'s/train':>8}")
    print("    " + "-" * 122)
    for r in rows:
        print(f"    {r['config']:<13}{r['parallel']:>10}{r['images_per_s']:>10.0f}{r['images_per_s'] / base:>8.2f}x"
              f"{r['ms_per_step']:>9.2f}{r.get('gpu_util', float('nan')):>7.0f}{r['gpu_mem_mb']:>8.0f}"
              f"{r.get('cpu_percent', float('nan')):>8.0f}{r.get('threads', 0):>8}{r.get('processes', 0):>6}"
              f"{r.get('ram_mb', 0):>8.0f}{r['start_s']:>8.1f}{r['total_s']:>8.1f}{r['per_training_s']:>8.1f}")
    print("\n    images/s: all the trainings together | CPU %: 100 = one core | GPU %: the whole GPU (other programs too)"
          "\n    start s: model + data loading + warm-up (torch.compile compiles here) | total s: the whole row, end to end"
          "\n    s/train: total s / trainings (the time one training effectively costs) | gpu+b512: 4x the images per step")
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    path = os.path.join(HERE, "runs", "train_bench.json")
    json.dump(rows, open(path, "w"), indent=1)
    print(f"    -> {path}")


if __name__ == "__main__":
    main()
