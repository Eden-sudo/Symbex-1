"""
train_gap_study.py -- how much of the binary ResNet-8's accuracy loss comes from the training, not from the bits?
Step A, the diagnostic: no new method, the benchmark's recipe as it is, three networks of the same graph:

    binary   cnn_cifar_resnet8's BoolResNet8x1 (thermometer input, 1-bit weights and activations, OR shortcuts)
    twin     the same graph with real weights and hardtanh activations: the smooth network the straight-through
             estimator pretends to train (sign forward, gradient 1 where |x| <= 1 is hardtanh's gradient)
    relu     real weights, ReLU activations, added shortcuts (a plain floating-point ResNet-8, same thermometer input)

    python train_gap_study.py --model binary --seed 42
    python train_gap_study.py --model all --seeds 42 43 44
    python train_gap_study.py --model binary --recipe clip bop twostep --seed 42

Same recipe for the three (the benchmark's, boolean.train): Adam lr 5e-3 + cosine, 150 epochs, batch 128, random crop
+ flip, BatchNorm momentum 0.3, labels only, the same starting weights (models/bool_resnet8_x1_init.pth) and the
CDT's split (45,000 training / 5,000 validation, carved out of the training data). The test set is never read.
Every epoch it records:
    train_acc        accuracy on 5,000 fixed training images (no augmentation): train ~ validation and both low means
                     the network underfits (optimization or capacity), not that it overfits
    val_acc, loss    validation accuracy, mean training loss of the epoch
    ste_dead         share of the activations' inputs with |x| > 1 (the estimator passes no gradient there; twin:
                     hardtanh saturated; relu: only reported), on 1,000 fixed validation images
    binary only:
    flips_step       share of the 1-bit weights whose sign changed, summed over every step of the epoch
    flips_epoch      share whose sign differs from the previous epoch; flips_step / flips_epoch >> 1 means the signs
                     oscillate (flip and flip back) instead of settling
    near_zero        share of latent weights with |w| < the current learning rate (one Adam step from flipping)
    frozen           share of latent weights with |w| > 1: model.py's estimator gives them no gradient, they never
                     flip again
Step B, the recipes (binary network only, --recipe, one or several; each is chosen on validation, never on test):
    base             the benchmark's recipe (above)
    clip             latent weights clamped to [-1, 1] after every step (BinaryConnect): none freezes
    bop              Bop for the 1-bit weights (--bop-gamma, --bop-tau; Adam for the BatchNorms), no latent weights
    twostep          half the epochs with binary activations and real weights, then binary weights too, a fresh
                     Adam + cosine (Liu et al. 2021); same total epochs
    twostep_bop      twostep, with Bop for the 1-bit weights in the second half (they start from the signs of the first)
--distill: the loss is 0.9 * T^2 * KL(teacher || student at temperature T = 4) + 0.1 * cross-entropy, the teacher the relu
network of the same seed (trained first: its weights in runs/train_gap/relu_s<seed>.pth). Shown next to the labels-only
recipe, never instead of it. --epochs other than 150 and --distill add _e<epochs> / _kd to the file names.
--stem int8: the three networks with the int8 first layer on the pixels (cnn_cifar_resnet8's int8stem variant, its
starting weights); files <model>_i8_...
--width 2: twice the channels (32-64-128, BoolResNet8x2): if the binary network's training accuracy rises with width
and not with better recipes, the limit is the capacity of the 1-bit network, not its training. Files <model>_w2_...
Results: results/train_gap/<model>_s<seed>.json (<model>_<recipe>_s<seed>.json for step B). Run from anywhere; env_bench (CUDA) recommended.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(REPO, "benchmarks", "conv2d", "cnn_cifar_resnet8")
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
import model as resnet8                                    # noqa: E402
from compiler_core.export.boolean import _augment          # noqa: E402

MODELS = ("binary", "twin", "relu")
RECIPES = ("base", "clip", "bop", "twostep", "twostep_bop")
RUNS = os.path.join(HERE, "runs", "train_gap")              # the trained weights (gitignored): the teachers of --distill
OUT = os.path.join(HERE, "results", "train_gap")

_binary_sign = resnet8.sign
STEM = "thermometer"                                       # --stem: the first layer of the three networks
WIDTH = 1                                                  # --width: channels x1 (16-32-64) or x2 (32-64-128)
_record = None                                             # list: the activations' inputs, while it is set


def _sign_for(kind, real_weights=False):
    """what model.py's sign() becomes: the binary network keeps it; the floating-point ones keep real weights
    (parameters pass through) and replace the activation. real_weights: the binary network with real weights (the
    first step of the twostep recipe)"""
    def act(x):
        if isinstance(x, nn.Parameter):
            return _binary_sign(x) if kind == "binary" and not real_weights else x
        if _record is not None:
            _record.append(x.detach())
        if kind == "binary":
            return _binary_sign(x)
        return F.hardtanh(x) if kind == "twin" else F.relu(x)
    return act


def build(kind):
    resnet8.sign = _sign_for(kind)                         # model.py reads its global sign() at every forward
    net = resnet8.BoolResNet8(float(WIDTH), 8, stem=STEM, shortcut="add" if kind == "relu" else "or")
    if STEM == "int8" and WIDTH != 1:
        raise SystemExit("[X] --stem int8 with --width 2: the benchmark has no such variant")
    if STEM == "int8":                                     # the benchmark's starting weights of each variant
        cls, init = resnet8.BoolResNet8Int8Stem, "bool_resnet8_int8stem_init.pth"
    elif WIDTH == 2:
        cls, init = resnet8.BoolResNet8x2, "bool_resnet8_x2_init.pth"
    else:
        cls, init = resnet8.BoolResNet8x1, "bool_resnet8_x1_init.pth"
    init = os.path.join(BENCH, "models", init)
    net.load_state_dict(torch.load(resnet8.initial_weights(cls, init), map_location="cpu"))
    return net


def tag(kind):
    """the network's name in the file names: <kind> (thermometer stem, x1), <kind>_i8 (--stem int8), <kind>_w2"""
    return kind + ("_i8" if STEM == "int8" else "") + ("_w2" if WIDTH == 2 else "")


def load_split(seed):
    """the CDT's (train, validation) split of the CIFAR-10 training set, as the benchmark runs it ([data] defaults)"""
    import cdt
    cfg = {"data": dict(cdt.DEFAULTS["data"], format="cifar10", directory=os.path.join(REPO, "benchmarks", "data", "cifar10")),
           "training": {"seed": seed}}
    train, val, _ = cdt.load_data(cfg)
    return train, val


def _to_gpu(ds, dev):
    from torch.utils.data import DataLoader
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=1024, num_workers=4):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(dev), torch.cat(ys).to(dev)


@torch.no_grad()
def _accuracy(net, x, y, bs=1000):
    net.eval()
    hits = sum((net(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    return 100.0 * hits / len(x)


@torch.no_grad()
def _ste_dead(net, x):
    global _record
    net.eval()
    _record = []
    net(x)
    inputs, _record = _record, None
    dead = sum((t.abs() > 1).sum().item() for t in inputs)
    return dead / sum(t.numel() for t in inputs)


class Bop:
    """Bop (Helwegen et al. 2019) for the 1-bit weights: no latent magnitude, an exponential moving average m of the
    gradient; a weight flips when |m| > tau and m pushes against its sign. gamma follows the learning rate's cosine"""
    def __init__(self, weights, gamma, tau):
        self.weights, self.gamma, self.tau = weights, gamma, tau
        self.m = [torch.zeros_like(w) for w in weights]
        with torch.no_grad():
            for w in weights:
                w.copy_(torch.where(w >= 0, 1.0, -1.0))

    @torch.no_grad()
    def step(self, scale):
        g = self.gamma * scale
        for w, m in zip(self.weights, self.m):
            if w.grad is None:
                continue
            m.mul_(1 - g).add_(w.grad, alpha=g)
            flip = (m.abs() > self.tau) & ((m >= 0) == (w >= 0))
            w.masked_fill_(flip, 0).add_(torch.where(flip, -torch.sign(m), 0.0))   # flipped: -sign(m) = the opposite sign

    def zero_grad(self):
        for w in self.weights:
            w.grad = None


def recipe_name(recipe, gamma, tau, epochs=150, distill=False):
    name = recipe + (f"_g{gamma:g}_t{tau:g}" if recipe in ("bop", "twostep_bop") else "")
    return name + (f"_e{epochs}" if epochs != 150 else "") + ("_kd" if distill else "")


def load_teacher(seed, dev):
    """the relu network of this seed, frozen, and a forward that runs it with its own activation (model.py's sign()
    is one global: swapped around the teacher's call, put back for the student)"""
    path = os.path.join(RUNS, f"{tag('relu')}_s{seed}.pth")
    if not os.path.exists(path):
        raise SystemExit(f"[X] --distill: {path} not found (train it first: --model relu --seed {seed}, same --stem)")
    student_sign = resnet8.sign
    teacher = build("relu")
    teacher.load_state_dict(torch.load(path, map_location="cpu"))
    teacher_sign = resnet8.sign
    resnet8.sign = student_sign
    teacher = teacher.to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    @torch.no_grad()
    def forward(x):
        saved = resnet8.sign
        resnet8.sign = teacher_sign
        try:
            return teacher(x)
        finally:
            resnet8.sign = saved
    return forward


def run(kind, seed, epochs, lr, bs, compile_mode, recipe="base", bop_gamma=1e-4, bop_tau=1e-8, distill=False):
    if kind != "binary" and (recipe != "base" or distill):
        raise SystemExit(f"[X] --recipe {recipe} / --distill: only for the binary network")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_split(seed)
    xs, ys = _to_gpu(train, dev)
    xv, yv = _to_gpu(val, dev)
    torch.manual_seed(seed)
    net = build(kind).to(dev)
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0.3
    probe = torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:5000].to(dev)
    x_probe, y_probe = xs[probe], ys[probe]
    latent = [m.weight for m in net.modules() if isinstance(m, (resnet8.BinaryConv, resnet8.BinaryLinear))]
    n_latent = sum(w.numel() for w in latent)
    step_model = net
    if compile_mode and dev.type == "cuda":
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = min(4, os.cpu_count() or 1)
        step_model = torch.compile(net, mode=None if compile_mode == "default" else compile_mode)
    bop = Bop(latent, bop_gamma, bop_tau) if recipe == "bop" else None
    ids = {id(w) for w in latent}
    two_steps = recipe in ("twostep", "twostep_bop")
    phase_epochs = epochs // 2 if two_steps else epochs
    teacher = load_teacher(seed, dev) if distill else None

    def new_optimizer(n_epochs):
        params = [p for p in net.parameters() if id(p) not in ids] if bop else net.parameters()
        o = torch.optim.Adam(params, lr=lr)
        return o, torch.optim.lr_scheduler.CosineAnnealingLR(o, n_epochs)

    if two_steps:                                          # step 1: binary activations, real weights
        resnet8.sign = _sign_for(kind, real_weights=True)
    opt, sched = new_optimizer(phase_epochs)
    gen = torch.Generator(device=dev).manual_seed(seed)
    name = recipe_name(recipe, bop_gamma, bop_tau, epochs, distill)
    print(f"[{kind} {name} s{seed}] {epochs} epochs, Adam lr {lr} + cosine, batch {bs}, {len(xs)} train / {len(xv)} "
          f"validation | {dev}" + (f", compile {compile_mode}" if compile_mode and dev.type == "cuda" else ""))
    history, t0 = [], time.time()
    for ep in range(epochs):
        if two_steps and ep == phase_epochs:               # step 2: binary weights too, a fresh Adam + cosine
            resnet8.sign = _sign_for(kind)
            if recipe == "twostep_bop":                    # the 1-bit weights to Bop, from the signs of step 1
                bop = Bop(latent, bop_gamma, bop_tau)
            opt, sched = new_optimizer(epochs - phase_epochs)
        net.train()
        lr_now = opt.param_groups[0]["lr"]
        start = [w.detach() >= 0 for w in latent]
        prev = [s.clone() for s in start]
        flips_step = torch.zeros((), device=dev)
        loss_sum = torch.zeros((), device=dev)
        perm = torch.randperm(len(xs), device=dev, generator=gen)
        steps = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            xa = _augment(xs[idx])
            out = step_model(xa)
            if teacher is None:
                loss = F.cross_entropy(out, ys[idx])
            else:
                kl = F.kl_div(F.log_softmax(out / 4, 1), F.log_softmax(teacher(xa) / 4, 1), log_target=True,
                              reduction="batchmean")
                loss = 0.9 * 16 * kl + 0.1 * F.cross_entropy(out, ys[idx])
            opt.zero_grad(set_to_none=True)
            if bop:
                bop.zero_grad()
            loss.backward()
            opt.step()
            if bop:
                bop.step(opt.param_groups[0]["lr"] / lr)
            elif recipe == "clip":                         # latent weights kept where the estimator passes gradient
                with torch.no_grad():
                    for w in latent:
                        w.clamp_(-1, 1)
            loss_sum += loss.detach()
            steps += 1
            if kind == "binary":
                with torch.no_grad():
                    for j, w in enumerate(latent):
                        now = w >= 0
                        flips_step += (now != prev[j]).sum()
                        prev[j] = now
        sched.step()
        row = dict(epoch=ep + 1, lr=lr_now, loss=loss_sum.item() / steps,
                   train_acc=_accuracy(net, x_probe, y_probe), val_acc=_accuracy(net, xv, yv),
                   ste_dead=_ste_dead(net, xv[:1000]))
        if kind == "binary":
            with torch.no_grad():
                row["flips_step"] = flips_step.item() / n_latent
                row["flips_epoch"] = sum(((w >= 0) != s).sum().item() for w, s in zip(latent, start)) / n_latent
                row["near_zero"] = sum((w.abs() < lr_now).sum().item() for w in latent) / n_latent
                row["frozen"] = sum((w.abs() > 1).sum().item() for w in latent) / n_latent
        history.append(row)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            extra = (f" | flips/step {row['flips_step']:.4f} epoch {row['flips_epoch']:.4f} near0 {row['near_zero']:.3f}"
                     f" frozen {row['frozen']:.3f}"
                     if kind == "binary" else "")
            print(f"    epoch {ep + 1:3d} | loss {row['loss']:.3f} | train {row['train_acc']:.2f}% | validation "
                  f"{row['val_acc']:.2f}% | ste dead {row['ste_dead']:.3f}{extra} | {(time.time() - t0) / 60:.1f} min")
    last = history[-1]
    info = dict(model=kind, stem=STEM, width=WIDTH, recipe=name, distill=distill, seed=seed, epochs=epochs, learning_rate=lr, batch_size=bs,
                n_train=len(xs),
                n_validation=len(xv), n_latent_binary=n_latent if kind == "binary" else None,
                train_acc=last["train_acc"], val_acc=last["val_acc"], history=history,
                minutes=round((time.time() - t0) / 60, 2))
    os.makedirs(OUT, exist_ok=True)
    stem = f"{tag(kind)}_s{seed}" if name == "base" else f"{tag(kind)}_{name}_s{seed}"
    path = os.path.join(OUT, stem + ".json")
    json.dump(info, open(path, "w"), indent=1)
    os.makedirs(RUNS, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, os.path.join(RUNS, stem + ".pth"))
    print(f"    -> {path} | train {last['train_acc']:.2f}% | validation {last['val_acc']:.2f}%")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", choices=MODELS + ("all",), default="all")
    ap.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--learning-rate", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--compile", default="reduce-overhead", help="torch.compile mode ('' to disable)")
    ap.add_argument("--recipe", nargs="+", choices=RECIPES, default=["base"],
                    help="step B, binary network only (see the docstring)")
    ap.add_argument("--bop-gamma", type=float, default=1e-4)
    ap.add_argument("--bop-tau", type=float, default=1e-8)
    ap.add_argument("--distill", action="store_true", help="binary network: distill from the relu network of the seed")
    ap.add_argument("--stem", choices=("thermometer", "int8"), default="thermometer",
                    help="first layer of the three networks: int8 = the benchmark's int8stem variant (files *_i8_*)")
    ap.add_argument("--width", type=int, choices=(1, 2), default=1,
                    help="2: twice the channels (BoolResNet8x2), the capacity check (files *_w2_*)")
    args = ap.parse_args()
    global STEM, WIDTH
    STEM, WIDTH = args.stem, args.width
    kinds = MODELS if args.model == "all" else (args.model,)
    done = []
    for seed in args.seeds:
        for kind in kinds:
            for recipe in (args.recipe if kind == "binary" else ["base"]):
                done.append(run(kind, seed, args.epochs, args.learning_rate, args.batch_size, args.compile, recipe,
                                args.bop_gamma, args.bop_tau, args.distill and kind == "binary"))
    print("\nmodel   recipe                       seed  train %  validation %")
    for r in done:
        print(f"{r['model']:7s} {r['recipe']:28s} {r['seed']:4d}  {r['train_acc']:7.2f}  {r['val_acc']:12.2f}")


if __name__ == "__main__":
    main()
