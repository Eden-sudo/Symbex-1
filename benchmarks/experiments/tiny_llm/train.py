"""
train.py -- does a tiny language model still write stories with 1-bit or ternary weights? A Llama-style transformer
(RMSNorm, rotary positions, SwiGLU, no biases) trained on TinyStories (prepare.py) with one of three weight formats,
same recipe for all:

    python train.py --weights fp32    --seeds 42 43 44
    python train.py --weights ternary --seeds 42 43 44     # {-1, 0, +1} x one scale per output row (BitNet b1.58)
    python train.py --weights binary  --seeds 42 43 44     # {-1, +1} x one scale per output row
    python train.py --weights ternary --steps 200 --name smoke   # quick check of the whole script
    python train.py --weights binary --head binary --name d384l8_head1        # also a 1-bit output layer
    python train.py --weights binary --int8-blocks 0 7 --name d384l8_b07i8    # blocks 0 and 7 with int8 weights
    python train.py --weights binary --acts 1 --name d384l8_a1                # 1-bit inputs too (W1A1)

--head binary: the output layer (tied to the embedding) gets 1-bit weights (sign x mean|w| per row, the embedding
lookup keeps the fp32 weights) and 8-bit inputs. --int8-blocks: those blocks' linear layers get int8 weights (absmax
per row, 127 levels) instead of --weights. Both come from layer_sensitivity.py (the head and blocks 0 / 7 are the most
damaging layers to binarize); intsim.py and the C do not support them yet. --head int4 (absmax per row, 15 levels)
/ int2 (four levels {-1.5, -0.5, 0.5, 1.5} x mean|w| per row): the same, but the embedding lookup uses the quantized
weights too (one table on the chip, so the bytes counted are the real ones).
--acts 1: the blocks' linear layers get 1-bit inputs (sign x mean |x| per token, identity straight-through gradient)
instead of 8-bit ones, so every block's matrix product is 1 bit x 1 bit; the output layer, the norms and attention
(scores, softmax, values) keep their formats. Not in intsim.py / the C yet.
W1A1 screening options (w1a1_diag.py, LOGBOOK.md): --ste clip [--ste-k 2] (gradient only where |x| <= k x mean|x|),
--learn-thr (sign(x - thr), one learned thr per input channel), --a8-inputs w2 (those layers keep 8-bit inputs;
--a8-blocks 0: only in block 0),
--clip-groups (gradient norm clipped per block, not over the whole model), --init-from runs/binary_d384l8_s{seed}/
model.pt (two-step: start from the W1A8 model), --w2-acts t|2 (w2's input ternary or 2 bits: the SwiGLU output
is mostly near 0, w1a1_diag.py), --ffn-act relu (relu gate instead of silu), --hadamard w2 o (block Hadamard on those inputs before
quantization: BitNet v2 / QuEST), --distill runs/fp32_d384l8_s{seed}/model.pt [--distill-alpha 0.5]
(loss = (1 - alpha) x cross-entropy on the labels + alpha x KL(teacher || student) on the teacher's next-token
distribution, temperature --distill-t).

Quantized runs: every linear layer of the blocks has quantized weights (scale = mean |w| of the row; ternary rounds
w / scale to -1, 0, 1, binary takes the sign) and 8-bit inputs (absmax per token, 127 levels), both through the
straight-through estimator. The token embedding (tied to the output layer), the norms and attention's softmax stay in
floating point here: an int8 embedding and the integer attention are for the chip stage.

Choices on the validation split only (train.py keeps the checkpoint with the lowest validation loss); --test reads the
test split once, for the final numbers. Writes runs/<weights>_<name>_s<seed>/{model.pt, train.json}: the loss history,
the best validation loss and perplexity, the parameter counts, the time and a sample story.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "tinystories")


# ---- quantization --------------------------------------------------------------------------------------------------

def ste(x, q):
    return x + (q - x).detach()


def quant_weight(w, kind):
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    if kind == "ternary":
        return ste(w, (w / scale).round().clamp(-1, 1) * scale)
    if kind == "int8":
        s8 = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / 127.0
        return ste(w, (w / s8).round().clamp(-127, 127) * s8)
    if kind == "int4":
        s4 = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / 7.0
        return ste(w, (w / s4).round().clamp(-7, 7) * s4)
    if kind == "int2":
        return ste(w, ((w / scale).floor().clamp(-2, 1) + 0.5) * scale)
    return ste(w, torch.where(w >= 0, 1.0, -1.0) * scale)


def quant_act8(x):
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return ste(x, (x * scale).round().clamp(-128, 127) / scale)


def quant_act1(x, ste_mode="identity", ste_k=2.0, thr=None):
    """1-bit input sign(x - thr) x mean|x - thr|. ste_mode "clip": the gradient passes only where
    |x - thr| <= ste_k x mean|x - thr| (clipped relative to the scale), "identity": everywhere."""
    if thr is not None:
        x = x - thr
    scale = x.abs().mean(dim=-1, keepdim=True)
    q = torch.where(x >= 0, 1.0, -1.0) * scale
    if ste_mode == "clip":
        x = torch.where(x.abs() <= ste_k * scale.detach(), x, x.detach())
    return ste(x, q)


def quant_act_low(x, acts, ste_mode="identity", ste_k=2.0, thr=None):
    """Low-bit inputs: 1 (sign, quant_act1), "t" ternary {-1, 0, +1} x mean|x| (|x| < mean|x| / 2 -> 0), 2 four
    levels {-1.5, -0.5, 0.5, 1.5} x mean|x|; the same clipped straight-through gradient as quant_act1."""
    if acts == 1:
        return quant_act1(x, ste_mode, ste_k, thr)
    if thr is not None:
        x = x - thr
    scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    if acts == "t":
        q = (x / scale).round().clamp(-1, 1) * scale
    else:
        q = ((x / scale).floor().clamp(-2, 1) + 0.5) * scale
    if ste_mode == "clip":
        x = torch.where(x.abs() <= ste_k * scale.detach(), x, x.detach())
    return ste(x, q)


def hadamard(n):
    """Orthonormal Sylvester-Hadamard matrix of size n (a power of 2)."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


class QLinear(nn.Linear):
    def __init__(self, n_in, n_out, kind, acts=8, act_opts=None, rotate=False):
        super().__init__(n_in, n_out, bias=False)
        self.kind, self.acts = kind, acts
        # rotate: the input is multiplied by a block Hadamard (largest power-of-2 block dividing n_in) before it is
        # quantized; the weights learn in the rotated basis (BitNet v2's H-BitLinear, QuEST)
        self.had_blk = n_in & -n_in if rotate else 0
        if rotate:
            self.register_buffer("had", hadamard(self.had_blk), persistent=False)
        self.act_opts = dict(act_opts or {})
        learn_thr = self.act_opts.pop("learn_thr", False)
        self.thr = nn.Parameter(torch.zeros(n_in)) if acts != 8 and learn_thr else None

    def forward(self, x):
        if self.kind == "fp32":
            return F.linear(x, self.weight)
        if self.had_blk:
            x = (x.reshape(*x.shape[:-1], -1, self.had_blk) @ self.had.to(x.dtype)).reshape(x.shape)
        xq = quant_act8(x) if self.acts == 8 else quant_act_low(x, self.acts, thr=self.thr, **self.act_opts)
        return F.linear(xq, quant_weight(self.weight, self.kind))


# ---- model ---------------------------------------------------------------------------------------------------------

def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, kind, acts=8, act_opts=None, a8_inputs=(), w2_acts=None, ffn_act="silu",
                 rotate=()):
        super().__init__()
        self.heads = heads
        self.gate = F.relu if ffn_act == "relu" else F.silu
        self.n1, self.n2 = nn.RMSNorm(dim, eps=1e-5), nn.RMSNorm(dim, eps=1e-5)

        def lin(name, n_in, n_out):
            return QLinear(n_in, n_out, kind, 8 if name in a8_inputs else acts, act_opts, name in rotate)
        self.q, self.k, self.v, self.o = (lin(n, dim, dim) for n in ("q", "k", "v", "o"))
        self.w1, self.w3 = lin("w1", dim, hidden), lin("w3", dim, hidden)
        self.w2 = lin("w2", hidden, dim)
        if w2_acts is not None and acts != 8 and "w2" not in a8_inputs:
            self.w2.acts = w2_acts

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = (m(h).view(b, t, self.heads, -1).transpose(1, 2) for m in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.w2(self.gate(self.w1(h)) * self.w3(h))


class TinyLM(nn.Module):
    def __init__(self, vocab, dim, layers, heads, hidden, seq, kind, head="fp32", int8_blocks=(), acts=8,
                 act_opts=None, a8_inputs=(), w2_acts=None, ffn_act="silu", rotate=(), a8_blocks=()):
        super().__init__()
        self.head_kind = head
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, hidden, "int8" if i in int8_blocks else kind, acts, act_opts,
                                          a8_inputs if not a8_blocks or i in a8_blocks else (), w2_acts, ffn_act,
                                          rotate) for i in range(layers))
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        hd = dim // heads
        freq = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(seq).float(), freq)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for blk in self.blocks:
            for m in (blk.o, blk.w2):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * layers))

    def forward(self, idx):
        t = idx.shape[1]
        x = F.embedding(idx, quant_weight(self.emb.weight, self.head_kind)) \
            if self.head_kind in ("int4", "int2") else self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[:t], self.sin[:t])
        if self.head_kind in ("binary", "int4", "int2"):
            return F.linear(quant_act8(self.norm(x)), quant_weight(self.emb.weight, self.head_kind))
        return F.linear(self.norm(x), self.emb.weight)


# ---- data, evaluation, sampling ------------------------------------------------------------------------------------

def load_split(name, device):
    arr = np.fromfile(os.path.join(DATA, f"{name}.bin"), dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int16)).to(device)


def batch(data, n, seq, gen):
    start = torch.randint(0, len(data) - seq - 1, (n,), device=data.device, generator=gen)
    idx = start[:, None] + torch.arange(seq + 1, device=data.device)
    w = data[idx].long()
    return w[:, :-1], w[:, 1:]


@torch.no_grad()
def evaluate(model, data, seq, n_batches, bsz):
    """Mean loss over fixed, non-overlapping windows from the start of the split (the same windows every time)."""
    model.eval()
    usable = min(n_batches * bsz, (len(data) - 1) // seq)
    total, count = 0.0, 0
    for i in range(0, usable, bsz):
        rows = torch.arange(i, min(i + bsz, usable), device=data.device)
        idx = rows[:, None] * seq + torch.arange(seq + 1, device=data.device)
        w = data[idx].long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(w[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1),
                                 reduction="sum").item()
        count += w[:, 1:].numel()
    model.train()
    return total / count


@torch.no_grad()
def sample(model, sp, prompt, seq, n_tokens, device):
    model.eval()
    ids = [sp.bos_id()] + sp.encode(prompt)
    for _ in range(n_tokens):
        x = torch.tensor([ids[-seq:]], device=device)
        nxt = int(model(x)[0, -1].argmax())
        if nxt == sp.bos_id():
            break
        ids.append(nxt)
    model.train()
    return sp.decode(ids[1:])


# ---- training ------------------------------------------------------------------------------------------------------

W2_ACTS = {"same": None, "1": 1, "t": "t", "2": 2}


def run(args, seed):
    import sentencepiece as spm
    device = "cuda"
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    train, val = load_split("train", device), load_split("val", device)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(DATA, "tok4096.model"))
    model = TinyLM(sp.get_piece_size(), args.dim, args.layers, args.heads, args.hidden, args.seq,
                   args.weights, args.head, tuple(args.int8_blocks), args.acts,
                   {"ste_mode": args.ste, "ste_k": args.ste_k, "learn_thr": args.learn_thr},
                   tuple(args.a8_inputs), W2_ACTS[args.w2_acts], args.ffn_act, tuple(args.hadamard),
                   tuple(args.a8_blocks)).to(device)
    if args.init_from:
        init = args.init_from.format(seed=seed)
        miss, extra = model.load_state_dict(torch.load(init, map_location=device)["model"], strict=False)
        print(f"initialized from {init} (missing {len(miss)}: {miss[:3]}, unexpected {len(extra)})")
    teacher = None
    if args.distill:
        tpath = args.distill.format(seed=seed)
        tck = torch.load(tpath, map_location=device, weights_only=False)
        ta = tck["args"]
        teacher = TinyLM(sp.get_piece_size(), ta["dim"], ta["layers"], ta["heads"], ta["hidden"], ta["seq"],
                         ta["weights"], ta.get("head", "fp32"), tuple(ta.get("int8_blocks", [])), ta.get("acts", 8),
                         {"ste_mode": ta.get("ste", "identity"), "ste_k": ta.get("ste_k", 2.0),
                          "learn_thr": ta.get("learn_thr", False)}, tuple(ta.get("a8_inputs", [])),
                         W2_ACTS[ta.get("w2_acts", "same")], ta.get("ffn_act", "silu"),
                         tuple(ta.get("hadamard", [])), tuple(ta.get("a8_blocks", []))).to(device)
        teacher.load_state_dict(tck["model"])
        teacher.eval().requires_grad_(False)
        print(f"distilling from {tpath} (alpha {args.distill_alpha}, temperature {args.distill_t})")
    n_lin = sum(m.weight.numel() for m in model.modules() if isinstance(m, QLinear))
    n_all = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    step_fn = torch.compile(model) if args.compile else model
    in_blocks = {id(p) for blk in model.blocks for p in blk.parameters()}
    clip_groups = [list(blk.parameters()) for blk in model.blocks] + [
        [p for p in model.parameters() if id(p) not in in_blocks]]
    name = f"{args.weights}_{args.name}_s{seed}"
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    print(f"{name}: {n_all / 1e6:.2f} M parameters, {n_lin / 1e6:.2f} M in the quantized linear layers")

    hist, best, t0 = [], float("inf"), time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(train, args.batch, args.seq, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = step_fn(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits = teacher(x)
            t = args.distill_t
            kl = F.kl_div(F.log_softmax(logits.float().reshape(-1, logits.shape[-1]) / t, -1),
                          F.log_softmax(t_logits.float().reshape(-1, t_logits.shape[-1]) / t, -1),
                          log_target=True, reduction="batchmean") * t * t
            loss = (1 - args.distill_alpha) * loss + args.distill_alpha * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_groups:
            for group in clip_groups:
                torch.nn.utils.clip_grad_norm_(group, 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate(model, val, args.seq, args.eval_batches, args.batch)
            hist.append({"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(vl, 4),
                         "minutes": round((time.time() - t0) / 60, 2)})
            print(f"  step {step:6d}  train {loss.item():.3f}  val {vl:.3f}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
                           os.path.join(out, "model.pt"))

    ckpt = torch.load(os.path.join(out, "model.pt"))
    model.load_state_dict(ckpt["model"])
    res = {"weights": args.weights, "seed": seed, "args": vars(args),
           "parameters": n_all, "quantized_linear_parameters": n_lin,
           "tokens_seen": args.steps * args.batch * args.seq,
           "best_step": ckpt["step"],
           "val_loss": round(best, 4), "val_perplexity": round(math.exp(best), 3),
           "minutes": round((time.time() - t0) / 60, 2), "history": hist,
           "sample": sample(model, sp, "Once upon a time", args.seq, 200, device)}
    if args.test:
        tl = evaluate(model, load_split("test", device), args.seq, 10 ** 9, args.batch)
        res.update(test_loss=round(tl, 4), test_perplexity=round(math.exp(tl), 3))
    with open(os.path.join(out, "train.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"  best val loss {best:.4f} (perplexity {math.exp(best):.2f}) at step {res['best_step']}")
    print(f"  sample: {res['sample'][:300]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", choices=["fp32", "ternary", "binary"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--head", choices=["fp32", "binary", "int4", "int2"], default="fp32", help="output layer weights (see above)")
    ap.add_argument("--int8-blocks", type=int, nargs="*", default=[], help="blocks with int8 weights (see above)")
    ap.add_argument("--acts", type=int, choices=[8, 1], default=8, help="bits of the blocks' linear inputs (see above)")
    ap.add_argument("--ste", choices=["identity", "clip"], default="identity",
                    help="--acts 1 gradient: everywhere, or only where |x| <= --ste-k x mean|x| (see above)")
    ap.add_argument("--ste-k", type=float, default=2.0)
    ap.add_argument("--learn-thr", action="store_true", help="--acts 1: learned per-channel threshold before sign")
    ap.add_argument("--a8-inputs", nargs="*", default=[], choices=["q", "k", "v", "o", "w1", "w3", "w2"],
                    help="--acts 1: these layers keep 8-bit inputs")
    ap.add_argument("--a8-blocks", type=int, nargs="*", default=[],
                    help="--a8-inputs only in these blocks (default: every block)")
    ap.add_argument("--w2-acts", choices=list(W2_ACTS), default="same",
                    help="--acts 1: w2's input as 1 bit, ternary (t) or 2 bits (same: as --acts)")
    ap.add_argument("--hadamard", nargs="*", default=[], choices=["q", "k", "v", "o", "w1", "w3", "w2"],
                    help="block Hadamard rotation of these layers' inputs before quantization")
    ap.add_argument("--ffn-act", choices=["silu", "relu"], default="silu", help="gate activation of the FFN")
    ap.add_argument("--clip-groups", action="store_true",
                    help="clip the gradient norm per block (and the rest apart) instead of the whole model")
    ap.add_argument("--init-from", default="", help="start from this checkpoint ({seed} is replaced): two-step")
    ap.add_argument("--distill", default="", help="teacher checkpoint ({seed} is replaced): distillation")
    ap.add_argument("--distill-alpha", type=float, default=0.5, help="weight of the KL term (see above)")
    ap.add_argument("--distill-t", type=float, default=1.0, help="distillation temperature")
    ap.add_argument("--name", default="d384l8", help="run name (folder: runs/<weights>_<name>_s<seed>)")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=40, help="validation windows = this x --batch")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--test", action="store_true", help="also score the test split (once, final numbers only)")
    args = ap.parse_args()
    for seed in args.seeds:
        run(args, seed)


if __name__ == "__main__":
    main()
