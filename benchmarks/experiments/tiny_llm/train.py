"""
train.py -- does a tiny language model still write stories with 1-bit or ternary weights? A Llama-style transformer
(RMSNorm, rotary positions, SwiGLU, no biases) trained on TinyStories (prepare.py) with one of three weight formats,
same recipe for all:

    python train.py --weights fp32    --seeds 42 43 44
    python train.py --weights ternary --seeds 42 43 44     # {-1, 0, +1} x one scale per output row (BitNet b1.58)
    python train.py --weights binary  --seeds 42 43 44     # {-1, +1} x one scale per output row
    python train.py --weights ternary --steps 200 --name smoke   # quick check of the whole script

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
    return ste(w, torch.where(w >= 0, 1.0, -1.0) * scale)


def quant_act8(x):
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return ste(x, (x * scale).round().clamp(-128, 127) / scale)


class QLinear(nn.Linear):
    def __init__(self, n_in, n_out, kind):
        super().__init__(n_in, n_out, bias=False)
        self.kind = kind

    def forward(self, x):
        if self.kind == "fp32":
            return F.linear(x, self.weight)
        return F.linear(quant_act8(x), quant_weight(self.weight, self.kind))


# ---- model ---------------------------------------------------------------------------------------------------------

def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, kind):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(dim, eps=1e-5), nn.RMSNorm(dim, eps=1e-5)
        self.q, self.k, self.v, self.o = (QLinear(dim, dim, kind) for _ in range(4))
        self.w1, self.w3 = QLinear(dim, hidden, kind), QLinear(dim, hidden, kind)
        self.w2 = QLinear(hidden, dim, kind)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = (m(h).view(b, t, self.heads, -1).transpose(1, 2) for m in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class TinyLM(nn.Module):
    def __init__(self, vocab, dim, layers, heads, hidden, seq, kind):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, hidden, kind) for _ in range(layers))
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
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[:t], self.sin[:t])
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

def run(args, seed):
    import sentencepiece as spm
    device = "cuda"
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    train, val = load_split("train", device), load_split("val", device)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(DATA, "tok4096.model"))
    model = TinyLM(sp.get_piece_size(), args.dim, args.layers, args.heads, args.hidden, args.seq,
                   args.weights).to(device)
    n_lin = sum(m.weight.numel() for m in model.modules() if isinstance(m, QLinear))
    n_all = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    step_fn = torch.compile(model) if args.compile else model
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
        opt.zero_grad(set_to_none=True)
        loss.backward()
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
