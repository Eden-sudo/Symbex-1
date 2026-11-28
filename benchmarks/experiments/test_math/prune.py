"""
prune.py -- remove the hidden neurons of a binary MLP that are constant on the training data (section 7 of the
README) and score the exact integer network on validation.

A neuron whose bit is c (+1 or -1) on every training sample is removed: its contribution s * c (s = its column of
signs in the next layer) moves into the next layer's threshold (threshold - s * c), or, for the classifier, into a
fixed integer offset per class added to the sum (+ s * c) before the scale. On any input where the neuron really is
c, the pruned network gives exactly the same result; it can differ only where the neuron would have moved. Most of
them are the CDT's dead neurons (threshold +-2^31, counted apart): constant on any input, so removing them is exact.

    python prune.py --seeds 42 43 44 --out results/prune.json
    python prune.py --depth 3 --seeds 42 43 44 --out results/prune.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import damage as D                                                                            # noqa: E402
import deep_mlp                                                                            # noqa: E402
from compiler_core.export.dense import INT32_MIN                                              # noqa: E402


def forward(layers, x_u8, offset=None):
    """integer logits and the hidden bits (bool) of the folded MLP (as damage.trace_mlp, no flip)"""
    from compiler_core.export.dense import Q_ACT, Q_MULT
    act, bits = x_u8.astype(np.int64), []
    for d in layers:
        raw = act @ d["signs"].T.astype(np.int64)
        if not d["last"]:
            act = np.where(raw >= d["threshold"][None, :], 1, -1)
            bits.append(act > 0)
        else:
            if offset is not None:
                raw = raw + offset[None, :]
            sh = Q_MULT - Q_ACT
            return bits, ((raw * d["scale"][None, :] + (1 << (sh - 1))) >> sh) + d["shift"][None, :]


def train_u8(seed, depth):
    """the whole CDT training split of the run, as the 8-bit values the network reads"""
    import tomllib
    import cdt
    import deep_mlp
    from compiler_core.export.dense import to_uint8
    folder = os.path.join(D.BENCH, "mlp", "mlp_fashion")
    two = os.path.join(folder, "runs", "initfix" if deep_mlp.FIXED_INIT else "sizing", f"h128_512_s{seed}_nodistill")
    run = deep_mlp.run_dir(depth, seed) if depth else two
    cfg = tomllib.load(open(os.path.join(run, "config_used.toml"), "rb"))
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        train = cdt.load_data(cfg)[0]
    finally:
        os.chdir(cwd)
    loader = torch.utils.data.DataLoader(train, batch_size=4096)
    return np.concatenate([to_uint8(x, cfg["data"]["mean"], cfg["data"]["std"]) for x, _ in loader])


def constant_masks(layers, x):
    """per hidden layer: (always on, always off) over x"""
    on = off = None
    for i in range(0, len(x), 5000):
        bits = forward(layers, x[i:i + 5000])[0]
        a = [b.all(0) for b in bits]
        n = [(~b).all(0) for b in bits]
        on = a if on is None else [p & q for p, q in zip(on, a)]
        off = n if off is None else [p & q for p, q in zip(off, n)]
    return on, off


def prune(layers, on, off):
    """the pruned layers, the classifier offset and the weight bits saved"""
    new = [dict(d) for d in layers]
    offset, saved = None, 0
    for i in range(len(layers) - 1):
        const = on[i] | off[i]
        c = np.where(on[i], 1, -1)[const]
        nxt = new[i + 1]
        s = nxt["signs"][:, const].astype(np.int64)
        saved += int(const.sum()) * (new[i]["signs"].shape[1] + nxt["signs"].shape[0])
        if nxt["last"]:
            offset = s @ c
        else:
            nxt["threshold"] = nxt["threshold"] - s @ c
        nxt["signs"] = nxt["signs"][:, ~const]
        new[i]["signs"] = new[i]["signs"][~const]
        new[i]["threshold"] = new[i]["threshold"][~const]
    return new, offset, saved


def bias_neurons(layers, offset):
    """the classifier offset as K always-1 neurons appended to the last hidden layer (no new C code): every offset
    is a sum of +-1 over the removed neurons, so all have the parity of their count; K = the smallest value >= max
    |offset| with that parity, and class j gets (K + offset_j) / 2 weights +1 and the rest -1 on those K inputs"""
    new = [dict(d) for d in layers]
    hid, cls = new[-2], new[-1]
    k = int(np.abs(offset).max())
    if (k - int(offset[0])) % 2:
        k += 1
    assert all((k + int(o)) % 2 == 0 for o in offset)
    hid["signs"] = np.concatenate([hid["signs"], np.ones((k, hid["signs"].shape[1]), hid["signs"].dtype)])
    hid["threshold"] = np.concatenate([np.asarray(hid["threshold"]), np.full(k, INT32_MIN, np.asarray(hid["threshold"]).dtype)])
    cols = np.array([[1] * ((k + int(o)) // 2) + [-1] * ((k - int(o)) // 2) for o in offset], cls["signs"].dtype)
    cls["signs"] = np.concatenate([cls["signs"], cols.reshape(len(offset), k)], axis=1)
    return new, k


def export(layers, out_dir, xv, yv):
    """model_symbex.h + validation files of the pruned net (chip_test.py reads them), C checked on the PC"""
    from compiler_core.export import dense
    os.makedirs(out_dir, exist_ok=True)
    pred = dense.simulate(layers, xv)
    header = os.path.join(out_dir, "model_symbex.h")
    dense.write_header(layers, header)
    equal, total, msg = dense.validate_c(header, os.path.join(D.BENCH, "..", "Symbex1_PIO", "lib", "lib_symbex"), xv, pred)
    dense.save_validation(out_dir, xv, yv, pred)
    buffers = int(next(l.split()[2] for l in open(header) if "_BYTES_BUFFERS" in l))
    return dict(header=os.path.relpath(header, HERE), flash_bytes=dense.flash_bytes(layers), ram_buffers=buffers,
                c_exactness=None if equal is None else 100 * equal / total, c_note=msg, pred=pred)


def study(seed, depth):
    layers, xv, yv, ref = D.load_mlp(seed, depth)[:4]
    _, lv = forward(layers, xv)
    p0 = lv.argmax(1)
    assert (p0 == ref).all(), "the clean pass is not the CDT simulation"
    on, off = constant_masks(layers, train_u8(seed, depth))
    new, offset, saved = prune(layers, on, off)
    _, lp = forward(new, xv, offset)
    p1 = lp.argmax(1)
    on_v, off_v = constant_masks(layers, xv)
    biased, k = bias_neurons(new, offset)
    assert (forward(biased, xv)[1] == lp).all(), "the bias neurons do not reproduce the classifier offset"
    tag = ("fixinit_" if deep_mlp.FIXED_INIT else "") + f"{f'd{depth}' if depth else 'd2'}_s{seed}"
    before = export(layers, os.path.join(HERE, "runs", "pruned", f"full_{tag}"), xv, yv)
    after = export(biased, os.path.join(HERE, "runs", "pruned", f"pruned_{tag}"), xv, yv)
    assert (before.pop("pred") == p0).all() and (after.pop("pred") == p1).all()
    hidden = [len(m) for m in on]
    total_bits = sum(d["signs"].size for d in layers)
    sentinel = [int((np.abs(np.asarray(d["threshold"], dtype=np.float64)) >= 2 ** 31).sum()) for d in layers[:-1]]
    return dict(hidden=hidden, removed=[int((a | b).sum()) for a, b in zip(on, off)], sentinel=sentinel,
                constant_validation=[int((a | b).sum()) for a, b in zip(on_v, off_v)],
                acc_before=100 * float((p0 == yv).mean()), acc_after=100 * float((p1 == yv).mean()),
                changed=100 * float((p0 != p1).mean()), logits_equal=100 * float((lv == lp).all(1).mean()),
                weight_bits=total_bits, weight_bits_saved=saved, bias_neurons=k,
                widths=[d["signs"].shape[0] for d in biased[:-1]], export_full=before, export_pruned=after,
                flash_bytes_no_bias_neurons=int(__import__("compiler_core.export.dense", fromlist=["x"]).flash_bytes(new)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--depth", type=int, help="the DeepMLP_<n> runs of deep_mlp.py (default: the 2-layer MLP)")
    ap.add_argument("--out", help="JSON to add these results to (e.g. results/prune.json)")
    ap.add_argument("--fixed-init", action="store_true", help="the runs of the fixed init_thresholds (section 8)")
    args = ap.parse_args()
    if args.fixed_init:
        deep_mlp.use_fixed_init()
    res = {}
    for seed in args.seeds:
        key = f"{deep_mlp.mlp_tag(args.depth)}_s{seed}"
        r = res[key] = study(seed, args.depth)
        print(f"  {key}: removed {sum(r['removed'])}/{sum(r['hidden'])} {r['removed']} (threshold +-2^31 {r['sentinel']}, constant on validation "
              f"{r['constant_validation']}), acc {r['acc_before']:.2f} -> {r['acc_after']:.2f}, changed "
              f"{r['changed']:.3f}%, logits equal {r['logits_equal']:.2f}%, weight bits -{100 * r['weight_bits_saved'] / r['weight_bits']:.1f}%")
        f, q = r["export_full"], r["export_pruned"]
        print(f"     header: widths {r['widths']} (+{r['bias_neurons']} bias neurons), flash {f['flash_bytes']} -> "
              f"{q['flash_bytes']} B, buffers {f['ram_buffers']} -> {q['ram_buffers']} B, C exact {f['c_exactness']} / "
              f"{q['c_exactness']} % {q['c_note']}")
    if args.out:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        old.update(res)
        with open(args.out, "w") as f:
            json.dump(old, f, indent=1)
        print(f"-> {args.out} ({len(old)} entries)")


if __name__ == "__main__":
    main()
