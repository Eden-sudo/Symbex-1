"""
cross_eval.py -- a trained run evaluated with another value-cache width: is the 2-bit value cache better because of
the training or because of the 2 bits at inference? (README "Step 9d")

Loads each run's best checkpoint (train.py's model.pt), rebuilds the model from its train.json arguments with
--v-bits replaced, and scores the same validation windows train.py scores (evaluate(), --eval-batches x --batch).
The run's own width reproduces its train.json val_loss (checked, printed).

    python cross_eval.py runs/binary_v512d64l5kv4_lr3e-3_20k_k1tr_s42 ... --v-bits 8 2
"""
import argparse
import json
import os

import torch

import train as T

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--v-bits", type=int, nargs="+", default=[8, 2])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "v512d64l5kv4_cross_eval.json"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    vals = {}
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        tj = json.load(open(os.path.join(run, "train.json")))
        a = T.run_args(tj["args"])
        if a.vocab not in vals:
            vals[a.vocab] = T.load_split("val", dev, a.vocab)
        sd = torch.load(os.path.join(run, "model.pt"), map_location=dev)["model"]
        row = {"trained_v_bits": a.v_bits, "train_json_val_loss": tj["val_loss"], "val_loss": {}}
        for vb in args.v_bits:
            a.v_bits = vb
            torch.manual_seed(0)
            model = T.build_model(a, 512 if a.vocab == 512 else 4096).to(dev)
            model.load_state_dict(sd)
            row["val_loss"][str(vb)] = round(T.evaluate(model, vals[a.vocab], a.seq, a.eval_batches, a.batch), 4)
        a.v_bits = row["trained_v_bits"]
        own = row["val_loss"].get(str(a.v_bits))
        print(f"{name}: trained v{a.v_bits} (train.json {tj['val_loss']}, here {own})  "
              + "  ".join(f"v{k} {v}" for k, v in row["val_loss"].items()), flush=True)
        res[name] = row
    json.dump(res, open(args.out, "w"), indent=1)
    print("[*] ->", args.out)


if __name__ == "__main__":
    main()
