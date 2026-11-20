"""Import a local llama2.c v0 checkpoint into a TinyLM fine-tuning run.

Usage: python import_model.py <src> <out_dir> [--weights binary|fp32]
No downloads, training, tokenizer substitution, or integer export are performed.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def local(name):
    spec = importlib.util.spec_from_file_location("tiny_cli_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_model(src, out_dir, weights="binary"):
    """Write model.pt and import.json to a new directory; never overwrite a run.

    The stock loader can represent expanded GQA, but cannot represent an untied
    classifier. Reject that case before writing anything. GGUF decoding is a
    separate reader, not a promise of architecture/tokenizer compatibility.
    """
    if weights not in ("binary", "fp32"):
        raise ValueError("weights must be binary or fp32")
    src, out = Path(src), Path(out_dir)
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    with src.open("rb") as f:
        if f.read(4) == b"GGUF":
            raise ValueError("GGUF architecture mapping is unsupported; use gguf.read_gguf for tensor inspection")
    config, tensors = local("llama2c").read_checkpoint(src)
    if not config["shared_classifier"]:
        raise ValueError("untied classifier cannot be loaded by stock train.py/intsim.py; use mapping.to_model for float inference")
    mapping = local("mapping")
    model = mapping.to_model(config, tensors)
    if weights == "binary":
        model = local("binarize").binarize_model(model)
    args = dict(dim=config["dim"], layers=config["n_layers"], heads=config["n_heads"],
                hidden=config["hidden_dim"], seq=config["seq_len"], weights=weights,
                head="fp32", acts=8)
    manifest = dict(source_format="llama2.c-v0", source_name=src.name, source_config=config,
                    args=args, vocab_size=config["vocab_size"],
                    kv_expanded=config["n_kv_heads"] != config["n_heads"],
                    trained=False, validated_on_real_checkpoint=False,
                    tokenizer="not included; supply the source tokenizer and preserve token IDs",
                    integer_attention_compatible=config["dim"] // config["n_heads"] == 64,
                    notes=["Binary initialization is lossy and requires fine-tuning.",
                           "Stock script entry points require CUDA and matching prepared data/tokenizer.",
                           "model.pt loads via train.TinyLM.load_state_dict or intsim.load_model on CPU."])
    out.mkdir(parents=True, exist_ok=False)
    mapping.torch.save({"model": model.state_dict(), "step": 0, "args": args}, out / "model.pt")
    (out / "import.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("out_dir")
    parser.add_argument("--weights", choices=("binary", "fp32"), default="binary")
    args = parser.parse_args(argv)
    try:
        result = import_model(args.src, args.out_dir, args.weights)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"import error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
