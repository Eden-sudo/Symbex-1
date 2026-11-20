"""Toy sensor pipeline, not a language-quality or hardware benchmark."""
import math

import torch

from tiny_import_helpers import load


def test_sensor_train_generate_evaluate(tmp_path):
    dataset = load("../sensor_text/make_dataset")
    tokenizer = load("../sensor_text/tokenizer")
    pre = load("../sensor_text/pretokenize")
    trainer = load("../sensor_text/train")
    decoding = load("../sensor_text/decoding")
    evaluate = load("../sensor_text/evaluate")
    splits, meta = dataset.generate_dataset([41, 9, 0], seed=31)
    assert sum(map(len, splits.values())) == 50
    dataset.write_dataset(tmp_path / "records", splits, meta)
    sp = tokenizer.train_tokenizer(tmp_path / "tokenizer", rows=41, seed=31)
    counts = pre.prepare(tmp_path / "records", tmp_path / "tokenizer/tokenizer.model", tmp_path / "data")
    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] == 0
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        history = trainer.train(tmp_path / "data", tmp_path / "run", vocab=sp.vocab_size(),
                                steps=5, batch_size=1, seed=31,
                                dim=16, layers=1, heads=2, hidden=32, seq=32)
        assert [row["step"] for row in history] == list(range(6))
        assert all(math.isfinite(row[key]) for row in history for key in ("train", "val"))
        saved = torch.load(tmp_path / "run/model.pt", weights_only=True, map_location="cpu")
        model, config = trainer.build_model(**saved["config"])
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        row = next(row for row in splits["val"] if row["event"]["source"] == "har_ultracompact")
        event = row["event"]
        # NumberCopy requires an explicit slot, not automatic number detection.
        # Template supplies prose; each number token is selected from model logits.
        lead = f"Activity {event['class']} lasted "
        context = [sp.bos_id(), *sp.encode(decoding.prompt.serialize(event) + lead)]
        copy = decoding.NumberCopy(event, sp, field="duration_seconds")
        prefix = []
        for _ in range(max(map(len, copy.sequences))):
            with torch.no_grad():
                logits = model(torch.tensor([context[-config["seq"]:]]))[0, -1]
            assert torch.isfinite(logits).all()
            scores = (logits.double() * (1 << 20)).round().to(torch.int64).numpy()
            token = copy.choose(scores, prefix)
            assert token in copy.allowed(prefix)
            prefix.append(token)
            context.append(token)
            if copy.complete(prefix):
                break
        assert copy.complete(prefix)
        number = sp.decode(prefix)
        assert number == f"{event['duration_seconds']} "
        sentence = lead + number + "seconds."
        report = evaluate.evaluate([dict(row, generated=sentence)])
        assert report["examples"] == 1
        assert math.isfinite(report["field_fact_accuracy"])
        assert report["hallucination_count"] == 0
        assert report["unsupported_count"] == 0
    finally:
        torch.set_num_threads(previous)
