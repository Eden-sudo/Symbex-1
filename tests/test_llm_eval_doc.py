import importlib.util
import json
from pathlib import Path
import re

from test_llm_eval_loss import make_run

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm"


def load(name):
    spec = importlib.util.spec_from_file_location("doc_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_documented_json_examples_and_links(tmp_path, capsys):
    text = (ROOT / "eval.md").read_text()
    examples = [json.loads(s) for s in re.findall(r"```json\n(.*?)\n```", text, re.S)]
    answers, summary = examples
    cases = json.loads((ROOT / "eval_conversations.json").read_text())
    assert load("eval_conversation").score(cases, answers)["macro"]["fact_recall"] == 1
    source, output = tmp_path / "input.json", tmp_path / "summary.json"
    source.write_text(json.dumps(summary))
    load("eval_results").main([str(source), str(output)])
    assert json.loads(output.read_text())["metrics"]["loss"]["mean"] == 2
    for link in re.findall(r"\]\(([^)]+)\)", text):
        assert (ROOT / link).is_file()
    for name in re.findall(r"`(test_llm_eval_\w+\.py)`", text):
        assert (ROOT.parents[2] / "tests" / name).is_file()
    answer_path = tmp_path / "answers.json"
    answer_path.write_text(json.dumps(answers))
    load("eval_conversation").main([str(ROOT / "eval_conversations.json"), str(answer_path)])
    assert json.loads(capsys.readouterr().out)["macro"]["length_ok"] == 1


def test_documented_model_commands_on_toy_run(tmp_path, capsys):
    _, tokens = make_run(tmp_path)
    for backend in ("float", "intsim"):
        load("eval_loss").main([str(tmp_path), str(tokens), "--max-tokens", "5", "--backend", backend])
        assert json.loads(capsys.readouterr().out)["tokens"] == 5
    load("eval_agreement").main([str(tmp_path), str(tokens), "--windows", "2", "--max-tokens", "5"])
    assert json.loads(capsys.readouterr().out)["windows"] == 2
    output = tmp_path / "determinism.json"
    load("eval_determinism").main([str(tmp_path), str(output), "--prompt", "1", "2",
                                  "--steps", "3", "--seed", "42", "--temperature", "16"])
    assert json.loads(output.read_text())["equal"]
